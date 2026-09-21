"""Solving the whole-video min-cost flow — one network, two engines that must agree edge for edge.

`FlowLinker` states the linking problem; this is where it is solved. The statement is the same either way —
a unit-capacity node per detection, a priced boundary arc at each end, a gated transition between consecutive
frames — so the solver is a strategy, not a branch inside the linker.

WHY TWO. networkx's `network_simplex` is pure Python and scales ~n^2 on this shape: measured 1.0s at 6k
detections, 5.9s at 18k, and the tracker logs `LinkerStage 50.2s` on a 53k-node movie, which is 97% of the
post-processing fold and ~75% of a whole tracker pass. Every training run pays that at every eval window, so
it is the binding constraint on experiment throughput. OR-tools solves the identical network in C++ at 0.05s
on the 18k shape — 120x — and the two return the SAME objective and the SAME selected edge set (see
`tests/unit/celltrack/linkers/flow_solving.py`, which asserts that equality over random shapes).

The networkx engine stays, and stays the fallback: the submission kernel bundles a fixed wheel set, so a
linker that REQUIRES a new dependency is a linker the kernel cannot run. `MinCostFlowSolve.or_network_simplex`
picks OR-tools when it imports and degrades silently when it does not.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import networkx as nx
import numpy as np
from jaxtyping import Int

logger = logging.getLogger(__name__)

_SOURCE, _SINK = "source", "sink"

# A flow node is either a boundary marker (`_SOURCE`/`_SINK`) or an `("in"|"out", detection_row)` split node.
type _Node = str | tuple[str, int]

# One linkable transition: `(source_row, target_row, integer_cost)`.
type Transition = tuple[int, int, int]


class FlowSolve(Protocol):
    """Select the min-cost set of transitions over the whole video, given the network's prices."""

    def selected(
        self,
        count: int,
        appearance: list[int],
        disappearance: list[int],
        transitions: list[Transition],
        division: int | None = None,
    ) -> list[tuple[int, int]]:
        """The `(source_row, target_row)` transitions carrying a unit of flow at the optimum.

        `division` is the SURCHARGE over a track start for a second outgoing link, or `None` for the 1-to-1
        network. A second unit entering a detection's out-node is what lets it emit two transitions, so a
        division is a parallel arc rather than a post-hoc edit; `None` omits those arcs and the network is the
        1-to-1 one, arc for arc.

        WHY A SURCHARGE AND NOT A PRICE. The extra unit is not obliged to leave beside a sibling: a detection
        with no parent can swallow it and fund its own track start, which prices an APPEARANCE at the division
        arc. Charging `appearance[row] + division` makes that route cost exactly what the appearance arc costs,
        so it is never the cheaper lie, and a real division then fires on the only thing that separates it — a
        second transition whose own cost beats the surcharge. So `division` is what a daughter must out-earn.
        """
        ...


@dataclass(frozen=True)
class NetworkSimplexSolve:
    """networkx's network simplex over the literal circulation — no dependency beyond the kernel's own set.

    A detection that carries no flow is simply absent from the solution, which is how a cell whose boundary
    prices beat every transition ends up isolated.
    """

    def selected(
        self,
        count: int,
        appearance: list[int],
        disappearance: list[int],
        transitions: list[Transition],
        division: int | None = None,
    ) -> list[tuple[int, int]]:
        """Build the circulation, solve it, and read the transition arcs that carry flow."""
        graph: nx.DiGraph[_Node] = nx.DiGraph()
        for row in range(count):
            graph.add_edge(("in", row), ("out", row), capacity=1, weight=0)  # one link INTO a cell, always
            graph.add_edge(_SOURCE, ("in", row), capacity=1, weight=appearance[row])
            graph.add_edge(("out", row), _SINK, capacity=1, weight=disappearance[row])
            if division is not None:
                graph.add_edge(_SOURCE, ("out", row), capacity=1, weight=appearance[row] + division)
        for source_row, target_row, weight in transitions:
            graph.add_edge(("out", source_row), ("in", target_row), capacity=1, weight=weight)
        graph.add_edge(_SINK, _SOURCE, capacity=2 * count, weight=0)  # the return arc closes the circulation
        _, flow = nx.network_simplex(graph)
        return [
            (node[1], target[1])
            for node, targets in flow.items()
            if isinstance(node, tuple) and node[0] == "out"
            for target, sent in targets.items()
            if isinstance(target, tuple) and target[0] == "in" and sent > 0
        ]


@dataclass(frozen=True)
class MinCostFlowSolve:
    """OR-tools' C++ min-cost flow over the same network under two rewrites that preserve the edge set.

    `SimpleMinCostFlow` takes non-negative arc costs, and a transition cost is routinely negative — that is
    what makes linking beat not linking once an affinity prices it (`distance - bonus*P`). The two rewrites
    that make the costs non-negative without moving the optimum:

    1. FORCED UNITS. Every detection emits one unit (supply +1 at its `out` node, -1 at its `in` node) rather
       than being skippable, and a BYPASS arc `out_row -> in_row` carries the unit straight through at the
       price of being isolated. A detection is still free to end up isolated; it now says so with an arc.
    2. A UNIFORM SHIFT. Exactly one arc consumes each unit at an in-node — a transition, an appearance, or a
       bypass — so adding the same constant to all three shifts the objective by exactly `shift * count`,
       whatever the solution. That is what the arithmetic at the end subtracts back out.

    Without the bypass, rewrite 1 alone is NOT edge-set-preserving: forcing a skipped detection to pay
    appearance + disappearance taxes isolation, and the solver buys links it should not. Measured at 18k
    detections that error was +1123 links; with the bypass the edge set matches networkx exactly.
    """

    def selected(
        self,
        count: int,
        appearance: list[int],
        disappearance: list[int],
        transitions: list[Transition],
        division: int | None = None,
    ) -> list[tuple[int, int]]:
        """Solve the shifted network and translate the transition arcs carrying flow back to row pairs.

        A division arc feeds the OUT-node, and the uniform shift is carried only by the arcs that feed an
        IN-node (appearance, bypass, transition). Every in-node still consumes exactly one unit — its demand is
        -1 whether or not anything divides — so the shift still displaces the objective by exactly
        `shift * count` and the division price enters unshifted. It must therefore be non-negative, which it is
        by construction: it is a penalty for the second link, not a reward.
        """
        # ortools is an optional accelerator, not a kernel dependency (see the module note), so it is imported
        # where it is used rather than at module scope, which would make this module unimportable without it.
        from ortools.graph.python import min_cost_flow  # noqa: PLC0415

        sources, targets, costs = self._arrays(transitions)
        shift = -int(min(costs.min(initial=0), 0))
        solver = min_cost_flow.SimpleMinCostFlow()
        rows, outs = np.arange(count), np.arange(count) + count
        ones = [1] * count
        source, sink = 2 * count, 2 * count + 1
        self._add_arcs(solver, np.full(count, source), rows, ones, (np.asarray(appearance) + shift).tolist())
        self._add_arcs(solver, outs, np.full(count, sink), ones, disappearance)
        self._add_arcs(solver, outs, rows, ones, [shift] * count)
        first_transition = solver.num_arcs()
        self._add_arcs(solver, sources + count, targets, [1] * len(sources), (costs + shift).tolist())
        if division is not None:
            self._add_arcs(solver, np.full(count, source), outs, ones, (np.asarray(appearance) + division).tolist())
        self._add_arcs(solver, [sink], [source], [2 * count], [0])
        self._set_supplies(solver, np.concatenate([outs, rows]), ones + [-1] * count)
        if solver.solve() != solver.OPTIMAL:
            logger.warning("min-cost flow did not solve to optimality; falling back to the network simplex")
            return NetworkSimplexSolve().selected(count, appearance, disappearance, transitions, division)
        carried = np.flatnonzero([solver.flow(arc) for arc in range(first_transition, first_transition + len(sources))])
        return [(int(sources[index]), int(targets[index])) for index in carried]

    @staticmethod
    def _nodes(values: Sequence[int] | Int[np.ndarray, " a"]) -> Int[np.ndarray, " a"]:
        """Node indices at the WIDTH OR-tools' batch API declares, not the one numpy happened to infer — int32.

        `np.arange` gives a platform int and a python list gives int64; neither is the int32 the arc arrays take,
        and jaxtyping cannot say so because it constrains the SHAPE. So every array reaching the solver is cast
        here, once.
        """
        return np.asarray(values, dtype=np.int32)

    @staticmethod
    def _prices(values: Sequence[int] | Int[np.ndarray, " a"]) -> Int[np.ndarray, " a"]:
        """Capacities, costs and supplies at the width OR-tools declares — int64, so a `_COST_SCALE`d cost fits."""
        return np.asarray(values, dtype=np.int64)

    @staticmethod
    def _add_arcs(
        solver: object,
        tails: Sequence[int] | Int[np.ndarray, " a"],
        heads: Sequence[int] | Int[np.ndarray, " a"],
        capacities: Sequence[int],
        costs: Sequence[int],
    ) -> None:
        """Add a batch of arcs, and be the ONE place the ortools stub has to be argued with.

        That stub declares the four parameters as `ndarray[signedinteger[_32Bit]]` — a one-parameter `ndarray`,
        whose single slot is really the SHAPE — so no numpy array, correctly typed or otherwise, is assignable to
        it; even spelling the declaration back verbatim is rejected as a bad specialization. Every arc batch goes
        through this function so that unsatisfiable call is written down once with its reason, instead of a
        suppression on each of the five call sites.
        """
        solver.add_arcs_with_capacity_and_unit_cost(  # pyrefly: ignore[missing-attribute]
            MinCostFlowSolve._nodes(tails),
            MinCostFlowSolve._nodes(heads),
            MinCostFlowSolve._prices(capacities),
            MinCostFlowSolve._prices(costs),
        )

    @staticmethod
    def _set_supplies(solver: object, nodes: Sequence[int] | Int[np.ndarray, " a"], supplies: Sequence[int]) -> None:
        """Declare each node's net flow — the other call across the same stub boundary `_add_arcs` describes."""
        solver.set_nodes_supplies(  # pyrefly: ignore[missing-attribute]
            MinCostFlowSolve._nodes(nodes), MinCostFlowSolve._prices(supplies)
        )

    @staticmethod
    def or_network_simplex() -> "FlowSolve":
        """This engine where ortools is installed, the network simplex where it is not — same answer, 120x apart.

        The choice is made once, when a linker is built, rather than per solve: it is a property of the
        environment (a kernel with a fixed wheel set, or a dev box), never of the network being solved.
        """
        try:
            import ortools.graph.python.min_cost_flow  # noqa: F401, PLC0415
        except ImportError:
            logger.info("ortools is not installed; linking with the networkx network simplex")
            return NetworkSimplexSolve()
        return MinCostFlowSolve()

    @staticmethod
    def _arrays(
        transitions: list[Transition],
    ) -> tuple[Int[np.ndarray, " e"], Int[np.ndarray, " e"], Int[np.ndarray, " e"]]:
        """The transition list as three aligned integer columns the solver's batch API takes."""
        if not transitions:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty
        stacked = np.array(transitions, dtype=np.int64)
        return stacked[:, 0], stacked[:, 1], stacked[:, 2]
