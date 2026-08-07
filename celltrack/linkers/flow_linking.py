"""A global min-cost-flow linker — one optimisation over the whole video, coupled across every frame gap.

`AssignmentLinker` is per-frame optimal: it solves each consecutive-frame gap as an independent min-cost
matching. That is provably the global optimum *when track boundaries are free* — when ending one track and
starting another costs nothing, no decision at gap `t` constrains gap `t+1`. The dense movie's residual is
crowding mislinks where that independence is exactly what fails: a per-frame matching grabs a locally-cheap
wrong successor, breaking a true track into two, and the per-frame solver cannot see that the break forces a
worse assignment a frame later. The fate decomposition (`dense_diagnosis`) measures this as the conflict
mislinks — a true target claimed by a competing source — the subset a whole-trajectory optimiser can free.

This linker prices a track boundary. Every detection is split into an in- and out-node joined by a
unit-capacity arc (so a cell carries at most one incoming and one outgoing link — the same 1-to-1 constraint
as the assignment linker); a source arc feeds every in-node at an appearance cost and every out-node drains to
a sink at a disappearance cost, and an in-gate transition `i->j` costs `distance - bonus * P(i->j)`, the
identical per-edge cost. Minimising the total over the whole graph (a min-cost circulation, network simplex)
couples the gaps through the boundary cost: a break now costs an appearance plus a disappearance, so the
optimiser keeps a true track intact across a crowded frame where the per-frame solver would sever it. At
`boundary_cost == 0` boundaries are free again and the flow decomposes back into the per-frame assignment —
the linker reduces exactly to `AssignmentLinker`, which is the A/B baseline the boundary cost is swept against.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from jaxtyping import Float, Int
from scipy.spatial.distance import cdist

from celltrack.affinity import EdgeAffinity
from core.data.tracks import TrackGraph
from core.geometry import Spacing

# Micrometre-scale costs are rounded to integers for the network-simplex solver, which needs integer weights;
# a 1000x scale keeps sub-micrometre and sub-percent-probability cost differences distinct after rounding.
_COST_SCALE = 1000
_SOURCE, _SINK = "source", "sink"

# A flow node is either a boundary marker (`_SOURCE`/`_SINK`) or an `("in"|"out", detection_row)` split node.
type _Node = str | tuple[str, int]


@dataclass(frozen=True)
class FlowLinker:
    """Whole-video min-cost-flow 1-to-1 linking with a priced track boundary — the assignment linker, coupled in time.

    Same per-edge cost and gate as `AssignmentLinker`; the difference is the `boundary_cost` charged for
    starting and ending a track, which couples the frame gaps so a globally cheaper set of longer trajectories
    can beat the per-frame-optimal one. `boundary_cost = 0` reproduces `AssignmentLinker` edge for edge.
    """

    spacing: Spacing
    max_distance_um: float
    affinity: EdgeAffinity | None = None
    affinity_bonus: float = 0.0
    boundary_cost: float = 0.0

    def link(self, detections: TrackGraph) -> TrackGraph:
        """Select the min-cost set of 1-to-1 links over the whole video, coupled through the track-boundary cost."""
        positions_um = self.spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        network = self._network(len(detections.node_ids), positions_um, timepoints)
        _, flow = nx.network_simplex(network)
        edges = self._selected_links(flow, detections.node_ids)
        return TrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=edges)

    def _network(
        self, count: int, positions_um: Float[np.ndarray, "n 3"], timepoints: Int[np.ndarray, "n"]
    ) -> "nx.DiGraph[_Node]":
        """The flow network: a unit-capacity node per detection, boundary arcs to source/sink, gated transitions."""
        boundary = round(_COST_SCALE * self.boundary_cost)
        graph: nx.DiGraph[_Node] = nx.DiGraph()
        for row in range(count):
            graph.add_edge(("in", row), ("out", row), capacity=1, weight=0)  # 1-to-1: at most one link through a cell
            graph.add_edge(_SOURCE, ("in", row), capacity=1, weight=boundary)  # appearance
            graph.add_edge(("out", row), _SINK, capacity=1, weight=boundary)  # disappearance
        for source_row, target_row, weight in self._transitions(positions_um, timepoints):
            graph.add_edge(("out", source_row), ("in", target_row), capacity=1, weight=weight)
        graph.add_edge(_SINK, _SOURCE, capacity=count, weight=0)  # return arc closes the circulation
        return graph

    def _transitions(
        self, positions_um: Float[np.ndarray, "n 3"], timepoints: Int[np.ndarray, "n"]
    ) -> list[tuple[int, int, int]]:
        """Every in-gate consecutive-frame `(source_row, target_row, integer_cost)` — the linkable transitions."""
        transitions: list[tuple[int, int, int]] = []
        for timepoint in np.unique(timepoints)[:-1]:
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            if len(sources) == 0 or len(targets) == 0:
                continue
            cost = self._gated_cost(int(timepoint), cdist(positions_um[sources], positions_um[targets]))
            source_local, target_local = np.nonzero(np.isfinite(cost))
            transitions.extend(
                (int(sources[s]), int(targets[t]), round(float(_COST_SCALE * cost[s, t])))
                for s, t in zip(source_local, target_local, strict=True)
            )
        return transitions

    def _gated_cost(self, timepoint: int, distance: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """`distance - bonus*P` in-gate, infinite beyond it — the assignment linker's transition cost."""
        cost = distance.astype(np.float64)
        if self.affinity is not None and self.affinity_bonus != 0.0:
            probability = self.affinity.probabilities(timepoint)
            if probability is not None:
                cost = cost - self.affinity_bonus * probability
        return np.where(distance <= self.max_distance_um, cost, np.inf)

    @staticmethod
    def _selected_links(flow: dict[_Node, dict[_Node, int]], node_ids: Int[np.ndarray, "n"]) -> Int[np.ndarray, "e 2"]:
        """The transition arcs carrying a unit of flow, translated from rows back to node ids — the chosen links.

        The flow dict also keys the string source/sink nodes, so a node is a transition endpoint only when it is
        an `("out", row)` / `("in", row)` tuple; every other arc (boundary, node-capacity, return) is skipped.
        """
        edges = [
            (int(node_ids[node[1]]), int(node_ids[target[1]]))
            for node, targets in flow.items()
            if isinstance(node, tuple) and node[0] == "out"
            for target, sent in targets.items()
            if isinstance(target, tuple) and target[0] == "in" and sent > 0
        ]
        return np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
