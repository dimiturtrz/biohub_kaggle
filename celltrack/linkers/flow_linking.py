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
from celltrack.linkers.admissible_affinity import AdmissibleAffinity
from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import LINK_REWARD as _LINK_REWARD
from celltrack.linkers.boundary_prior import BoundaryFactorSource
from celltrack.linkers.evidence_ramp import EvidenceRamp
from celltrack.linkers.motion_prediction import MotionPrediction
from celltrack.linkers.mutual_bonus import MutualBonus
from celltrack.linkers.ranker_bonus import RankerBonus
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

    That charge is flat by default — the same wherever and whenever a track ends. An optional `BoundaryPrior`
    makes it position- and time-aware, discounting a boundary the observation window itself explains. An
    optional `MotionPrediction` changes the other half of the cost, replacing the raw distance a pair competes
    on with a motion-predicted one while the gate keeps reading the raw separation.
    """

    spacing: Spacing
    max_distance_um: float
    affinity: EdgeAffinity | None = None
    affinity_bonus: float = 0.0
    boundary_cost: float = 0.0
    # `None` charges a track's START the same as its END, which is the shipped behaviour. Set it to price them
    # apart: a track can begin mid-movie for ordinary reasons — a cell enters the volume, a division makes a
    # daughter, the annotation starts — whereas a track ENDING mid-movie is the rarer event, so one price forces
    # the solver to treat "a cell appeared" and "a cell vanished" as equally implausible.
    appearance_cost: float | None = None
    # Off by default: the same mutual-agreement admission filter `AssignmentLinker` takes. A transition the gate
    # excludes is simply absent from the network, so the flow must route the track through an admitted edge.
    agreement: AgreementGate | None = None
    # Off by default: the same second cost term `AssignmentLinker` takes (`MutualBonus`) — the fused probability
    # added to the transition cost as evidence, so preference and agreement both price an arc.
    mutual: MutualBonus | None = None
    # Off by default: the same third cost term `AssignmentLinker` takes (`RankerBonus`) — the CC0
    # local-association re-ranker's probability, subtracted like the other two so the transition price reads
    # the linking-state evidence the pairwise affinity cannot see.
    ranker: RankerBonus | None = None
    # Off by default (`None` = one flat price everywhere, wherever and whenever a track ends). Set, it discounts
    # the boundary arcs of detections whose appearance or disappearance the observation window already explains —
    # see `BoundaryPrior`. It is purely a per-arc PRICE: the arcs, the gate and the transition costs are untouched.
    boundary: BoundaryFactorSource | None = None
    # Off by default (`None` = the shipped raw-distance cost, byte for byte). Set, the `distance` term of the
    # transition cost becomes the MOTION-PREDICTED distance — how far the target is from where the source was
    # heading (`MotionPrediction`) — while the gate keeps reading the raw one, since admissibility is a physical
    # statement about travel and a prediction is an estimate. The learned terms are untouched: this changes only
    # the geometry they are corrections to.
    prediction: MotionPrediction | None = None
    # Off by default (`None` = one flat evidence weight at every separation, the shipped cost byte for byte).
    # The same fourth option `AssignmentLinker` takes (`EvidenceRamp`): the forward affinity's weight becomes a
    # function of the candidate's separation, normalised by the mean separation this gap's gate admitted, so the
    # evidence budget is spent where the head out-separates proximity. The gate is READ for that normalisation
    # (the mean is over admitted pairs), never changed by it.
    ramp: EvidenceRamp | None = None
    # Whether the priced probability is conditioned on the ADMISSIBLE parents (`AdmissibleAffinity`).
    admissible: bool = False

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
        appearance, disappearance = self._boundary_costs(positions_um, timepoints)
        graph: nx.DiGraph[_Node] = nx.DiGraph()
        for row in range(count):
            graph.add_edge(("in", row), ("out", row), capacity=1, weight=0)  # 1-to-1: at most one link through a cell
            graph.add_edge(_SOURCE, ("in", row), capacity=1, weight=appearance[row])  # appearance
            graph.add_edge(("out", row), _SINK, capacity=1, weight=disappearance[row])  # disappearance
        for source_row, target_row, weight in self._transitions(positions_um, timepoints):
            graph.add_edge(("out", source_row), ("in", target_row), capacity=1, weight=weight)
        graph.add_edge(_SINK, _SOURCE, capacity=count, weight=0)  # return arc closes the circulation
        return graph

    def _boundary_costs(
        self, positions_um: Float[np.ndarray, "n 3"], timepoints: Int[np.ndarray, "n"]
    ) -> tuple[list[int], list[int]]:
        """The integer appearance and disappearance charge per detection — flat, or discounted by the prior.

        Without a prior every detection pays the same `boundary_cost`, which is the shipped behaviour. With one,
        the flat charge is scaled by that detection's factor; the prior's margin is this linker's own gate, so the
        band of cheap births is exactly the region a cell could have entered from outside within one frame gap.
        """
        leaving = round(_COST_SCALE * self.boundary_cost)
        entering = leaving if self.appearance_cost is None else round(_COST_SCALE * self.appearance_cost)
        if self.boundary is None or len(timepoints) == 0:
            return [entering] * len(timepoints), [leaving] * len(timepoints)
        factors = self.boundary.factors(self.spacing, self.max_distance_um, positions_um, timepoints)
        return (
            [round(entering * factor) for factor in factors.appearance.tolist()],
            [round(leaving * factor) for factor in factors.disappearance.tolist()],
        )

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
            gap = int(timepoint)
            distance = cdist(positions_um[sources], positions_um[targets])
            priced = (
                distance
                if self.prediction is None
                else self.prediction.distances(gap, sources, targets, timepoints, positions_um)
            )
            cost = self._gated_cost(gap, distance, priced)
            source_local, target_local = np.nonzero(np.isfinite(cost))
            transitions.extend(
                (int(sources[s]), int(targets[t]), round(float(_COST_SCALE * cost[s, t])))
                for s, t in zip(source_local, target_local, strict=True)
            )
        return transitions

    def _priced_affinity(self) -> EdgeAffinity | None:
        """The affinity that actually prices the cost, or None when no learned term is in play.

        The two ways of having no learned term must behave identically, and they used to not: the link reward
        switched on `affinity is None`, so a mounted affinity at `affinity_bonus = 0` fell between the modes —
        prob-driven selection with no probability to drive it. Every arc then cost a positive distance, nothing
        linked, and the short-track filter deleted the whole graph, so a sweep to zero read 0.0000 and looked
        like "the affinity is everything" when it was an empty graph. A knob's zero must be its limiting case.
        """
        return self.affinity if self.affinity_bonus != 0.0 else None

    def _gated_cost(
        self, timepoint: int, distance: Float[np.ndarray, "s t"], priced: Float[np.ndarray, "s t"]
    ) -> Float[np.ndarray, "s t"]:
        """`priced - bonus*P` in-gate, infinite beyond it — the assignment linker's transition cost.

        Two distances, deliberately: `distance` is the RAW separation and is what the gate reads, because the
        gate states how far a cell can physically travel in one gap; `priced` is the geometry the surviving
        pairs then compete on, which `MotionPrediction` may replace with a motion-predicted distance. They are
        the same array unless that term is wired, so the shipped cost is untouched.

        An infinite cost is how a pair is kept out of the network at all (`_transitions` keeps only the finite
        ones), so the agreement gate excludes a candidate here by the same route the distance gate does.
        """
        within_gate = distance <= self.max_distance_um
        if self.agreement is not None:
            within_gate = self.agreement.narrow(timepoint, within_gate)
        priced_by = self._priced_affinity()
        cost = priced.astype(np.float64) - (0.0 if priced_by is not None else _LINK_REWARD)
        if priced_by is not None:
            probability = priced_by.probabilities(timepoint)
            if probability is not None:
                priced_probability = AdmissibleAffinity.applied(self.admissible, probability, within_gate)
                weight = EvidenceRamp.applied(self.ramp, distance, within_gate)
                cost = cost - self.affinity_bonus * weight * priced_probability
        if self.mutual is not None:
            cost = self.mutual.discount(timepoint, cost)
        if self.ranker is not None:
            cost = self.ranker.discount(timepoint, cost)
        return np.where(within_gate, cost, np.inf)

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
