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

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float, Int
from scipy.spatial.distance import cdist

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.admissible_affinity import AdmissibleAffinity
from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import LINK_REWARD as _LINK_REWARD
from celltrack.linkers.boundary_prior import BoundaryFactorSource
from celltrack.linkers.evidence_ramp import EvidenceRamp
from celltrack.linkers.flow_solving import FlowSolve, MinCostFlowSolve, Transition
from celltrack.linkers.frame_gap import FrameGap
from celltrack.linkers.motion_prediction import MotionPrediction
from celltrack.linkers.mutual_bonus import MutualBonus
from celltrack.linkers.ranker_bonus import RankerBonus
from core.data.tracks import TrackGraph
from core.geometry import Spacing

# Micrometre-scale costs are rounded to integers for the network-simplex solver, which needs integer weights;
# a 1000x scale keeps sub-micrometre and sub-percent-probability cost differences distinct after rounding.
_COST_SCALE = 1000


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
    # WHICH ENGINE solves the network — never WHAT is solved. The default takes OR-tools where it is installed
    # and the networkx network simplex where it is not; both return the same objective and the same edge set,
    # 120x apart on the measured shape (`flow_solving`). Pin it to hold an engine fixed in a test.
    solve: FlowSolve = field(default_factory=MinCostFlowSolve.or_network_simplex)

    def link(self, detections: TrackGraph) -> TrackGraph:
        """Select the min-cost set of 1-to-1 links over the whole video, coupled through the track-boundary cost."""
        positions_um = self.spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        appearance, disappearance = self._boundary_costs(positions_um, timepoints)
        links = self.solve.selected(
            len(detections.node_ids), appearance, disappearance, self._transitions(positions_um, timepoints)
        )
        edges = self._selected_links(links, detections.node_ids)
        return TrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=edges)

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
    ) -> list[Transition]:
        """Every in-gate consecutive-frame `(source_row, target_row, integer_cost)` — the linkable transitions."""
        transitions: list[Transition] = []
        for gap in FrameGap.sweep(positions_um, timepoints):
            if len(gap.sources) == 0 or len(gap.targets) == 0:
                continue
            distance = cdist(positions_um[gap.sources], positions_um[gap.targets])
            priced = distance if self.prediction is None else self.prediction.distances(gap)
            cost = self._gated_cost(gap.timepoint, distance, priced)
            source_local, target_local = np.nonzero(np.isfinite(cost))
            transitions.extend(
                (int(gap.sources[s]), int(gap.targets[t]), round(float(_COST_SCALE * cost[s, t])))
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
    def _selected_links(links: list[tuple[int, int]], node_ids: Int[np.ndarray, "n"]) -> Int[np.ndarray, "e 2"]:
        """The selected transitions translated from detection rows back to node ids — the chosen links."""
        edges = [(int(node_ids[source]), int(node_ids[target])) for source, target in links]
        return np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
