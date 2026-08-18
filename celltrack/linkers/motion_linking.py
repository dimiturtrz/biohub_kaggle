"""Linking detections into tracks with a motion model and a two-pass gated assignment.

The nearest-neighbour baseline joins each cell to its optimal successor by raw distance, which fails exactly
where cells move fast or pack tightly: a migrating nucleus is nearer to a stationary neighbour than to its
own displaced self. This linker predicts where each cell is *going* — from the velocity that carried it into
its current position — and assigns against the predicted position, so a consistent mover keeps its track
across a large step while a jitterer stays put.

The assignment runs in two passes. The first is tight: only pairs whose raw displacement is within a small
gate compete, so confident short links are committed before anything ambiguous. The leftovers — cells that
found no close partner — get a second, looser pass, recovering the fast movers and the cells a missed
detection pushed apart, without letting that loose gate corrupt the easy links already made. Each committed
link updates the successor's velocity, so the motion estimate propagates along a track.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.frame_gap import FrameGap
from celltrack.linkers.product_of_experts import MotionDiffusionPrior, ProductOfExperts
from celltrack.linkers.ranker_bonus import RankerBonus
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_BEYOND_GATE = 1.0e9

# A per-frame drift estimated from too few committed pairs is dominated by those cells' own motion rather than
# the shared tissue shift, so the median is trusted only once enough tight links agree on a common-mode.
_MIN_DRIFT_SAMPLES = 3


# The successor is predicted half a step ahead, not a full one: a full-velocity extrapolation overshoots a
# cell that is decelerating or turning, and cells here move smoothly rather than ballistically, so a damped
# prediction tracks the real displacement more closely than a naive constant-velocity one.
# Public because `MotionPrediction` carries the same prediction into the flow linker's cost: the argument is
# about how these cells move, not about which optimiser reads it, so both must damp by one constant.
VELOCITY_DAMPING = 0.5


@dataclass(frozen=True)
class MotionHungarianLinker:
    """Two-pass optimal assignment against a damped constant-velocity prediction, gated tight then loose.

    An optional `affinity` folds a learned source->target association into the assignment: the gate stays
    pure geometry (so no candidate the physics excludes is admitted), but within the gate the cost a pair
    competes on is `distance - affinity_bonus * probability`, so geometry and learned appearance jointly
    pick the successor. A high-confidence learned link wins a tie the raw distance would have mis-broken —
    the crowding-mislink the dense movies pay. Left unset it is exactly the pure-geometry linker.

    Setting `prior` swaps the linear blend for a PRODUCT OF EXPERTS: distance and probability are read as the
    log-likelihoods of one assignment and summed (`cost = distance^2 / 2*sigma^2 - log P`), so the fitted
    `affinity_bonus` vanishes and the fusion sharpens only where an expert is confident — the ambiguous dense
    crowd stays soft. Left unset the cost is the linear `distance - affinity_bonus * P` unchanged.
    """

    spacing: Spacing
    tight_gate_um: float
    loose_gate_um: float
    affinity: EdgeAffinity | None = None
    affinity_bonus: float = 0.0
    # The motion expert's diffusion width for the product-of-experts cost. Set it to fuse distance and
    # probability as log-likelihoods instead of the linear blend; left None the cost is the linear one and the
    # gate/prediction path is untouched. sigma derives from the persistent-random-walk fit, not a tuned constant.
    prior: MotionDiffusionPrior | None = None
    # Correct a track START for whole-frame tissue drift, self-estimated per gap from that gap's own tight-pass
    # matches. Whole-frame drift is 61-92% of the mean annotated step here; a track's measured `velocity` already
    # carries it once the track has one committed link, so the ONLY place a drift-blind prediction pays for it is
    # a track start, where velocity is still zero. The tight pass commits the confident links first; the drift is
    # then the common-mode of those committed displacements (each cell's own motion is roughly zero-mean across a
    # frame, so the median displacement is the shared tissue shift), and it is added to the prediction of the
    # still-unmoved sources for the loose pass alone. No precomputed field, no tuned constant — the offset is the
    # frame's own measured drift, applied where and only where velocity has not yet supplied it. Off by default.
    use_drift: bool = False
    # Off by default: a further cost term carrying the CC0 local-association re-ranker's probability
    # (`RankerBonus`), subtracted after the forward affinity's discount exactly as on `AssignmentLinker`. This
    # is the pairing the re-ranker was trained under — its features read a motion-predicted distance and gain,
    # which are in-distribution only where the cost is itself motion-predicted.
    ranker: RankerBonus | None = None

    def link(self, detections: TrackGraph) -> TrackGraph:
        """Join each cell to its motion-predicted successor, committing tight links before loose ones."""
        positions_um = self.spacing.to_micrometres(detections.positions())
        velocity = np.zeros_like(positions_um)
        edges: list[tuple[int, int]] = []
        for gap in FrameGap.sweep(positions_um, detections.timepoints()):
            for source, target in self._match(gap, velocity):
                edges.append((int(detections.node_ids[source]), int(detections.node_ids[target])))
                velocity[target] = positions_um[target] - positions_um[source]
        stacked = np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
        return TrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=stacked)

    def _match(self, gap: FrameGap, velocity: Float[np.ndarray, "n 3"]) -> list[tuple[int, int]]:
        """Source→target row pairs for one frame gap: a tight assignment, then a loose one over the leftovers."""
        if len(gap.sources) == 0 or len(gap.targets) == 0:
            return []
        positions_um, sources, targets = gap.positions_um, gap.sources, gap.targets
        predicted = positions_um[sources] + VELOCITY_DAMPING * velocity[sources]
        cost = self._blended_cost(gap.timepoint, cdist(predicted, positions_um[targets]))
        displacement = cdist(positions_um[sources], positions_um[targets])
        tight = self._assign(cost, displacement <= self.tight_gate_um)
        if self.use_drift:
            drift = self._estimate_drift(tight, gap)
            if drift is not None:
                unmoved = ~velocity[sources].any(axis=1)
                predicted[unmoved] = predicted[unmoved] + drift
                cost = self._blended_cost(gap.timepoint, cdist(predicted, positions_um[targets]))
        loose = self._assign(cost, displacement <= self.loose_gate_um, taken=tight)
        return [(int(sources[s]), int(targets[t])) for s, t in tight + loose]

    def _estimate_drift(self, tight: list[tuple[int, int]], gap: FrameGap) -> Float[np.ndarray, "3"] | None:
        """The frame's common-mode tissue shift: the median displacement over the tight-committed pairs, or None."""
        if len(tight) < _MIN_DRIFT_SAMPLES:
            return None
        src = gap.sources[[s for s, _ in tight]]
        tgt = gap.targets[[t for _, t in tight]]
        return np.median(gap.positions_um[tgt] - gap.positions_um[src], axis=0)

    def _blended_cost(self, timepoint: int, distance: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """The distance cost fused with each learned-association term for this gap — appearance, then re-ranker."""
        cost = self._appearance_cost(timepoint, distance)
        if self.ranker is not None:
            cost = self.ranker.discount(timepoint, cost)
        return cost

    def _appearance_cost(self, timepoint: int, distance: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """Fuse motion distance with appearance — a product of experts when `prior` is set, else the linear blend."""
        if self.prior is None:
            return self._linear_cost(timepoint, distance)
        log_appearance = self._log_appearance(timepoint, distance.shape)
        return ProductOfExperts().cost(log_appearance, self.prior.log_likelihood(distance))

    def _linear_cost(self, timepoint: int, distance: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """`distance - affinity_bonus * P` — the incommensurate terms coupled by the fitted bonus."""
        cost = distance
        if self.affinity is not None and self.affinity_bonus != 0.0:
            probability = self.affinity.probabilities(timepoint)
            if probability is not None:
                cost = cost - self.affinity_bonus * probability
        return cost

    def _log_appearance(self, timepoint: int, shape: tuple[int, int]) -> Float[np.ndarray, "s t"]:
        """The appearance expert's log-probability, flat (zero) where no association is scored for the gap."""
        if self.affinity is None:
            return np.zeros(shape)
        probability = self.affinity.probabilities(timepoint)
        if probability is None:
            return np.zeros(shape)
        return ProductOfExperts().as_log_probability(probability)

    def _assign(
        self,
        cost: Float[np.ndarray, "s t"],
        within_gate: Bool[np.ndarray, "s t"],
        taken: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """The optimal one-to-one pairs within the gate, excluding rows and columns already committed."""
        used_sources = {s for s, _ in taken or []}
        used_targets = {t for _, t in taken or []}
        free = within_gate.copy()
        free[list(used_sources), :] = False
        free[:, list(used_targets)] = False
        if not free.any():
            return []
        chosen_sources, chosen_targets = linear_sum_assignment(np.where(free, cost, _BEYOND_GATE))
        return [(int(s), int(t)) for s, t in zip(chosen_sources, chosen_targets, strict=True) if free[s, t]]
