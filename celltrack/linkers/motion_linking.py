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


# The published 0.926/0.927 kernels' `motion_relink_edges` competes pairs on `motion + 0.05*raw - bonus*prob`
# (evgendvorkin biohub-0-927-lb_code.py:2033), where `raw` is the RAW source->target distance and `motion` the
# damped-velocity-predicted distance. Our linear cost drops the `0.05*raw` term deliberately (see the SCALE note
# on `LinkerConfig.ranker_bonus`), so these constants live behind the `kernel_faithful` toggle rather than in the
# shipped path. The raw term's weight and the large-frame bail size are the kernel's, transcribed here.
_KERNEL_RAW_WEIGHT = 0.05
_KERNEL_MAX_FRAME_NODES = 2600
# The kernel's env-var DEFAULT for the learned bonus is 0.75, but BOTH published kernels override it to 1.0 at
# module load (biohub-0-927-lb_code.py:18, 0-926-biohub-divsub_code.py:26), so 1.0 is the constant that actually
# produced the ~0.926-0.927 LB. Faithful mode defaults to it; the field below makes it configurable regardless.
_KERNEL_LEARNED_BONUS = 1.0
# The published kernels admit an edge candidate only when its learned edge probability clears a global floor of
# 0.48 (`edge_candidate_threshold`, set from `BIOHUB_DUAL_SEED_EDGE_THRESHOLD="0.48"` at biohub-0-927-lb_code.py:1003
# into `cfg.threshold` at :1100, applied as the unchanged downstream candidate threshold per :1218). It is a floor
# on the softmax edge probability, SEPARATE from detection's own threshold and from the geometric relink gates —
# so faithful mode ANDs it onto our geometry gate rather than replacing it. Off (None) the gate is pure geometry.
_KERNEL_EDGE_ADMISSION_THRESHOLD = 0.48
# The kernel clamps a stray logit into probability: a value already in [0,1] is a probability and is kept (clipped
# to the interval); anything outside is read as a logit and passed through a sigmoid over the same [-20,20] clamp
# the kernel uses (biohub-0-927-lb_code.py:1985-1986), so a scorer that leaked logits still prices sanely.
_LOGIT_CLAMP = 20.0


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
    # OFF = our shipped cost, byte for byte. ON reproduces the published kernel's `motion_relink_edges` cost
    # EXACTLY: `motion + 0.05*raw - learned_bonus*prob` (evgendvorkin biohub-0-927-lb_code.py:2033), where `raw`
    # is the raw source->target distance our gate already computes and `motion` the damped-velocity prediction.
    # It also applies the kernel's sigmoid guard on an out-of-[0,1] probability (:1985-1986) and its >2600-node
    # large-frame bail (:1996-1998, applied per gap here — the natural analog of the kernel's whole-video skip).
    # A faithful REPLICA path kept ALONGSIDE the shipped linear cost, never a replacement: the `0.05*raw` term
    # and the absolute `learned_bonus` are the kernel's own weights, which do not transfer to our derived P-budget
    # (see the SCALE note on `LinkerConfig.ranker_bonus`) — so it reproduces the floor, it does not tune our cost.
    # `affinity_bonus`/`prior`/`ranker` are ignored while it is on.
    kernel_faithful: bool = False
    # The faithful cost's weight on the learned probability. Defaults to the constant BOTH kernels actually ran
    # (1.0, set by their module-load override of the 0.75 env default); only read when `kernel_faithful` is on.
    learned_bonus: float = _KERNEL_LEARNED_BONUS
    # OFF (None) the gate is pure geometry, byte for byte. Set to the kernel's 0.48 floor to also require every
    # admitted pair's learned edge probability to clear it — the kernel's `edge_candidate_threshold`. The floor is
    # ANDed onto the geometric gate (never a replacement); with no affinity, or a gap the affinity does not score,
    # nothing is excluded (all-admitted), so an unset scorer stays byte-identical. Read under any cost mode.
    edge_admission_prob: float | None = None

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
        if self.kernel_faithful and max(len(gap.sources), len(gap.targets)) > _KERNEL_MAX_FRAME_NODES:
            return []
        positions_um, sources, targets = gap.positions_um, gap.sources, gap.targets
        predicted = positions_um[sources] + VELOCITY_DAMPING * velocity[sources]
        displacement = cdist(positions_um[sources], positions_um[targets])
        cost = self._cost(gap.timepoint, cdist(predicted, positions_um[targets]), displacement)
        admitted = self._admitted(gap.timepoint, displacement.shape)
        tight = self._assign(cost, (displacement <= self.tight_gate_um) & admitted)
        if self.use_drift:
            drift = self._estimate_drift(tight, gap)
            if drift is not None:
                unmoved = ~velocity[sources].any(axis=1)
                predicted[unmoved] = predicted[unmoved] + drift
                cost = self._cost(gap.timepoint, cdist(predicted, positions_um[targets]), displacement)
        loose = self._assign(cost, (displacement <= self.loose_gate_um) & admitted, taken=tight)
        return [(int(sources[s]), int(targets[t])) for s, t in tight + loose]

    @staticmethod
    def _guarded_probability(probability: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """The kernel's per-value probability guard: keep an in-range value, sigmoid an out-of-range one as a logit."""
        out_of_range = (probability < 0.0) | (probability > 1.0)
        as_logit = 1.0 / (1.0 + np.exp(-np.clip(probability, -_LOGIT_CLAMP, _LOGIT_CLAMP)))
        return np.where(out_of_range, as_logit, np.clip(probability, 0.0, 1.0))

    def _admitted(self, timepoint: int, shape: tuple[int, int]) -> Bool[np.ndarray, "s t"]:
        """The kernel's edge-candidate admission mask: every pair whose learned probability clears the floor.

        All-True (nothing excluded) whenever the floor is unset, no affinity is attached, or the affinity does
        not score this gap — so the shipped pure-geometry gate is byte-identical unless the floor is engaged.
        """
        if self.edge_admission_prob is None or self.affinity is None:
            return np.ones(shape, dtype=bool)
        probability = self.affinity.probabilities(timepoint)
        if probability is None:
            return np.ones(shape, dtype=bool)
        return self._guarded_probability(probability) >= self.edge_admission_prob

    def _estimate_drift(self, tight: list[tuple[int, int]], gap: FrameGap) -> Float[np.ndarray, "3"] | None:
        """The frame's common-mode tissue shift: the median displacement over the tight-committed pairs, or None."""
        if len(tight) < _MIN_DRIFT_SAMPLES:
            return None
        src = gap.sources[[s for s, _ in tight]]
        tgt = gap.targets[[t for _, t in tight]]
        return np.median(gap.positions_um[tgt] - gap.positions_um[src], axis=0)

    def _cost(
        self,
        timepoint: int,
        motion: Float[np.ndarray, "s t"],
        raw: Float[np.ndarray, "s t"],
    ) -> Float[np.ndarray, "s t"]:
        """This gap's assignment cost: the kernel-faithful replica when toggled on, else our shipped blend."""
        if self.kernel_faithful:
            return self._faithful_cost(timepoint, motion, raw)
        return self._blended_cost(timepoint, motion)

    def _faithful_cost(
        self,
        timepoint: int,
        motion: Float[np.ndarray, "s t"],
        raw: Float[np.ndarray, "s t"],
    ) -> Float[np.ndarray, "s t"]:
        """The published kernel's exact per-pair cost: `motion + 0.05*raw - learned_bonus*prob`."""
        probability = self._faithful_probability(timepoint, motion.shape)
        return motion + _KERNEL_RAW_WEIGHT * raw - self.learned_bonus * probability

    def _faithful_probability(self, timepoint: int, shape: tuple[int, int]) -> Float[np.ndarray, "s t"]:
        """The learned probability for the gap under the kernel's guard: zero where unscored, sigmoid where raw."""
        if self.affinity is None:
            return np.zeros(shape)
        probability = self.affinity.probabilities(timepoint)
        if probability is None:
            return np.zeros(shape)
        return self._guarded_probability(probability)

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
