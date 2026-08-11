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
from jaxtyping import Bool, Float, Int
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.ranker_bonus import RankerBonus
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_BEYOND_GATE = 1.0e9


# The successor is predicted half a step ahead, not a full one: a full-velocity extrapolation overshoots a
# cell that is decelerating or turning, and cells here move smoothly rather than ballistically, so a damped
# prediction tracks the real displacement more closely than a naive constant-velocity one.
_VELOCITY_DAMPING = 0.5


@dataclass(frozen=True)
class MotionHungarianLinker:
    """Two-pass optimal assignment against a damped constant-velocity prediction, gated tight then loose.

    An optional `affinity` folds a learned source->target association into the assignment: the gate stays
    pure geometry (so no candidate the physics excludes is admitted), but within the gate the cost a pair
    competes on is `distance - affinity_bonus * probability`, so geometry and learned appearance jointly
    pick the successor. A high-confidence learned link wins a tie the raw distance would have mis-broken —
    the crowding-mislink the dense movies pay. Left unset it is exactly the pure-geometry linker.
    """

    spacing: Spacing
    tight_gate_um: float
    loose_gate_um: float
    affinity: EdgeAffinity | None = None
    affinity_bonus: float = 0.0
    # Off by default: a further cost term carrying the CC0 local-association re-ranker's probability
    # (`RankerBonus`), subtracted after the forward affinity's discount exactly as on `AssignmentLinker`. This
    # is the pairing the re-ranker was trained under — its features read a motion-predicted distance and gain,
    # which are in-distribution only where the cost is itself motion-predicted.
    ranker: RankerBonus | None = None

    def link(self, detections: TrackGraph) -> TrackGraph:
        """Join each cell to its motion-predicted successor, committing tight links before loose ones."""
        positions_um = self.spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        velocity = np.zeros_like(positions_um)
        edges: list[tuple[int, int]] = []
        for timepoint in np.unique(timepoints)[:-1]:
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            for source, target in self._match(int(timepoint), sources, targets, positions_um, velocity):
                edges.append((int(detections.node_ids[source]), int(detections.node_ids[target])))
                velocity[target] = positions_um[target] - positions_um[source]
        stacked = np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
        return TrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=stacked)

    def _match(
        self,
        timepoint: int,
        sources: Int[np.ndarray, "s"],
        targets: Int[np.ndarray, "t"],
        positions_um: Float[np.ndarray, "n 3"],
        velocity: Float[np.ndarray, "n 3"],
    ) -> list[tuple[int, int]]:
        """Source→target row pairs for one frame gap: a tight assignment, then a loose one over the leftovers."""
        if len(sources) == 0 or len(targets) == 0:
            return []
        predicted = positions_um[sources] + _VELOCITY_DAMPING * velocity[sources]
        cost = self._blended_cost(timepoint, cdist(predicted, positions_um[targets]))
        displacement = cdist(positions_um[sources], positions_um[targets])
        tight = self._assign(cost, displacement <= self.tight_gate_um)
        loose = self._assign(cost, displacement <= self.loose_gate_um, taken=tight)
        return [(int(sources[s]), int(targets[t])) for s, t in tight + loose]

    def _blended_cost(self, timepoint: int, distance: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """The distance cost minus each learned-association term wired for this gap — forward, then re-ranker."""
        cost = distance
        if self.affinity is not None and self.affinity_bonus != 0.0:
            probability = self.affinity.probabilities(timepoint)
            if probability is not None:
                cost = cost - self.affinity_bonus * probability
        if self.ranker is not None:
            cost = self.ranker.discount(timepoint, cost)
        return cost

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
