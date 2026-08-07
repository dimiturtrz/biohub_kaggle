"""Turning a set of detections into a track graph by joining consecutive timepoints.

A linker receives nodes with no edges and returns the same nodes wired into tracks. The baseline here is
the honest floor of the family: an optimal one-to-one assignment between each pair of consecutive frames,
gated at a physical radius, minimising total travel. It is deliberately the simplest thing that could
work, so the gap between what it scores and what a learned linker scores is the budget the learned one
has to earn.

Two structural limits are worth stating rather than discovering: a one-to-one assignment gives every cell
exactly one successor, so it *cannot* represent a division (a parent with two children) and cannot split
a merged detection. Those are not tuning failures — they are the reason a division-aware linker is a
separate task, and the reason this baseline's division term is expected near zero.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from core.data.tracks import TrackGraph
from core.geometry import Spacing

_BEYOND_GATE = 1.0e9


class Linker(Protocol):
    """Wires detections into tracks. The nodes come fixed; only the edges are the linker's to decide."""

    def link(self, detections: TrackGraph) -> TrackGraph:
        """Return the detections with temporal edges added."""
        ...


@dataclass(frozen=True)
class NearestNeighbourLinker:
    """Optimal one-to-one assignment between consecutive frames, gated at a physical radius.

    The gate is wider than the metric's 7 um matching cutoff on purpose: the cost of missing a true
    successor that moved far outweighs the cost of considering a distant candidate, since the assignment
    still prefers the nearest available partner. It is not so wide that it links across a dense frame — see
    the operating-point discussion in the bracket interpretation.
    """

    spacing: Spacing
    max_distance_um: float

    def link(self, detections: TrackGraph) -> TrackGraph:
        """Join each cell to its optimal successor in the next timepoint, within the gate."""
        positions_um = self.spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        edges = [
            detections.node_ids[[source, target]]
            for timepoint in np.unique(timepoints)[:-1]
            for source, target in self._assign(
                np.flatnonzero(timepoints == timepoint),
                np.flatnonzero(timepoints == timepoint + 1),
                positions_um,
            )
        ]
        stacked = np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
        return TrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=stacked)

    def _assign(
        self,
        source_rows: Int[np.ndarray, "s"],
        target_rows: Int[np.ndarray, "t"],
        positions_um: Float[np.ndarray, "n 3"],
    ) -> list[tuple[int, int]]:
        """The optimal source→target row pairs for one frame gap, dropping any pair beyond the gate."""
        if len(source_rows) == 0 or len(target_rows) == 0:
            return []
        distances = cdist(positions_um[source_rows], positions_um[target_rows])
        within_gate = distances <= self.max_distance_um
        costs = np.where(within_gate, distances, _BEYOND_GATE)
        chosen_sources, chosen_targets = linear_sum_assignment(costs)
        return [
            (int(source_rows[source]), int(target_rows[target]))
            for source, target in zip(chosen_sources, chosen_targets, strict=True)
            if within_gate[source, target]
        ]
