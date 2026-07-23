"""Pairing predicted cell detections with ground-truth ones, one timepoint at a time.

The competition matches nodes by centroid distance in micrometres, capped at 7 um, with an optimal
bipartite assignment per timepoint. The objective is the sum of `1 / (1 + distance)` — deliberately not
the sum of distances, so the assignment favours several close pairs over one very close pair, and the
two objectives disagree on crowded frames. The solver chain below (full bipartite matching, then the
same with empty rows and columns filled, then a dense assignment) mirrors the reference implementation
step for step, because a matcher that merely looks equivalent scores differently exactly where cells
are dense.
"""

import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.sparse as sp
from jaxtyping import Bool, Float, Int
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from core.data.tracks import TrackGraph
from core.geometry import Spacing

UNMATCHED = -1

_FILL_VALUE = -1.0
_MAX_DISTANCE_UM = 7.0


@dataclass(frozen=True)
class NodeMatching:
    """For every predicted node, the row of the ground-truth node it was paired with, or `UNMATCHED`."""

    gt_rows: Int[np.ndarray, "n"]

    def is_matched(self) -> Bool[np.ndarray, "n"]:
        """Which predicted nodes found a partner."""
        return self.gt_rows != UNMATCHED

    def matched_count(self) -> int:
        """How many predicted nodes were paired — the size of the assignment."""
        return int(self.is_matched().sum())


@dataclass(frozen=True)
class DistanceMatcher:
    """Optimal per-timepoint assignment by physical centroid distance."""

    spacing: Spacing
    max_distance_um: float = _MAX_DISTANCE_UM

    def match(self, prediction: TrackGraph, truth: TrackGraph) -> NodeMatching:
        """Pair predicted nodes with ground-truth nodes, independently within each timepoint."""
        predicted_um = self.spacing.to_micrometres(prediction.positions())
        truth_um = self.spacing.to_micrometres(truth.positions())
        gt_rows = np.full(len(prediction.node_ids), UNMATCHED, dtype=np.int64)
        for timepoint in np.unique(prediction.timepoints()):
            predicted_here = np.flatnonzero(prediction.timepoints() == timepoint)
            truth_here = np.flatnonzero(truth.timepoints() == timepoint)
            if len(truth_here) == 0:
                continue
            truth_local, predicted_local = self._assign(predicted_um[predicted_here], truth_um[truth_here])
            gt_rows[predicted_here[predicted_local]] = truth_here[truth_local]
        return NodeMatching(gt_rows=gt_rows)

    def _assign(
        self,
        predicted_um: Float[np.ndarray, "p 3"],
        truth_um: Float[np.ndarray, "g 3"],
    ) -> tuple[Int[np.ndarray, "k"], Int[np.ndarray, "k"]]:
        """Solve one frame's assignment, returning the paired `(truth, predicted)` local row indices."""
        empty = np.empty(0, dtype=np.int64)
        distances = cdist(predicted_um, truth_um)
        predicted_index, truth_index = np.nonzero(distances <= self.max_distance_um)
        if len(predicted_index) == 0:
            return empty, empty

        weights = sp.csr_array(
            (1.0 / (1.0 + distances[predicted_index, truth_index]), (truth_index, predicted_index)),
            dtype=np.float32,
        )
        try:
            return sp.csgraph.min_weight_full_bipartite_matching(weights, maximize=True)
        except ValueError:
            return self._assign_without_full_cover(weights)

    @staticmethod
    def _assign_without_full_cover(
        weights: sp.csr_array,
    ) -> tuple[Int[np.ndarray, "k"], Int[np.ndarray, "k"]]:
        """Fall back when no assignment covers every node: pad the gaps, then drop the padded pairs."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)
            DistanceMatcher._pad_gaps(weights, _FILL_VALUE)
        try:
            truth_rows, predicted_rows = sp.csgraph.min_weight_full_bipartite_matching(weights, maximize=True)
        except ValueError:
            rows, columns = cast(tuple[int, int], weights.shape)
            dense = np.full((rows, columns), _FILL_VALUE, dtype=np.float32)
            sparse = weights.tocoo()
            dense[sparse.row, sparse.col] = sparse.data
            truth_rows, predicted_rows = linear_sum_assignment(dense, maximize=True)
        real = ~np.isclose(np.asarray(weights[truth_rows, predicted_rows]), _FILL_VALUE)
        return truth_rows[real], predicted_rows[real]

    @staticmethod
    def _pad_gaps(weights: sp.csr_array, fill_value: float) -> None:
        """Give every all-zero row and column a uniform negative weight, so a full assignment exists."""
        empty_rows = weights.sum(axis=1) == 0
        empty_columns = weights.sum(axis=0) == 0
        if empty_rows.any():
            weights[empty_rows, :] = fill_value
        if empty_columns.any():
            weights[:, empty_columns] = fill_value
