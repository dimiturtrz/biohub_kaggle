"""Unit tests for node-level detection precision/recall — the equivalence classes of a matching."""

import numpy as np

from core.metrics.detection import NodeCounts
from core.metrics.matching import UNMATCHED, NodeMatching


def _matching(gt_rows: list[int]) -> NodeMatching:
    return NodeMatching(gt_rows=np.array(gt_rows, dtype=np.int64))


def test_of():
    """Two of three predictions hit distinct GT nodes, against four annotated nodes."""
    counts = NodeCounts.of(_matching([0, UNMATCHED, 2]), ground_truth_nodes=4)
    assert (counts.matched_pred, counts.covered_gt, counts.predictions, counts.ground_truth) == (2, 2, 3, 4)


def test_of_counts_distinct_ground_truth_for_recall():
    """Two predictions landing on the SAME GT node cover one node, not two — recall is distinct-GT."""
    counts = NodeCounts.of(_matching([1, 1, UNMATCHED]), ground_truth_nodes=2)
    assert counts.covered_gt == 1  # distinct GT covered
    assert counts.matched_pred == 2  # both predictions matched (precision numerator)


def test_pooled():
    """Pooling sums the raw tallies so precision/recall micro-average across videos."""
    pooled = NodeCounts.pooled([NodeCounts(1, 1, 2, 3), NodeCounts(2, 3, 4, 5)])
    assert (pooled.covered_gt, pooled.matched_pred, pooled.predictions, pooled.ground_truth) == (3, 4, 6, 8)


def test_precision():
    """Precision is matched predictions over all predictions; NaN when nothing was predicted."""
    assert NodeCounts(2, 3, 4, 5).precision() == 0.75
    assert np.isnan(NodeCounts(0, 0, 0, 5).precision())


def test_recall():
    """Recall is covered GT over all annotated GT; NaN when there is no ground truth."""
    assert NodeCounts(3, 4, 6, 6).recall() == 0.5
    assert np.isnan(NodeCounts(0, 0, 4, 0).recall())


def test_f1():
    """F1 is the harmonic mean, and zero when either component is zero."""
    counts = NodeCounts(covered_gt=1, matched_pred=1, predictions=2, ground_truth=2)  # p=0.5, r=0.5
    assert counts.f1() == 0.5
    assert NodeCounts(0, 0, 2, 2).f1() == 0.0
