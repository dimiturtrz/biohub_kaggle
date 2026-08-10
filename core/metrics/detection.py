"""Node-level detection precision and recall against the sparse ground truth.

The edge Jaccard in `score.py` is the leaderboard number, but it hides *how* a detector is doing: a run can
raise it by finding more true cells (recall) or by trimming false ones (precision), and the two want opposite
moves. These counts separate them. Recall is over the annotated nodes a video actually labels (not the full
`estimated_node_count`), so it reads as "of the cells we can check, how many did we detect". Precision is over
the predictions that fall on an annotated node — over-detection in unannotated tissue is invisible here, the
same asymmetry the edge tally has, so precision is a floor, not the whole story (the node-count ratio in
`score.py` is its companion).

Matching need not be one-to-one, so recall counts *distinct* ground-truth nodes covered while precision counts
predictions that landed on one; pooling keeps the four raw tallies so a micro-average matches the leaderboard.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from core.metrics.matching import NodeMatching


@dataclass(frozen=True)
class NodeCounts:
    """The detection tallies precision and recall are built from, kept raw so they pool by summation."""

    covered_gt: int  # distinct ground-truth nodes a prediction matched — the recall numerator
    matched_pred: int  # predictions that landed on some ground-truth node — the precision numerator
    predictions: int  # all predicted nodes
    ground_truth: int  # all annotated ground-truth nodes

    @classmethod
    def of(cls, matching: NodeMatching, ground_truth_nodes: int) -> "NodeCounts":
        """Tally detection hits from a node matching and the video's annotated node count."""
        matched = matching.is_matched()
        covered = np.unique(matching.gt_rows[matched]).size if matched.any() else 0
        return cls(
            covered_gt=covered,
            matched_pred=int(matched.sum()),
            predictions=len(matching.gt_rows),
            ground_truth=ground_truth_nodes,
        )

    @classmethod
    def pooled(cls, counts: Iterable["NodeCounts"]) -> "NodeCounts":
        """Sum the raw tallies across videos, so precision/recall micro-average the way the leaderboard does."""
        summed = cls(covered_gt=0, matched_pred=0, predictions=0, ground_truth=0)
        for count in counts:
            summed = cls(
                covered_gt=summed.covered_gt + count.covered_gt,
                matched_pred=summed.matched_pred + count.matched_pred,
                predictions=summed.predictions + count.predictions,
                ground_truth=summed.ground_truth + count.ground_truth,
            )
        return summed

    def precision(self) -> float:
        """Predictions on an annotated node over all predictions, or NaN when nothing was predicted."""
        return self.matched_pred / self.predictions if self.predictions > 0 else float("nan")

    def recall(self) -> float:
        """Annotated nodes covered over all annotated nodes, or NaN when there is no ground truth."""
        return self.covered_gt / self.ground_truth if self.ground_truth > 0 else float("nan")

    def f1(self) -> float:
        """Harmonic mean of precision and recall, or 0.0 when either is zero or undefined."""
        precision, recall = self.precision(), self.recall()
        if not (precision > 0 and recall > 0):
            return 0.0
        return 2 * precision * recall / (precision + recall)
