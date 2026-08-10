"""Edge PRC-AUC — how well predicted source->target affinities rank against the annotated edges.

The joint trainer selects on val-loss, which leaves the association model's quality invisible: a falling
loss says nothing about whether the true parent of each cell is being scored above its many rivals. This
metric makes that quality one number. Each cell `(i, j)` of the affinity matrix is a candidate carrying a
predicted score and a 0/1 label (1 where the annotation links source `i` to target `j`); ranking every
candidate by score and integrating the precision-recall curve gives the average precision. PRC-AUC rather
than ROC-AUC because the positives are sparse — each target has ~one true parent among many candidates — and
average precision reads that regime honestly where ROC-AUC is flattered by the abundant true negatives.

Only cells in *active* rows and columns count, mirroring the loss's `active` mask: a source row the
annotation gives a successor, or a target column it gives a parent. Everywhere else the ground truth is
silent, so an unannotated candidate is neither a true nor a false positive — it is ignored, the same
sparse-annotation asymmetry the edge tally in `edges.py` has. With no positives the average precision is
undefined and returns NaN, as `EdgeCounts.jaccard` does on an empty tally.
"""

import numpy as np
from jaxtyping import Float


class EdgeAUC:
    """The edge association model's ranking quality, as the area under its precision-recall curve."""

    @staticmethod
    def of(scores: Float[np.ndarray, "s t"], gt_matrix: Float[np.ndarray, "s t"]) -> float:
        """Average precision of the affinity ranking over the annotated (active) cells, or NaN if none.

        A cell is active when its source row or its target column carries a ground-truth edge; the average
        precision is computed only over those candidates, ranked by score. Returns NaN when no active cell
        is a positive, matching how the edge Jaccard is undefined on an empty ground truth.
        """
        labels = gt_matrix > 0
        active = labels.any(axis=1, keepdims=True) | labels.any(axis=0, keepdims=True)
        return EdgeAUC._average_precision(scores[active], labels[active])

    @staticmethod
    def _average_precision(
        scores: Float[np.ndarray, "n"],
        labels: Float[np.ndarray, "n"],
    ) -> float:
        """Area under the precision-recall curve for scored, labelled candidates (average precision)."""
        positives = int(labels.sum())
        if positives == 0:
            return float("nan")
        order = np.argsort(scores, kind="stable")[::-1]
        ranked = labels[order].astype(np.float64)
        true_positives = np.cumsum(ranked)
        precision = true_positives / np.arange(1, ranked.size + 1)
        recall = true_positives / positives
        recall_gain = np.diff(recall, prepend=0.0)
        return float((precision * recall_gain).sum())
