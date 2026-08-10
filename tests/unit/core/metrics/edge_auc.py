"""Unit tests for edge PRC-AUC — the equivalence classes of ranking affinities against a sparse GT."""

import numpy as np

from core.metrics.edge_auc import EdgeAUC


def test_of():
    """Perfect ranking scores ~1, worst ranking scores low, the active mask ignores unannotated cells."""
    gt = np.array([[1.0, 0.0], [0.0, 1.0]])

    perfect = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert EdgeAUC.of(perfect, gt) == 1.0

    worst = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert EdgeAUC.of(worst, gt) < 0.6

    inactive_ignored = _active_mask_excludes_unannotated()
    assert inactive_ignored == 1.0

    assert np.isnan(EdgeAUC.of(perfect, np.zeros((2, 2))))


def _active_mask_excludes_unannotated() -> float:
    """The cell whose row AND column the annotation never touches must not enter the ranking.

    Its score is the highest and its label is 0, so were it counted the average precision would fall below
    1.0; that it stays 1.0 proves the fully-unannotated cell is ignored.
    """
    gt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 5.0]])
    return EdgeAUC.of(scores, gt)


def test_worst_ranking_is_bounded_below():
    """Ranking every true edge last drives average precision toward the positive rate, not zero."""
    gt = np.array([[1.0, 0.0], [0.0, 1.0]])
    worst = np.array([[0.1, 0.9], [0.8, 0.2]])
    ap = EdgeAUC.of(worst, gt)
    assert 0.0 < ap < 0.6


def test_no_positives_is_nan():
    """With no active positive the average precision is undefined, like an empty edge Jaccard."""
    rng = np.random.default_rng(0)
    scores = rng.random((4, 4))
    assert np.isnan(EdgeAUC.of(scores, np.zeros((4, 4))))
