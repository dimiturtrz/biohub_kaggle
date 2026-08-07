from pathlib import Path

import numpy as np
import pytest
import torch

from celltrack.training import edge_finetune
from celltrack.training.edge_finetune import (
    EdgeFinetuneConfig,
    EdgeHardNegativeFinetuner,
    GapSupervision,
)
from core.metrics.matching import UNMATCHED


class _StubScorer:
    """A scorer whose UNet is frozen and whose transformer is one weight, so a step provably moves that weight."""

    def __init__(self) -> None:
        self.detector = torch.nn.Linear(1, 1)
        self.transformer = torch.nn.Linear(1, 1)

    def _gap_logits(self, source, timepoint, source_positions, target_positions, device):
        pairs = torch.ones(len(source_positions), len(target_positions))
        return self.transformer.weight.reshape(()) * pairs  # (s, t) logits carrying the transformer's grad


def test_finetune(monkeypatch: pytest.MonkeyPatch):
    """One optimisation over a supervised gap moves the transformer weight (and leaves the frozen UNet untouched)."""
    monkeypatch.setattr(edge_finetune.TemporalUNetDetector, "_open_source", staticmethod(lambda path: "src"))
    gap = GapSupervision(
        timepoint=0,
        source_positions=np.zeros((2, 3), dtype=np.float32),
        target_positions=np.ones((2, 3), dtype=np.float32),
        gt_matrix=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),  # source 0 -> target 0 is the true edge
        hard_negatives=np.array([[0, 1]], dtype=np.int64),  # source 0 -> target 1 is a mined hard negative
    )
    scorer = _StubScorer()
    before = scorer.transformer.weight.detach().clone()
    detector_frozen_before = scorer.detector.weight.requires_grad

    EdgeHardNegativeFinetuner(scorer, EdgeFinetuneConfig(device="cpu", steps=3)).finetune([(Path("video"), gap)])

    assert not torch.equal(scorer.transformer.weight, before)  # the head learned
    assert detector_frozen_before and not scorer.detector.weight.requires_grad  # the UNet was frozen


def test_gt_matrix():
    """A cell of the matrix is 1 exactly where both detections match GT nodes the annotation links."""
    source_gt = np.array([10, UNMATCHED], dtype=np.int64)  # source 0 matches GT node 10; source 1 is unmatched
    target_gt = np.array([20, 21], dtype=np.int64)
    gt_edge_codes = {10 * 100 + 20}  # GT links node 10 -> node 20

    matrix = GapSupervision._gt_matrix(source_gt, target_gt, gt_edge_codes, count=100)

    assert matrix.tolist() == [[1.0, 0.0], [0.0, 0.0]]  # only (source 0 -> target 0) is an annotated edge


def test_mine_hard_negatives():
    """Each source with a true successor contributes its nearest *wrong* targets, closest first, up to the budget."""
    gt_matrix = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # source 0's true successor is target 1
    source_positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    # target 0 is nearest, target 2 next, target 1 (the true one) is far and must be excluded as a negative.
    target_positions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 9.0], [0.0, 0.0, 2.0]], dtype=np.float32)

    mined = GapSupervision._mine_hard_negatives(gt_matrix, source_positions, target_positions, keep=2)

    assert mined.tolist() == [[0, 0], [0, 2]]  # the two nearest wrong targets, nearest first; never the true one


def test_mine_hard_negatives_ignores_a_source_without_a_true_successor():
    """A source the annotation does not continue supplies no hard negatives — there is no positive to contrast."""
    gt_matrix = np.zeros((1, 3), dtype=np.float32)
    positions = np.zeros((1, 3), dtype=np.float32)
    targets = np.ones((3, 3), dtype=np.float32)

    assert GapSupervision._mine_hard_negatives(gt_matrix, positions, targets, keep=2).shape == (0, 2)


def test_frontier_loss():
    """The focal-BCE (softmax over sources) is far lower when the logits already place mass on the true parent."""
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])  # source 0 is the parent of target 0
    right = EdgeHardNegativeFinetuner._frontier_loss(torch.tensor([[9.0, 0.0], [-9.0, 0.0]]), target)  # src0 wins col0
    wrong = EdgeHardNegativeFinetuner._frontier_loss(torch.tensor([[-9.0, 0.0], [9.0, 0.0]]), target)  # other src wins

    assert right < wrong
    assert right < 0.1
