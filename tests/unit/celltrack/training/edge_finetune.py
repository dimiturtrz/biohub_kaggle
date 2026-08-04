import numpy as np
import torch

from celltrack.training.edge_finetune import _frontier_loss, _gt_matrix, _mine_hard_negatives
from core.metrics.matching import UNMATCHED


def test_gt_matrix():
    """A cell of the matrix is 1 exactly where both detections match GT nodes the annotation links."""
    source_gt = np.array([10, UNMATCHED], dtype=np.int64)  # source 0 matches GT node 10; source 1 is unmatched
    target_gt = np.array([20, 21], dtype=np.int64)
    gt_edge_codes = {10 * 100 + 20}  # GT links node 10 -> node 20

    matrix = _gt_matrix(source_gt, target_gt, gt_edge_codes, count=100)

    assert matrix.tolist() == [[1.0, 0.0], [0.0, 0.0]]  # only (source 0 -> target 0) is an annotated edge


def test_mine_hard_negatives():
    """Each source with a true successor contributes its nearest *wrong* targets, closest first, up to the budget."""
    gt_matrix = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # source 0's true successor is target 1
    source_positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    # target 0 is nearest, target 2 next, target 1 (the true one) is far and must be excluded as a negative.
    target_positions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 9.0], [0.0, 0.0, 2.0]], dtype=np.float32)

    mined = _mine_hard_negatives(gt_matrix, source_positions, target_positions, keep=2)

    assert mined.tolist() == [[0, 0], [0, 2]]  # the two nearest wrong targets, nearest first; never the true one


def test_mine_hard_negatives_ignores_a_source_without_a_true_successor():
    """A source the annotation does not continue supplies no hard negatives — there is no positive to contrast."""
    gt_matrix = np.zeros((1, 3), dtype=np.float32)
    positions = np.zeros((1, 3), dtype=np.float32)
    targets = np.ones((3, 3), dtype=np.float32)

    assert _mine_hard_negatives(gt_matrix, positions, targets, keep=2).shape == (0, 2)


def test_frontier_loss():
    """The focal-BCE (softmax over sources) is far lower when the logits already place mass on the true parent."""
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])  # source 0 is the parent of target 0
    right = _frontier_loss(torch.tensor([[9.0, 0.0], [-9.0, 0.0]]), target)  # source 0 dominates column 0
    wrong = _frontier_loss(torch.tensor([[-9.0, 0.0], [9.0, 0.0]]), target)  # the other source dominates

    assert right < wrong
    assert right < 0.1
