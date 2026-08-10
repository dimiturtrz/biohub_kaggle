"""Unit tests for the joint detector+edge loss composition — the pilkwang joint objective this repo owns.

Exercised on tiny synthetic tensors (4x4x4 detection maps, a 2x2 edge matrix) so a full forward/backward is
covered on the CPU without the model or any data: the loss is finite/positive/differentiable, and a
confident-correct frame pair scores strictly below a confident-wrong one.
"""

import torch

from celltrack.training.joint_loss import JointLoss


def _centres() -> tuple[torch.Tensor, torch.Tensor]:
    """One ground-truth cell per frame at distinct voxels, as downsampled integer indices."""
    return torch.tensor([[1, 1, 1]]), torch.tensor([[2, 2, 2]])


def test_compute():
    """Finite/positive/differentiable, and a confident-correct pair beats a confident-wrong one."""
    torch.manual_seed(0)
    centres_t, centres_t1 = _centres()
    edge_matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0]])  # source 0 -> target 0 is the true edge

    detection_t = torch.zeros(4, 4, 4, requires_grad=True)
    detection_t1 = torch.zeros(4, 4, 4, requires_grad=True)
    edge_logits = torch.zeros(2, 2, requires_grad=True)
    loss = JointLoss.compute(
        detection_t=detection_t,
        detection_t1=detection_t1,
        centres_t=centres_t,
        centres_t1=centres_t1,
        edge_logits=edge_logits,
        edge_matrix=edge_matrix,
        neg_weight=1e-2,
        det_weight=0.1,
    )
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert edge_logits.grad is not None

    right_t = torch.full((4, 4, 4), -6.0)
    right_t[1, 1, 1] = 6.0
    right_t1 = torch.full((4, 4, 4), -6.0)
    right_t1[2, 2, 2] = 6.0
    right_edges = torch.tensor([[6.0, -6.0], [-6.0, -6.0]])
    lower = JointLoss.compute(
        detection_t=right_t,
        detection_t1=right_t1,
        centres_t=centres_t,
        centres_t1=centres_t1,
        edge_logits=right_edges,
        edge_matrix=edge_matrix,
        neg_weight=1e-2,
        det_weight=0.1,
    ).item()

    wrong_t = torch.full((4, 4, 4), 6.0)
    wrong_t[1, 1, 1] = -6.0
    wrong_t1 = torch.full((4, 4, 4), 6.0)
    wrong_t1[2, 2, 2] = -6.0
    wrong_edges = torch.tensor([[-6.0, 6.0], [6.0, -6.0]])
    higher = JointLoss.compute(
        detection_t=wrong_t,
        detection_t1=wrong_t1,
        centres_t=centres_t,
        centres_t1=centres_t1,
        edge_logits=wrong_edges,
        edge_matrix=edge_matrix,
        neg_weight=1e-2,
        det_weight=0.1,
    ).item()

    assert lower < higher
