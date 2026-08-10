"""Unit test for the softmax-focal BCE link loss — one parent per target."""

import torch

from celltrack.training.softmax_focal_bce import SoftmaxFocalBCE


def test_softmax_focal_b_c_e_of():
    """A confident-correct link scores below a confident-wrong one; finite + differentiable; empty target is zero."""
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0]])  # source 0 is the parent of target 0
    right = torch.tensor([[6.0, -6.0], [-6.0, -6.0]], requires_grad=True)
    wrong = torch.tensor([[-6.0, 6.0], [6.0, 6.0]])
    lower = SoftmaxFocalBCE.of(right, target)
    assert torch.isfinite(lower) and lower.item() < SoftmaxFocalBCE.of(wrong, target).item()
    lower.backward()
    assert right.grad is not None
    assert SoftmaxFocalBCE.of(torch.zeros(2, 2), torch.zeros(2, 2)).item() == 0.0
