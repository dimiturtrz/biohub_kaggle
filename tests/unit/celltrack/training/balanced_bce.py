"""Unit test for the class-balanced BCE detection loss — equivalence classes of the weighting."""

import torch

from celltrack.training.balanced_bce import BalancedBCE


def test_balanced_b_c_e_of():
    """A confident-correct map scores below the inverse; finite + differentiable; out-of-bounds/empty are defined."""
    logits = torch.zeros((1, 4, 4, 4), requires_grad=True)
    centres = [torch.tensor([[1, 1, 1]])]
    loss = BalancedBCE.of(logits, centres, neg_weight=1e-2)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert logits.grad is not None
    right = torch.full((1, 4, 4, 4), -6.0)
    right[0, 1, 1, 1] = 6.0
    wrong = torch.full((1, 4, 4, 4), 6.0)
    wrong[0, 1, 1, 1] = -6.0
    assert BalancedBCE.of(right, centres, 1e-2).item() < BalancedBCE.of(wrong, centres, 1e-2).item()
    assert torch.isfinite(BalancedBCE.of(logits, [torch.tensor([[9, 0, 0]])], 1e-2))  # out of bounds dropped
    assert torch.isfinite(BalancedBCE.of(logits, [torch.empty((0, 3), dtype=torch.long)], 1e-2))  # empty frame
    # batch mean is scale-stable: two identical frames == one.
    centre = torch.tensor([[1, 1, 1]])
    one = BalancedBCE.of(torch.zeros((1, 4, 4, 4)), [centre], 1e-2).item()
    two = BalancedBCE.of(torch.zeros((2, 4, 4, 4)), [centre, centre], 1e-2).item()
    assert abs(one - two) < 1e-5
