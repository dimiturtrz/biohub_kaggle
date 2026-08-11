"""Unit test for the hard-negative RANKING term — the property that separates it from the refuted absolute form."""

import torch

from celltrack.losses.hard_negative_margin import HardNegativeMargin

_LINKS = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # source 0 -> target 0; source 1 is unannotated


def _positions(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows)


def test_mine():
    """Each ANNOTATED source's nearest wrong targets, in distance order; unannotated rows contribute nothing."""
    sources = _positions([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
    targets = _positions([[0.0, 0.0, 9.0], [0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])

    assert HardNegativeMargin.mine(sources, targets, _LINKS, keep=1).tolist() == [[0, 1]]
    assert HardNegativeMargin.mine(sources, targets, _LINKS, keep=2).tolist() == [[0, 1], [0, 2]]
    # keep beyond the wrong targets available takes what exists, never a true column and never a repeat.
    assert HardNegativeMargin.mine(sources, targets, _LINKS, keep=9).tolist() == [[0, 1], [0, 2]]
    assert HardNegativeMargin.mine(sources, targets, torch.zeros(2, 3), keep=2).shape == (0, 2)


def test_of():
    """THE property the form exists for: depressing a whole row leaves the loss EXACTLY unchanged.

    The refuted zni term (the decoy's absolute probability) is minimised by pushing every candidate down, which
    is what collapsed `P_true` with `P_chosen`. A difference of two logits cannot be paid for that way.
    """
    logits = torch.tensor([[2.0, 1.0, 0.5], [0.0, 0.0, 0.0]])
    decoys = HardNegativeMargin.mine(
        _positions([[0.0, 0.0, 0.0], [0.0, 0.0, 9.0]]),
        _positions([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        _LINKS,
        keep=2,
    )
    flattened = logits - torch.tensor([[7.0], [7.0]])

    assert torch.allclose(
        HardNegativeMargin.of(logits, _LINKS, decoys), HardNegativeMargin.of(flattened, _LINKS, decoys)
    )


def test_of_falls_as_the_truth_out_ranks_the_decoys():
    """Ranking is the only way down: an inverted row scores above a tied one, which scores above a ranked one."""
    decoys = torch.tensor([[0, 1], [0, 2]])
    inverted = HardNegativeMargin.of(torch.tensor([[-4.0, 4.0, 4.0], [0.0, 0.0, 0.0]]), _LINKS, decoys)
    tied = HardNegativeMargin.of(torch.zeros(2, 3), _LINKS, decoys)
    ranked = torch.tensor([[4.0, -4.0, -4.0], [0.0, 0.0, 0.0]], requires_grad=True)
    separated = HardNegativeMargin.of(ranked, _LINKS, decoys)

    assert inverted.item() > tied.item() > separated.item() > 0.0
    separated.backward()
    assert ranked.grad is not None and ranked.grad.abs().sum() > 0
    assert HardNegativeMargin.of(ranked, _LINKS, torch.zeros(0, 2, dtype=torch.int64)).item() == 0.0


def test_of_holds_every_daughter_of_a_division_above_the_decoy():
    """A dividing source is scored on its WEAKEST true daughter, so one well-ranked daughter cannot mask the other."""
    division = torch.tensor([[1.0, 1.0, 0.0]])
    decoys = torch.tensor([[0, 2]])
    both_high = HardNegativeMargin.of(torch.tensor([[4.0, 4.0, 0.0]]), division, decoys)
    one_low = HardNegativeMargin.of(torch.tensor([[4.0, -4.0, 0.0]]), division, decoys)

    assert one_low.item() > both_high.item()
