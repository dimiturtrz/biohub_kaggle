"""Unit test for the class-balanced BCE detection loss — equivalence classes of the weighting and the mask."""

import torch
import torch.nn.functional as F
from torch import Tensor

from celltrack.losses.balanced_bce import BalancedBCE

_NEG_WEIGHT = 1e-2
_OPERATING_POINT = 0.97  # the tracker's inference threshold, the value the trainers derive their default from
_HIGH_LOGIT = 6.0  # sigmoid ~0.9975, above the operating point
_LOW_LOGIT = -3.0  # sigmoid ~0.047, below it


def _reference(logits: Tensor, centres: list[Tensor], neg_weight: float, supervised: Tensor | None = None) -> float:
    """An independent re-derivation of the objective: per frame, positives 1/n_pos and supervised negatives w/n_neg."""
    target = torch.zeros_like(logits)
    for index, coords in enumerate(centres):
        for z, y, x in coords.tolist():
            target[index, z, y, x] = 1.0
    keep = torch.ones_like(logits) if supervised is None else supervised.to(logits.dtype)
    total = 0.0
    for index in range(logits.shape[0]):
        n_pos = max(float(target[index].sum()), 1.0)
        n_neg = max(float(keep[index].sum()) - n_pos, 1.0)
        weight = torch.where(target[index] == 1.0, 1.0 / n_pos, neg_weight / n_neg) * keep[index]
        elementwise = F.binary_cross_entropy_with_logits(logits[index], target[index], reduction="none")
        total += float((elementwise * weight).sum())
    return total / logits.shape[0]


def _high_response_frame() -> tuple[Tensor, list[Tensor], Tensor]:
    """One 2x2x2 frame with an annotated centre, an UNANNOTATED high-response voxel, and the mask that spares it."""
    logits = torch.full((1, 2, 2, 2), _LOW_LOGIT)
    logits[0, 0, 0, 0] = _HIGH_LOGIT  # the annotated centre, firing correctly
    logits[0, 1, 1, 1] = _HIGH_LOGIT  # a confident response our sparse annotation says nothing about
    supervised = torch.ones_like(logits)
    supervised[0, 1, 1, 1] = 0.0
    return logits, [torch.tensor([[0, 0, 0]])], supervised


def test_per_frame():
    """The per-frame vector holds one scalar per frame and averages to exactly `of` — same weights, same masking."""
    torch.manual_seed(0)
    logits = torch.randn(3, 4, 4, 4)
    centres = [torch.tensor([[1, 1, 1]]), torch.tensor([[2, 2, 2], [0, 0, 0]]), torch.empty((0, 3), dtype=torch.long)]
    per_frame = BalancedBCE.per_frame(logits, centres, _NEG_WEIGHT)
    assert per_frame.shape == (3,)
    assert torch.allclose(per_frame.mean(), BalancedBCE.of(logits, centres, _NEG_WEIGHT), atol=1e-6)


def test_balanced_b_c_e_of():
    """A confident-correct map scores below the inverse; finite + differentiable; out-of-bounds/empty are defined."""
    logits = torch.zeros((1, 4, 4, 4), requires_grad=True)
    centres = [torch.tensor([[1, 1, 1]])]
    loss = BalancedBCE.of(logits, centres, neg_weight=_NEG_WEIGHT)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert logits.grad is not None
    right = torch.full((1, 4, 4, 4), -6.0)
    right[0, 1, 1, 1] = 6.0
    wrong = torch.full((1, 4, 4, 4), 6.0)
    wrong[0, 1, 1, 1] = -6.0
    assert BalancedBCE.of(right, centres, _NEG_WEIGHT).item() < BalancedBCE.of(wrong, centres, _NEG_WEIGHT).item()
    assert torch.isfinite(BalancedBCE.of(logits, [torch.tensor([[9, 0, 0]])], _NEG_WEIGHT))  # out of bounds dropped
    empty = [torch.empty((0, 3), dtype=torch.long)]
    assert torch.isfinite(BalancedBCE.of(logits, empty, _NEG_WEIGHT))  # empty frame
    # batch mean is scale-stable: two identical frames == one.
    centre = torch.tensor([[1, 1, 1]])
    one = BalancedBCE.of(torch.zeros((1, 4, 4, 4)), [centre], _NEG_WEIGHT).item()
    two = BalancedBCE.of(torch.zeros((2, 4, 4, 4)), [centre, centre], _NEG_WEIGHT).item()
    assert abs(one - two) < 1e-5

    # DEFAULT OFF: an unasked run is byte-identical to the objective before the mask existed, high responses
    # included — the empty frame too, where the positive-count clamp drives the negative normaliser.
    high, annotated, _ = _high_response_frame()
    for case, coords in ((high, annotated), (high, empty), (right, centres)):
        assert (
            BalancedBCE.of(case, coords, _NEG_WEIGHT).item() == BalancedBCE.of(case, coords, _NEG_WEIGHT, None).item()
        )
        assert abs(BalancedBCE.of(case, coords, _NEG_WEIGHT).item() - _reference(case, coords, _NEG_WEIGHT)) < 1e-6

    # MASKING: an unannotated voxel over the operating point contributes nothing — loss and weight both zero.
    masked = BalancedBCE.of(high, annotated, _NEG_WEIGHT, _OPERATING_POINT).item()
    _, _, supervised = _high_response_frame()
    assert abs(masked - _reference(high, annotated, _NEG_WEIGHT, supervised)) < 1e-6
    assert masked < BalancedBCE.of(high, annotated, _NEG_WEIGHT).item()  # the false "you are wrong" term is gone

    # The NEGATIVE NORMALISER counts what is supervised (6 survivors), not the raw negative count (7): the
    # supervised MEAN of the surviving negatives is preserved, so masking removes terms without rescaling the
    # objective — and the naive form, which drops the term but keeps the old divisor, is a different number.
    survivors = torch.tensor([_LOW_LOGIT] * 6)
    per_negative = float(F.binary_cross_entropy_with_logits(survivors, torch.zeros(6), reduction="sum")) / 6
    positive = float(F.binary_cross_entropy_with_logits(torch.tensor(_HIGH_LOGIT), torch.tensor(1.0)))
    assert abs(masked - (positive + _NEG_WEIGHT * per_negative)) < 1e-6
    naive = positive + _NEG_WEIGHT * per_negative * 6 / 7
    assert abs(masked - naive) > 1e-5

    # An ANNOTATED centre is never masked, however confidently it fires: with the ambiguous voxel removed from
    # the frame there is nothing left to mask, and the masked loss falls back onto the plain one.
    only_centre = torch.full((1, 2, 2, 2), _LOW_LOGIT)
    only_centre[0, 0, 0, 0] = _HIGH_LOGIT
    assert (
        BalancedBCE.of(only_centre, annotated, _NEG_WEIGHT, _OPERATING_POINT).item()
        == BalancedBCE.of(only_centre, annotated, _NEG_WEIGHT).item()
    )

    # ALL MASKED — a saturated frame with no annotation supervises nothing. Zero, not NaN from an empty divisor.
    saturated = torch.full((1, 2, 2, 2), _HIGH_LOGIT)
    all_masked = BalancedBCE.of(saturated, empty, _NEG_WEIGHT, _OPERATING_POINT)
    assert torch.isfinite(all_masked) and all_masked.item() == 0.0


def test_balanced_b_c_e_ignored_fraction():
    """The share of voxels the mask spares — zero when nothing is ambiguous, one when the whole frame is."""
    logits, centres, supervised = _high_response_frame()
    fraction = BalancedBCE.ignored_fraction(logits, centres, _OPERATING_POINT)
    assert abs(fraction - float(1.0 - supervised.mean())) < 1e-6
    quiet = torch.full((1, 2, 2, 2), _LOW_LOGIT)
    assert BalancedBCE.ignored_fraction(quiet, centres, _OPERATING_POINT) == 0.0
    # A confident ANNOTATED voxel is supervised, so it never counts as ignored.
    only_centre = quiet.clone()
    only_centre[0, 0, 0, 0] = _HIGH_LOGIT
    assert BalancedBCE.ignored_fraction(only_centre, centres, _OPERATING_POINT) == 0.0
    empty = [torch.empty((0, 3), dtype=torch.long)]
    assert BalancedBCE.ignored_fraction(torch.full((1, 2, 2, 2), _HIGH_LOGIT), empty, _OPERATING_POINT) == 1.0
