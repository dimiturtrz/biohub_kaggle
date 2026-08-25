"""Unit tests for the seed logit moment alignment — the moments themselves, the affine map, and the clamp."""

import math

import torch

from celltrack.edges.seed_moment_alignment import LogitMoments, SeedMomentAlignment


def test_moments():
    """`moments` reports what it claims: the arithmetic mean and the POPULATION std of the whole matrix."""
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reported = SeedMomentAlignment.moments(logits)
    assert reported.mean == 2.5
    assert math.isclose(reported.std, math.sqrt(1.25), rel_tol=1e-6)


def test_moments_of_a_single_candidate_is_zero_spread():
    """A one-entry gap has an honest population spread of zero — where a sample std would be undefined."""
    reported = SeedMomentAlignment.moments(torch.tensor([[7.0]]))
    assert reported == LogitMoments(mean=7.0, std=0.0)


def test_align():
    """The aligned logits take the reference's mean and spread exactly, when the ratio is inside the clamp."""
    logits = torch.tensor([[0.0, 2.0], [4.0, 6.0]])  # mean 3.0, std sqrt(5)
    reference = torch.tensor([[9.0, 11.0], [13.0, 15.0]])  # mean 12.0, same shape of spread
    aligned = SeedMomentAlignment.align(logits, reference)
    moments = SeedMomentAlignment.moments(aligned)
    expected = SeedMomentAlignment.moments(reference)
    assert math.isclose(moments.mean, expected.mean, rel_tol=1e-5)
    assert math.isclose(moments.std, expected.std, rel_tol=1e-5)
    assert torch.allclose(aligned, torch.tensor([[9.0, 11.0], [13.0, 15.0]]), atol=1e-5)


def test_align_preserves_the_ranking():
    """Alignment is affine with a positive scale, so it never reorders a target's candidate sources."""
    logits = torch.tensor([[0.5, -3.0], [2.0, 1.0]])
    aligned = SeedMomentAlignment.align(logits, torch.tensor([[4.0, 8.0], [12.0, 16.0]]))
    assert torch.equal(aligned.argsort(dim=0), logits.argsort(dim=0))


def test_align_clamps_a_too_large_ratio():
    """A reference four times as wide is corrected by the ceiling ratio, not by the raw four."""
    logits = torch.tensor([[-1.0, 1.0]])  # mean 0.0, std 1.0
    reference = torch.tensor([[-4.0, 4.0]])  # mean 0.0, std 4.0 -> raw ratio 4.0
    aligned = SeedMomentAlignment.align(logits, reference)
    assert torch.allclose(aligned, logits * SeedMomentAlignment.STD_RATIO_MAX)
    assert not torch.allclose(aligned, reference)


def test_align_clamps_a_too_small_ratio():
    """A reference four times as narrow is corrected by the floor ratio — the clamp bites in both directions."""
    logits = torch.tensor([[-4.0, 4.0]])
    reference = torch.tensor([[-1.0, 1.0]])
    aligned = SeedMomentAlignment.align(logits, reference)
    assert torch.allclose(aligned, logits * SeedMomentAlignment.STD_RATIO_MIN)


def test_align_of_a_flat_seed_is_finite():
    """A seed with no spread at all rescales by the clamped ceiling instead of dividing by zero."""
    aligned = SeedMomentAlignment.align(torch.full((2, 2), 5.0), torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    assert torch.isfinite(aligned).all()
    assert torch.allclose(aligned, torch.full((2, 2), 1.5))


def test_align_per_column():
    """Each TARGET column is z-scored by its OWN moments over the sources, unlike the scalar-global `align`.

    On a column-varying matrix the per-column map re-centres each column onto the reference column's mean
    (so every output column has ~zero mean once the reference's is subtracted back), and it differs from the
    global `align`, which shares one scalar mean/std over the whole matrix.
    """
    logits = torch.tensor([[0.0, 10.0], [2.0, 14.0]])  # col 0: mean 1 std 1; col 1: mean 12 std 2
    reference = torch.zeros(2, 2)  # zero mean per column -> output columns re-centre onto zero
    aligned = SeedMomentAlignment.align_per_column(logits, reference)

    assert torch.isfinite(aligned).all()
    assert torch.allclose(aligned.mean(dim=0), torch.zeros(2), atol=1e-6)  # each column centred on the reference's mean

    global_aligned = SeedMomentAlignment.align(logits, reference)
    assert not torch.allclose(aligned, global_aligned)  # per-column calibration is not the scalar-global one
