"""Unit tests for the two mixing policies — the guarantees are "nothing starves" and "selection stays honest"."""

import pytest

from celltrack.data.pair_curriculum import FixedMixture, PacedMixture, PairCurriculum, _Blocks

_REAL, _SYNTHETIC = 10, 4
_WINDOWS = 4


def _drawn(policy: PairCurriculum, steps: int) -> list[int]:
    return [policy.step().index for _ in range(steps)]


def test_of():
    """A mixture over one population is a configuration error, not a silently single-corpus run."""
    with pytest.raises(ValueError, match="both populations"):
        _Blocks.of(_REAL, 0, seed=0)
    with pytest.raises(ValueError, match="both populations"):
        _Blocks.of(0, _SYNTHETIC, seed=0)


def test_index():
    """The index space is the real block then the synthetic one; a draw never crosses into the other."""
    blocks = _Blocks.of(_REAL, _SYNTHETIC, seed=0)

    assert all(blocks.index(synthetic=False) < _REAL for _ in range(50))
    assert all(_REAL <= blocks.index(synthetic=True) < _REAL + _SYNTHETIC for _ in range(50))


def test_fixed_mixture_step():
    """The realised synthetic share converges on the asked-for fraction, at unit weight for every pair."""
    policy = FixedMixture(_REAL, _SYNTHETIC, 0.25, seed=0)
    draws = [policy.step() for _ in range(4000)]

    assert all(0 <= draw.index < _REAL + _SYNTHETIC for draw in draws)
    assert {draw.weight for draw in draws} == {1.0}  # a fixed diet needs no per-sample weight
    assert sum(draw.index >= _REAL for draw in draws) / len(draws) == pytest.approx(0.25, abs=0.02)


def test_fixed_mixture_health():
    """The row carries what the run actually DREW, not what it asked for."""
    policy = FixedMixture(_REAL, _SYNTHETIC, 0.5, seed=0)
    assert policy.health() == {"synthetic_share": 0.0}  # nothing drawn yet

    drawn = _drawn(policy, 1000)

    assert policy.health()["synthetic_share"] == pytest.approx(sum(i >= _REAL for i in drawn) / len(drawn))


def test_fixed_mixture_selection_weight():
    """A constant difficulty rescales nothing, so the fixed arms stay comparable to the runs before them."""
    assert FixedMixture(_REAL, _SYNTHETIC, 0.5, seed=0).selection_weight() == 1.0


def test_fixed_mixture_rejects_a_share_that_removes_a_population():
    """0 or 1 is not a mixture; the caller wanting one corpus alone should not build a mixture at all."""
    for fraction in (0.0, 1.0):
        with pytest.raises(ValueError, match="synthetic share"):
            FixedMixture(_REAL, _SYNTHETIC, fraction, seed=0)


def test_paced_mixture_step():
    """The two populations alternate STRUCTURALLY — the whole answer to the sampler that starved."""
    indices = _drawn(PacedMixture(_REAL, _SYNTHETIC, _WINDOWS, seed=0), 200)

    assert all(index < _REAL for index in indices[::2])
    assert all(index >= _REAL for index in indices[1::2])


def test_update():
    """One bad window moves the difficulty part-way, never all the way — the anti-oscillation property."""
    policy = PacedMixture(_REAL, _SYNTHETIC, _WINDOWS, seed=0)

    policy.update(0.9)  # a new best
    policy.update(0.1)  # a regression

    # strictly between the extremes an instantaneous rule would have jumped to
    assert 0.5 < policy.selection_weight() < 1.0


def test_paced_mixture_health():
    """Hard-early, and bounded: difficulty starts at its maximum and neither weight can ever reach zero."""
    policy = PacedMixture(_REAL, _SYNTHETIC, _WINDOWS, seed=0)
    assert policy.health()["difficulty"] == 1.0
    assert policy.health()["synthetic_weight"] > policy.health()["real_weight"]

    for _ in range(50):
        policy.update(0.0)  # never improves: drive the difficulty as low as the rule allows
    health = policy.health()

    assert health["synthetic_weight"] >= 0.5
    assert health["real_weight"] <= 1.5
    assert health["synthetic_weight"] + health["real_weight"] == pytest.approx(2.0)


def test_paced_mixture_selection_weight():
    """Save-best credits the diet the window trained on: `0.5 + 0.5 * difficulty`, 1.0 at full difficulty."""
    policy = PacedMixture(_REAL, _SYNTHETIC, _WINDOWS, seed=0)
    assert policy.selection_weight() == pytest.approx(1.0)

    for _ in range(50):
        policy.update(0.0)

    assert policy.selection_weight() == pytest.approx(0.5, abs=0.02)


def test_falls_below():
    """The blocks own the randomness: one seeded stream serves both the index draw and the mixture draw."""
    blocks = _Blocks.of(real_count=4, synthetic_count=4, seed=0)

    assert blocks.falls_below(1.0) is True  # a probability of one always fires
    assert blocks.falls_below(0.0) is False  # and zero never does
