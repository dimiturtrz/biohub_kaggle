"""Unit tests for the difficulty sampler — optimistic init, the uniform mixture that keeps every pair live, and
the two health numbers."""

from collections import Counter

import numpy as np

from celltrack.data.difficulty_sampler import _DIFFICULTY_SHARE, _MIN_ENTROPY, DifficultySampler


def test_draw():
    """Optimistic init still puts the unseen first: two pairs measured easy, the third (still 1.0) takes the bulk."""
    sampler = DifficultySampler(count=3, seed=0)
    sampler.observe(0, 0.0)
    sampler.observe(1, 0.0)
    counts = Counter(sampler.draw() for _ in range(300))
    assert counts[2] > counts[0] + counts[1]


def test_draw_is_proportional_to_difficulty():
    """The mixture preserves the proportional rule exactly: strip the uniform share and the ratios are unchanged."""
    sampler = DifficultySampler(count=3, seed=0)
    sampler.observe(0, 0.75)
    sampler.observe(1, 0.25)
    sampler.observe(2, 0.0)
    weighted = sampler._weights() - (1.0 - _DIFFICULTY_SHARE) / 3

    assert np.allclose(weighted, _DIFFICULTY_SHARE * np.array([0.75, 0.25, 0.0]))
    counts = Counter(sampler.draw() for _ in range(4000))
    assert counts[0] > counts[1] > counts[2] > 0


def test_draw_never_starves_a_pair():
    """The bug: a pair measured at 0.0 while others are positive used to have probability 0 and could never refresh."""
    sampler = DifficultySampler(count=8, seed=0)
    for index in range(1, 8):
        sampler.observe(index, 0.0)

    weights = sampler._weights()

    assert weights.min() == (1.0 - _DIFFICULTY_SHARE) / 8
    assert set(Counter(sampler.draw() for _ in range(2000))) == set(range(8))


def test_entropy_holds_its_floor_on_the_worst_difficulty_vector():
    """One pair at 1.0 and the rest at 0.0 is the most concentrated vector there is, and it still clears the floor."""
    for count in (2, 16, 1024):
        sampler = DifficultySampler(count=count, seed=0)
        for index in range(1, count):
            sampler.observe(index, 0.0)
        assert sampler.entropy() > _MIN_ENTROPY


def test_draw_falls_back_to_uniform_when_every_pair_is_solved():
    """All-zero difficulty has no proportional answer, so the distribution degenerates to uniform rather than to NaN."""
    sampler = DifficultySampler(count=4, seed=0)
    for index in range(4):
        sampler.observe(index, 0.0)
    assert sampler.entropy() == 1.0
    assert set(Counter(sampler.draw() for _ in range(400))) == {0, 1, 2, 3}


def test_observe():
    """A pair with no annotated source reports `None`: it counts as covered, but its stored difficulty is untouched."""
    sampler = DifficultySampler(count=2, seed=0)
    sampler.observe(0, 0.0)
    sampler.observe(1, None)
    assert sampler.coverage() == 1.0
    counts = Counter(sampler.draw() for _ in range(200))
    assert counts[1] > counts[0]  # pair 1 still sits at its optimistic 1.0


def test_coverage():
    """Coverage is the fraction of the corpus drawn at least once, and a redraw does not double-count."""
    sampler = DifficultySampler(count=4, seed=0)
    assert sampler.coverage() == 0.0
    sampler.observe(0, 0.5)
    sampler.observe(0, 0.5)
    assert sampler.coverage() == 0.25
    sampler.observe(3, None)
    assert sampler.coverage() == 0.5


def test_coverage_reaches_the_whole_corpus_in_about_one_pass():
    """Optimistic init is the half that works: solved pairs step aside, so coverage completes in ~n draws."""
    count = 16
    for seed in range(8):
        sampler = DifficultySampler(count=count, seed=seed)
        for _ in range(2 * count):
            sampler.observe(sampler.draw(), 0.0)
        assert sampler.coverage() == 1.0  # i.i.d. with replacement needs ~n ln n = 44 draws for this in expectation


def test_entropy():
    """Normalised Shannon entropy: 1.0 on the uniform init, and bounded below by the mixture once mass concentrates."""
    sampler = DifficultySampler(count=4, seed=0)
    assert sampler.entropy() == 1.0
    for index in range(1, 4):
        sampler.observe(index, 0.0)
    collapsed = sampler.entropy()
    assert _MIN_ENTROPY < collapsed < 1.0
    sampler.observe(1, 1.0)
    assert sampler.entropy() > collapsed  # two live pairs instead of one is strictly less collapsed
    assert DifficultySampler(count=1, seed=0).entropy() == 1.0  # a one-pair corpus has nowhere to collapse to
