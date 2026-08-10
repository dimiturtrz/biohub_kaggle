"""Unit tests for the difficulty sampler — optimistic init, the unchanged-on-None rule, and the two health numbers."""

from collections import Counter

from celltrack.data.difficulty_sampler import DifficultySampler


def test_draw():
    """Optimistic init alone puts the unseen first: two pairs measured easy, the third (still 1.0) takes every draw."""
    sampler = DifficultySampler(count=3, seed=0)
    sampler.observe(0, 0.0)
    sampler.observe(1, 0.0)
    assert {sampler.draw() for _ in range(20)} == {2}


def test_draw_is_proportional_to_difficulty():
    """A pair three times as defective is drawn about three times as often; a zero-difficulty pair not at all."""
    sampler = DifficultySampler(count=3, seed=0)
    sampler.observe(0, 0.75)
    sampler.observe(1, 0.25)
    sampler.observe(2, 0.0)
    counts = Counter(sampler.draw() for _ in range(4000))
    assert counts[2] == 0
    assert 2.5 < counts[0] / counts[1] < 3.5


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
    assert {sampler.draw() for _ in range(20)} == {1}  # pair 1 still sits at its optimistic 1.0


def test_coverage():
    """Coverage is the fraction of the corpus drawn at least once, and a redraw does not double-count."""
    sampler = DifficultySampler(count=4, seed=0)
    assert sampler.coverage() == 0.0
    sampler.observe(0, 0.5)
    sampler.observe(0, 0.5)
    assert sampler.coverage() == 0.25
    sampler.observe(3, None)
    assert sampler.coverage() == 0.5


def test_entropy():
    """Normalised Shannon entropy: 1.0 on the uniform init, 0.0 once one pair holds all the mass."""
    sampler = DifficultySampler(count=4, seed=0)
    assert sampler.entropy() == 1.0
    for index in range(1, 4):
        sampler.observe(index, 0.0)
    assert sampler.entropy() == 0.0
    sampler.observe(1, 1.0)
    assert abs(sampler.entropy() - 0.5) < 1e-12  # two of four pairs, equally weighted: ln 2 / ln 4
    assert DifficultySampler(count=1, seed=0).entropy() == 1.0  # a one-pair corpus has nowhere to collapse to
