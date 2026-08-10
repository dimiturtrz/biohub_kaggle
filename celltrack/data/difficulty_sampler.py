"""Which training pair the next optimiser step should see, when uniform-with-replacement is the wrong answer.

`PairDataset` draws i.i.d. with replacement, so over 18024 GT pairs one nominal epoch touches only
`1 - e^-1` = 63% of them and a 3000-step run touches ~17%: the pairs the model is worst on are seen no more
often than the ones it already solves. This sampler replaces the uniform draw with one proportional to a
per-pair DIFFICULTY, and initialises that difficulty OPTIMISTICALLY at its maximum (1.0) so a pair nobody has
looked at outranks every pair already measured. Optimistic init is why there is no separate "visible" flag and
no "have we covered everything yet" phase: one array, one rule, no branch. It is also why there is no decay
and no recency window — a stale estimate reads too HIGH as the model improves, so the pair is redrawn and
refreshed by the same rule that made it stale.

Difficulty is not loss. With ~1-2% dense annotation, a pair's loss scales with how many annotated edges it
happens to carry, so loss-weighting chases label count and annotation noise; the producer instead reports the
fraction of annotated sources whose true successor is not ranked top-1, which is bounded, edge-count
independent, and exactly the defect the campaign is chasing.

`entropy()` is the collapse guard: difficulty weighting can over-concentrate on genuinely unlearnable pairs
(the crowd-ambiguous cases are not separable from the features at this resolution), and that shows up as a
falling entropy long before it shows up in a score.
"""

import numpy as np
from jaxtyping import Float


class DifficultySampler:
    """Draws pair indices in proportion to a stored per-pair difficulty, optimistically initialised to 1.0."""

    def __init__(self, count: int, seed: int) -> None:
        self._difficulty = np.ones(count, dtype=np.float64)
        self._observed = np.zeros(count, dtype=bool)
        self._rng = np.random.default_rng(seed)

    def draw(self) -> int:
        """One pair index, sampled with probability proportional to its stored difficulty."""
        return int(self._rng.choice(len(self._difficulty), p=self._weights()))

    def observe(self, index: int, difficulty: float | None) -> None:
        """Record a drawn pair's measured difficulty; `None` (no annotated source) leaves the stored value alone."""
        self._observed[index] = True
        if difficulty is not None:
            self._difficulty[index] = difficulty

    def coverage(self) -> float:
        """Fraction of pairs drawn at least once. Reporting only — being observed never enters the sampling rule."""
        return float(self._observed.mean())

    def entropy(self) -> float:
        """Shannon entropy of the sampling distribution, NORMALISED to [0, 1]: 1.0 is uniform, 0.0 is one pair.

        Normalised (divided by `ln(n)`) rather than left in nats so the number is comparable across corpus
        sizes and reads directly as "how collapsed is the distribution".
        """
        weights = self._weights()
        spread = np.log(len(weights))
        if not spread:
            return 1.0
        live = weights[weights > 0.0]
        return float(-(live * np.log(live)).sum() / spread)

    def _weights(self) -> Float[np.ndarray, "n"]:
        """The normalised sampling distribution.

        All-zero difficulty (every pair solved) has no proportional answer, so it falls back to uniform — the
        limit of "proportional to a constant vector", not a floor: it introduces no constant, and the moment
        any pair regresses the proportional rule resumes with nothing to unwind. Note the flip side, which is
        a property of the rule and not of this fallback: a pair that reaches exactly 0.0 while others are
        positive is never redrawn, so its estimate can no longer refresh.
        """
        total = self._difficulty.sum()
        if not total:
            return np.full(len(self._difficulty), 1.0 / len(self._difficulty))
        return self._difficulty / total
