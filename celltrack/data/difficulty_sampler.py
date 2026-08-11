"""Which training pair the next optimiser step should see, when uniform-with-replacement is the wrong answer.

`PairDataset` draws i.i.d. with replacement, so over 18024 GT pairs one nominal epoch touches only
`1 - e^-1` = 63% of them and a 3000-step run touches ~17%: the pairs the model is worst on are seen no more
often than the ones it already solves. This sampler replaces the uniform draw with one proportional to a
per-pair DIFFICULTY, and initialises that difficulty OPTIMISTICALLY at its maximum (1.0) so a pair nobody has
looked at outranks every pair already measured. Optimistic init is why there is no separate "visible" flag and
no "have we covered everything yet" phase: one array, one rule, no branch. It is also why there is no decay
and no recency window — a stale estimate reads too HIGH as the model improves, so the pair is redrawn and
refreshed by the same rule that made it stale.

That refresh argument has a precondition the purely proportional rule does not meet: it holds only while every
pair keeps a NONZERO draw probability. A pair measured at exactly 0.0 is never redrawn, so its estimate can
never be revised, and mass concentrates permanently on whatever stays hard — measured, that was a run whose
sampling entropy fell 0.961 -> 0.235 over 2.5 epochs while held-out recall peaked at 1.5 epochs and then
collapsed, the whole budget spent on the crowd-ambiguous pairs that are not separable from these features at
this resolution. The repair is a floor on the DISTRIBUTION, not on difficulty: a difficulty floor would be a
tuned constant and would distort the very proportions it is protecting, whereas mixing the difficulty
distribution with the uniform one, `p = (1 - a) * uniform + a * proportional`, leaves every ratio between
measured pairs intact and still gives every pair at least `(1 - a) / n`.

`a` is the even mixture (see `_DIFFICULTY_SHARE`), so this stays a two-line rule with no decay constant, no
recency window and no temperature.

Difficulty is not loss. With ~1-2% dense annotation, a pair's loss scales with how many annotated edges it
happens to carry, so loss-weighting chases label count and annotation noise; the producer instead reports the
fraction of annotated sources whose true successor is not ranked top-1, which is bounded, edge-count
independent, and exactly the defect the campaign is chasing.

`entropy()` remains the collapse guard, and is now a guarantee rather than only a diagnostic: the mixture
bounds it below by `_MIN_ENTROPY` whatever the difficulty vector does. It stays the acceptance test for any
future change here.
"""

import numpy as np
from jaxtyping import Float

from celltrack.data.pair_curriculum import PairDraw

_DIFFICULTY_SHARE = 0.5
"""How much of the draw mass follows difficulty rather than uniform — the even mixture, argued not swept.

`a` trades two failure modes that are not comparable in any shared unit: too small and the sampler is the
i.i.d. stream it exists to replace, too large and a pair reaching difficulty 0.0 is frozen out and its
estimate can never be revised. Nothing in the problem ranks exploration above exploitation or the reverse, so
the maximum-ignorance choice is the one that states no preference: half each. It is also the choice that
needs no re-derivation per corpus, per campaign or per run, which a value picked off a sweep would.

Read as a timescale it says: an epoch is `n` draws, the uniform half delivers each pair `(1 - a)` times per
epoch in expectation, so at `a = 1/2` the coldest possible pair still refreshes about once every two epochs.
The measured collapse ran its full course inside 2.5 epochs, so that is the timescale the guarantee has to
live on — a consequence of the even split, not the reason for it.
"""

_MIN_ENTROPY = 0.5
"""The normalised sampling entropy `_DIFFICULTY_SHARE` guarantees — a derived bound, not a target to tune to.

Normalised entropy `H` is the log of the distribution's perplexity in units of `ln n`, so `n ** H` is the
EFFECTIVE number of pairs the sampler is live on: `H = 1` is the whole corpus (the i.i.d. baseline) and
`H = 0` is a single pair (the collapse). Under the worst difficulty vector there is — all mass on one pair —
the even mixture has entropy `1/2 + ln 2 / ln n` in the large-`n` limit and strictly more at every finite `n`
(0.81 at n=2, 0.71 at n=16, 0.5707 at the 18024-pair corpus), so `1/2` is the infimum over corpus sizes and
over difficulty vectors alike. Equivalently: the effective support can never fall below `sqrt(n)`, the
geometric mean of collapse and uniform.
"""


class DifficultySampler:
    """Draws pair indices from a difficulty-proportional distribution evenly mixed with the uniform one.

    It is one of the loop's `PairCurriculum` policies (`celltrack.data.pair_curriculum`) — the one whose
    feedback is per-PAIR rather than per-window, and whose knob is the draw probability rather than a loss
    weight. Sharing that seam is what lets the trainer hold one policy object instead of branching per family.
    """

    def __init__(self, count: int, seed: int) -> None:
        self._difficulty = np.ones(count, dtype=np.float64)
        self._observed = np.zeros(count, dtype=bool)
        self._rng = np.random.default_rng(seed)

    def step(self) -> PairDraw:
        """One drawn pair at unit weight — this policy expresses difficulty as a draw, never as a weight."""
        return PairDraw(index=self.draw())

    def update(self, score: float) -> None:
        """No window-level channel: this sampler's feedback arrives per pair, through `observe`."""

    def selection_weight(self) -> float:
        """1.0 — the corpus never changes here, so save-best reads the raw validation metric."""
        return 1.0

    def health(self) -> dict[str, float]:
        """The two numbers a run watches for the collapse this sampler is known to be able to reach."""
        return {"coverage": self.coverage(), "sampling_entropy": self.entropy()}

    def draw(self) -> int:
        """One pair index, sampled from the mixed distribution; every pair keeps a probability of at least 1/(2n)."""
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
        sizes and reads directly as "how collapsed is the distribution". The mixture bounds it below by
        `_MIN_ENTROPY`, so a run that sinks towards that value is one whose difficulty has concentrated as far
        as the rule permits — the diagnostic still reads, it just can no longer run to zero.
        """
        weights = self._weights()
        spread = np.log(len(weights))
        if not spread:
            return 1.0
        live = weights[weights > 0.0]
        return float(-(live * np.log(live)).sum() / spread)

    def _weights(self) -> Float[np.ndarray, "n"]:
        """The sampling distribution: `_DIFFICULTY_SHARE` of the proportional rule, the rest spread uniformly."""
        count = len(self._difficulty)
        share = (1.0 - _DIFFICULTY_SHARE) / count
        return share + _DIFFICULTY_SHARE * self._proportional()

    def _proportional(self) -> Float[np.ndarray, "n"]:
        """Difficulty, normalised.

        All-zero difficulty (every pair solved) has no proportional answer, so it falls back to uniform — the
        limit of "proportional to a constant vector", not a floor: it introduces no constant, and the moment
        any pair regresses the proportional rule resumes with nothing to unwind.
        """
        total = self._difficulty.sum()
        if not total:
            return np.full(len(self._difficulty), 1.0 / len(self._difficulty))
        return self._difficulty / total
