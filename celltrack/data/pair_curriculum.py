"""Which pair the next step sees, and how much that step is allowed to count.

Two populations now feed the joint loop — the competition's own pairs, whose candidate set is the ~1-2%
of cells our annotation covers, and the synthetic pairs, whose candidate set is COMPLETE (see
`celltrack.data.synthetic_pairs`). Mixing them is a policy, and this module is the seam that policy plugs
into: one `step()` per optimiser step returning WHICH corpus index and WITH WHAT WEIGHT, one `update()` per
eval window, and one `selection_weight()` so a policy that moves the data's difficulty cannot silently
corrupt the checkpoint choice it is being judged by.

`FixedMixture` is the honest first experiment and holds no feedback at all: a fixed synthetic share, unit
weight, so a 0%/25%/50% ladder measures whether complete candidate sets help BEFORE any controller exists.

`PacedMixture` is the controller, and each of its four properties answers a failure this campaign has
already had.

1. FEEDBACK ON A CONTINUOUS KNOB. One scalar difficulty, read off the model's own eval score once per
   window — teacher-paced, not per-sample resampling.
2. EMA, NOT THE LAST OBSERVATION. `DifficultySampler` stored the latest measurement and starved on it
   (sampling entropy 0.961 -> 0.235, edge loss to exactly 0.0000, held-out recall 0.672 -> 0.201). The
   standard mitigation for that oscillation is to smooth the signal, so the difficulty here is an EMA whose
   span is the run's own window count — one derived rate, not a tuned constant.
3. WEIGHT, DO NOT SELECT. That same sampler made difficulty a DRAW PROBABILITY, which is exactly what let
   pairs vanish. Here the two populations alternate STRUCTURALLY — neither can ever be starved of draws —
   and difficulty moves a per-sample LOSS WEIGHT instead. The weight is itself the even mixture of a flat
   half and a difficulty-proportional half (`_PACED_SHARE`, the same maximum-ignorance split
   `DifficultySampler._DIFFICULTY_SHARE` argues for and reused rather than re-derived), so the two weights
   always sum to 2 and neither can reach zero however the difficulty moves.
4. DIFFICULTY-ADJUSTED SELECTION, in the same object. A curriculum that changes the data mid-run breaks
   save-best: later windows can read worse purely because the diet got harder, the selector keeps the
   easy-window checkpoint, and the experiment measures itself wrong. `selection_weight()` is the correction,
   and it is 1.0 for every policy whose difficulty does NOT move — a constant rescaling changes no argmax,
   so the fixed arms stay byte-comparable to the runs that came before them.

RAMP DIRECTION: HARD-EARLY. Difficulty starts at its MAXIMUM and the feedback only relaxes it. The condition
for anti-curriculum is "easy examples create shortcuts", and that is measurably our situation: the easy
population is the GT-only pairs where the truth already wins by ~4 logits, and `corr(P, distance) = -0.893`
with within-band AUC at chance below 2 um because 96% of in-gate candidates there ARE the true successor.
Easy-first would spend the early, most plastic steps training the near=successor shortcut harder. Optimistic
initialisation is also the rule `DifficultySampler` already argues for, for the same reason: an unmeasured
state should outrank a measured one.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# The even split between a flat weight and a difficulty-proportional one — the same maximum-ignorance choice
# `DifficultySampler` derives for its draw distribution, applied to a weight instead. It is what bounds both
# populations' weights into [0.5, 1.5]: nothing starves, whatever the difficulty does.
_PACED_SHARE = 0.5

# The two populations take strictly alternating steps, so neither can be starved of draws by the controller.
_POPULATIONS = 2


@dataclass(frozen=True)
class PairDraw:
    """One step's assignment: which pair of the concatenated corpus, and how much its loss counts."""

    index: int
    weight: float = 1.0


@runtime_checkable
class PairCurriculum(Protocol):
    """A policy over (which pair, how much it counts), fed the model's own numbers as the run goes."""

    def step(self) -> PairDraw:
        """The next optimiser step's pair index and loss weight."""
        ...

    def observe(self, index: int, difficulty: float | None) -> None:
        """Record the just-stepped pair's own measured defect; policies that do not read it ignore it."""
        ...

    def update(self, score: float) -> None:
        """Close the window-level loop with the model's eval score; a policy without feedback ignores it."""
        ...

    def selection_weight(self) -> float:
        """What the validation metric is multiplied by before save-best — 1.0 unless difficulty MOVES."""
        ...

    def health(self) -> dict[str, float]:
        """The policy's own diagnostics for this window's tracked row."""
        ...


@dataclass
class _Blocks:
    """The concatenated corpus's geometry — a real block, then a synthetic one — and a uniform draw from either.

    Held by each policy rather than inherited from, so a policy IS its rule and nothing else; the index space
    is a fact about the corpus, shared by composition.
    """

    real_count: int
    synthetic_count: int
    rng: np.random.Generator

    @classmethod
    def of(cls, real_count: int, synthetic_count: int, seed: int) -> "_Blocks":
        """The two blocks and their seeded generator; a missing population is a configuration error, not a run."""
        if not real_count or not synthetic_count:
            raise ValueError(f"a mixture needs both populations, got {real_count} real and {synthetic_count} synthetic")
        return cls(real_count=real_count, synthetic_count=synthetic_count, rng=np.random.default_rng(seed))

    def index(self, *, synthetic: bool) -> int:
        """A uniform draw from one of the two blocks, in the concatenated `real + synthetic` index space."""
        if synthetic:
            return self.real_count + int(self.rng.integers(self.synthetic_count))
        return int(self.rng.integers(self.real_count))

    def falls_below(self, probability: float) -> bool:
        """A Bernoulli draw at `probability`, taken from the corpus's own generator.

        The blocks own the randomness, so a policy asks them for an outcome rather than reaching through to the
        generator — one seeded stream serves the index draw and the mixture draw, which is what makes a run
        reproducible from a single seed.
        """
        return bool(self.rng.random() < probability)


class FixedMixture:
    """A constant synthetic share at unit weight — the arm that measures the corpus, not a controller."""

    def __init__(self, real_count: int, synthetic_count: int, fraction: float, seed: int) -> None:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"a mixture's synthetic share must leave both populations present, got {fraction}")
        self._blocks = _Blocks.of(real_count, synthetic_count, seed)
        self._fraction = fraction
        self._drawn = 0
        self._synthetic_drawn = 0

    def step(self) -> PairDraw:
        """A synthetic pair with probability `fraction`, a real one otherwise; both weigh the same."""
        synthetic = self._blocks.falls_below(self._fraction)
        self._drawn += 1
        self._synthetic_drawn += int(synthetic)
        return PairDraw(index=self._blocks.index(synthetic=synthetic), weight=1.0)

    def observe(self, index: int, difficulty: float | None) -> None:
        """Per-pair defect is not this family's feedback channel — the eval score is (see `update`)."""

    def update(self, score: float) -> None:
        """A fixed diet has no feedback loop to close; the score changes nothing here."""

    def selection_weight(self) -> float:
        """1.0 — the diet never changes, so save-best needs no correction and the score stays comparable."""
        return 1.0

    def health(self) -> dict[str, float]:
        """The realised synthetic share, so the log carries what the run actually drew, not what it asked for."""
        return {"synthetic_share": self._synthetic_drawn / max(self._drawn, 1)}


class PacedMixture:
    """The feedback controller: structural alternation, an EMA'd difficulty on the loss weight, hard-early."""

    def __init__(self, real_count: int, synthetic_count: int, windows: int, seed: int) -> None:
        self._blocks = _Blocks.of(real_count, synthetic_count, seed)
        # EMA span = the run's own eval windows, so the smoothing horizon is half the run by the standard
        # span identity — a rate derived from the schedule rather than a constant chosen for it.
        self._rate = _POPULATIONS / (max(windows, 1) + 1)
        self._difficulty = 1.0  # hard-early: start at the maximum, let the feedback relax it
        self._best = float("-inf")
        self._drawn = 0

    def step(self) -> PairDraw:
        """Alternate the two populations — neither can starve — and weight the pair by the current difficulty."""
        synthetic = bool(self._drawn % _POPULATIONS)
        self._drawn += 1
        return PairDraw(index=self._blocks.index(synthetic=synthetic), weight=self._weight(synthetic=synthetic))

    def observe(self, index: int, difficulty: float | None) -> None:
        """Per-pair defect is not this policy's channel: it paces on the model's own eval score, once a window."""

    def update(self, score: float) -> None:
        """Fold this window's verdict into the difficulty: it improved on the run's best, or it did not.

        The signal is the model's own eval score reduced to the one bit a curriculum can act on — a raw score
        difference is not on the difficulty's scale and would need a constant to map it there, while "did this
        window earn a new best" is scale-free and is exactly the condition "the model can take more".
        """
        improved = float(score > self._best)
        self._best = max(self._best, score)
        self._difficulty = (1.0 - self._rate) * self._difficulty + self._rate * improved

    def selection_weight(self) -> float:
        """`0.5 + 0.5 * difficulty` — save-best credits a window for the diet it was trained on.

        Without it the curriculum breaks its own selector: a later window can read worse purely because the
        mix got harder, so the easy-window checkpoint wins and the experiment measures the wrong thing.
        """
        return 0.5 + 0.5 * self._difficulty

    def health(self) -> dict[str, float]:
        """The controller's state — the difficulty itself, and the two weights it currently implies."""
        return {
            "difficulty": self._difficulty,
            "synthetic_weight": self._weight(synthetic=True),
            "real_weight": self._weight(synthetic=False),
        }

    def _weight(self, *, synthetic: bool) -> float:
        """Half flat, half difficulty-proportional — bounded into [0.5, 1.5], summing to 2 across the pair."""
        share = self._difficulty if synthetic else 1.0 - self._difficulty
        return _POPULATIONS * ((1.0 - _PACED_SHARE) / _POPULATIONS + _PACED_SHARE * share)
