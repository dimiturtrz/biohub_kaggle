"""Early-stopping policy for the tracked epoch loop.

The joint recipe saves the best checkpoint every improving epoch, so stopping early never costs quality
— only compute. Patience must tolerate the observed non-monotonic validation curve (it dips 3-5 epochs,
then recovers to a new best) while still cutting a non-recovering late collapse; default patience 8
clears the longest observed recovering dip. Higher score is better (score = test_acc * test_recall, or
the fold graft).
"""

from __future__ import annotations


class EarlyStop:
    """Tracks the best score seen and how many epochs have passed without a real improvement. `update`
    records a score and reports whether it was a new best; `should_stop` fires once `patience`
    consecutive epochs pass with no gain of at least `min_delta`."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience, self.min_delta = patience, min_delta
        self.best = float("-inf")
        self.waited = 0

    def update(self, score: float) -> bool:
        """Record `score`; return True if it beats the best-so-far by more than `min_delta` (resets the
        wait counter), else False (increments it)."""
        if score > self.best + self.min_delta:
            self.best, self.waited = score, 0
            return True
        self.waited += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.waited >= self.patience
