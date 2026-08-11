import numpy as np

from celltrack.linkers.mutual_bonus import MutualBonus


class _Mutual:
    """The bidirectionally fused probabilities for one gap — the `EdgeAffinity` shape the cost term consumes."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


def test_discount():
    """Each pair's cost drops by `bonus * P_mutual` — the agreement term, subtracted like the forward one."""
    term = MutualBonus(mutual=_Mutual({0: np.array([[0.9, 0.2]])}), bonus=10.0)

    assert term.discount(0, np.array([[1.0, 4.0]])).tolist() == [[1.0 - 9.0, 4.0 - 2.0]]


def test_discount_leaves_an_unscored_gap_to_the_forward_term():
    """A gap the fused affinity does not score keeps the cost it arrived with — the forward term's own rule."""
    term = MutualBonus(mutual=_Mutual({}), bonus=10.0)
    cost = np.array([[1.0, 4.0]])

    assert term.discount(3, cost).tolist() == cost.tolist()


def test_discount_at_zero_bonus_is_inert():
    """At a zero weight the term is arithmetically absent — the two-term cost collapses to the shipped one."""
    term = MutualBonus(mutual=_Mutual({0: np.array([[0.9, 0.2]])}), bonus=0.0)
    cost = np.array([[1.0, 4.0]])

    assert term.discount(0, cost).tobytes() == cost.tobytes()
