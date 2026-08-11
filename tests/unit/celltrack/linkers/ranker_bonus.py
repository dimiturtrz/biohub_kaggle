import numpy as np

from celltrack.linkers.ranker_bonus import RankerBonus


class _Ranker:
    """The re-ranker probabilities for one gap — the `EdgeAffinity` shape the cost term consumes."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


def test_discount():
    """Each pair's cost drops by `bonus * P_ranker` — the re-ranker term, subtracted like the other two."""
    term = RankerBonus(ranker=_Ranker({0: np.array([[0.9, 0.2]])}), bonus=10.0)

    assert term.discount(0, np.array([[1.0, 4.0]])).tolist() == [[1.0 - 9.0, 4.0 - 2.0]]


def test_discount_leaves_an_unscored_gap_to_the_other_terms():
    """A gap the ranker does not score keeps the cost it arrived with — the forward term's own rule."""
    term = RankerBonus(ranker=_Ranker({}), bonus=10.0)
    cost = np.array([[1.0, 4.0]])

    assert term.discount(3, cost).tolist() == cost.tolist()


def test_discount_at_zero_bonus_is_inert():
    """At a zero weight the term is arithmetically absent — the three-term cost collapses to the shipped one."""
    term = RankerBonus(ranker=_Ranker({0: np.array([[0.9, 0.2]])}), bonus=0.0)
    cost = np.array([[1.0, 4.0]])

    assert term.discount(0, cost).tobytes() == cost.tobytes()
