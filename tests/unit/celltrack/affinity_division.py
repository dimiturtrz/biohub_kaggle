import numpy as np
from jaxtyping import Float

from celltrack.affinity_division import AffinityDivisionRecovery
from core.data.tracks import TrackGraph
from core.geometry import Spacing


class FakeAffinity:
    """An `EdgeAffinity` from an explicit per-gap matrix, rows the t sources, columns the t+1 targets."""

    def __init__(self, by_timepoint: dict[int, Float[np.ndarray, "s t"]]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:
        return self._by_timepoint.get(timepoint)


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2) if edges else np.empty((0, 2), dtype=np.int64),
    )


def recovery(matrix: list[list[float]], **overrides: float) -> AffinityDivisionRecovery:
    settings: dict[str, float] = {
        "min_second_prob": 0.1,
        "parent_gate_um": 5.0,
        "sister_gate_um": 5.0,
        "max_added_fraction": 1.0,
        "min_kept_prob": 0.5,
    }
    return AffinityDivisionRecovery(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        affinity=FakeAffinity({0: np.array(matrix, dtype=np.float32)}),
        **(settings | overrides),
    )


# One parent (row 0) at t=0; two targets at t=1 — the kept child (row 1) and an orphan runner-up (row 2).
_MITOSIS = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 2]]
_KEPT_TOP = [[0.7, 0.2]]


def test_transform():
    """The parent's top target is its kept child and its second is an unparented cell — a fork is proposed."""
    forked = recovery(_KEPT_TOP).transform(graph(_MITOSIS, [[0, 1]]))
    assert forked.edges.tolist() == [[0, 10], [0, 20]]
    assert forked.division_parents().tolist() == [0]


def test_rejects_when_runner_up_is_below_the_floor():
    """A second target the head barely scores is not a daughter — the probability floor rejects it."""
    assert recovery([[0.7, 0.05]]).transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_when_kept_child_is_not_the_top_target():
    """If the head's first choice is not the linker's kept child the parent is not a confident divider."""
    assert recovery([[0.2, 0.7]]).transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_when_kept_child_probability_is_low():
    """A parent whose own primary link the head is unsure of does not earn a second daughter."""
    low_kept = recovery([[0.4, 0.35]], min_kept_prob=0.5)
    assert low_kept.transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_a_runner_up_that_is_already_parented():
    """Only an unparented cell can be a missing second daughter — a linked target is someone's child already."""
    # Row 3 (t=0, far) already links to row 2, so the runner-up is not an orphan.
    coordinates = [*_MITOSIS, [0, 0, 0, 20]]
    two_source_top = [[0.7, 0.2], [0.1, 0.6]]
    forked = recovery(two_source_top).transform(graph(coordinates, [[0, 1], [3, 2]]))
    assert forked.edges.tolist() == [[0, 10], [30, 20]]


def test_rejects_a_runner_up_beyond_the_sister_gate():
    """The two daughters must sit close — a runner-up far from the kept child fails the sister gate."""
    far_sister = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 4]]
    assert recovery(_KEPT_TOP, sister_gate_um=2.0).transform(graph(far_sister, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_rejects_a_runner_up_beyond_the_parent_gate():
    """A candidate too far from the mother is an appearance, not a division."""
    far_parent = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 4]]
    gated = recovery(_KEPT_TOP, parent_gate_um=2.0, sister_gate_um=10.0)
    assert gated.transform(graph(far_parent, [[0, 1]])).edges.tolist() == [[0, 10]]


def test_honours_the_global_cap():
    """A zero budget adds nothing however confident the head is — the cap bounds speculation."""
    assert recovery(_KEPT_TOP, max_added_fraction=0.0).transform(graph(_MITOSIS, [[0, 1]])).edges.tolist() == [[0, 10]]
