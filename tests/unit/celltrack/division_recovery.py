import numpy as np

from celltrack.division_recovery import DivisionRecovery
from core.data.tracks import TrackGraph
from core.geometry import Spacing


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2) if edges else np.empty((0, 2), dtype=np.int64),
    )


def recovery(**overrides: float) -> DivisionRecovery:
    settings: dict[str, float] = {"parent_gate_um": 10.5, "sister_gate_um": 8.0, "max_added_fraction": 1.0}
    return DivisionRecovery(spacing=Spacing(z=1.0, y=1.0, x=1.0), **(settings | overrides))


def test_transform():
    """An unparented cell beside a single-child parent becomes that parent's second daughter."""
    mitosis = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -1]], [[0, 1]])
    forked = recovery().transform(mitosis)
    assert forked.edges.tolist() == [[0, 10], [0, 20]]
    assert forked.division_parents().tolist() == [0]


def test_transform_rejects_a_daughter_whose_sister_is_far_away():
    """Proximity to the mother is not enough — a real mitosis leaves its two products adjacent."""
    lost_link = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -9]], [[0, 1]])
    assert recovery().transform(lost_link).edges.tolist() == [[0, 10]]


def test_transform_rejects_a_daughter_beyond_the_parent_gate():
    """A cell too far from any mother is an appearance, not a division."""
    distant = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 20]], [[0, 1]])
    assert recovery().transform(distant).edges.tolist() == [[0, 10]]


def test_transform_never_gives_a_parent_a_third_child():
    """A node that already forked is not a candidate — the metric's divisions are binary."""
    already_forked = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -1], [1, 0, 1, 0]], [[0, 1], [0, 2]])
    assert recovery().transform(already_forked).edges.tolist() == [[0, 10], [0, 20]]


def test_transform_honours_the_global_cap():
    """The cap is what stops a permissive gate from inventing forks across the whole video."""
    mitosis = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -1]], [[0, 1]])
    assert recovery(max_added_fraction=0.0).transform(mitosis).edges.tolist() == [[0, 10]]


def test_transform_prefers_the_nearer_mother_when_two_compete():
    """With one orphan and two eligible mothers the closer one gets the daughter."""
    contested = graph(
        [[0, 0, 0, 0], [0, 0, 0, 6], [1, 0, 0, 1], [1, 0, 0, 5], [1, 0, 0, 4]],
        [[0, 2], [1, 3]],
    )
    forked = recovery().transform(contested)
    assert forked.edges.tolist() == [[0, 20], [10, 30], [10, 40]]
