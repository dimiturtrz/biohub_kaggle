import numpy as np

from celltrack.short_track_filter import ShortTrackFilter, ShortTrackRescue, ShortTrackRescueConfig
from core.data.tracks import TrackGraph
from core.geometry import Spacing

ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)


class _Affinity:
    """A fixed per-gap probability matrix indexed by timepoint — the edge scorer's `EdgeAffinity` contract."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2) if edges else np.empty((0, 2), dtype=np.int64),
    )


def test_transform():
    """A short division-free fragment is dropped while a long chain survives."""
    chain_and_fragment = graph(
        [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [0, 9, 9, 9], [1, 9, 9, 9]],
        [[0, 1], [1, 2], [2, 3], [4, 5]],
    )
    kept = ShortTrackFilter(min_length=3).transform(chain_and_fragment)
    assert set(kept.node_ids.tolist()) == {0, 10, 20, 30}
    assert kept.edges.tolist() == [[0, 10], [10, 20], [20, 30]]


def test_transform_keeps_a_short_component_with_a_division():
    """A component below the length floor is kept when it contains a division — mitoses are too valuable to cut."""
    division = graph([[0, 0, 0, 0], [1, 0, 3, 0], [1, 0, -3, 0]], [[0, 1], [0, 2]])
    kept = ShortTrackFilter(min_length=5).transform(division)
    assert set(kept.node_ids.tolist()) == {0, 10, 20}


def _rescue(min_mean_probability: float) -> ShortTrackRescue:
    # budget_fraction 1.0 so the two-node fixture (min(0.006*2, cap) would round to 0) can rescue one component.
    config = ShortTrackRescueConfig(
        min_mean_probability=min_mean_probability,
        max_mean_step_um=2.75,
        max_length=3,
        budget_fraction=1.0,
    )
    affinity = _Affinity({0: np.array([[0.95]], dtype=np.float64)})
    return config.build(ISOTROPIC, affinity)


def test_protected():
    """A short, confident, tight-step component is protected; the same component below the floor is not."""
    fragment = graph([[0, 0, 0, 0], [1, 0, 0, 1]], [[0, 1]])  # one 2-node component, a 1um step, P=0.95
    labels = np.zeros(2, dtype=np.int64)
    sizes = np.array([2], dtype=np.int64)
    kept = np.array([False])

    assert _rescue(0.90).protected(fragment, labels, sizes, kept).tolist() == [True]  # 0.95 >= 0.90 floor
    assert _rescue(0.99).protected(fragment, labels, sizes, kept).tolist() == [False]  # 0.95 < 0.99 floor


def test_transform_rescues_a_confident_short_track():
    """With a rescue configured, a short high-probability tight fragment survives the length cut."""
    fragment = graph([[0, 0, 0, 0], [1, 0, 0, 1]], [[0, 1]])
    dropped = ShortTrackFilter(min_length=5).transform(fragment)
    rescued = ShortTrackFilter(min_length=5, rescue=_rescue(0.90)).transform(fragment)
    assert dropped.node_ids.size == 0  # blunt rule drops the 2-node fragment
    assert set(rescued.node_ids.tolist()) == {0, 10}  # the rescue keeps it


def test_build():
    """`build` binds the video's spacing and affinity to a rescue carrying the configured gates."""
    rescue = ShortTrackRescueConfig(max_length=4, budget_cap=7).build(ISOTROPIC, _Affinity({}))
    assert rescue.max_length == 4
    assert rescue.budget_cap == 7
    assert rescue.spacing is ISOTROPIC
