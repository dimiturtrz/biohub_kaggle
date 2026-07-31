import numpy as np

from celltrack.short_track_filter import ShortTrackFilter
from core.data.tracks import TrackGraph


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
