import numpy as np

from celltrack.postproc.linefit_smoother import LinefitSmoother
from core.data.tracks import TrackGraph


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2),
    )


def test_transform():
    """An interior node off the line between its neighbours is pulled toward it; the endpoints do not move."""
    zigzag = graph([[0, 0, 0, 0], [1, 0, 10, 0], [2, 0, 0, 0]], [[0, 1], [1, 2]])
    smoothed = LinefitSmoother(strength=0.8).transform(zigzag)
    assert smoothed.coordinates[1].tolist() == [1, 0, 2, 0]  # y: 0.2*10 + 0.8*0
    assert smoothed.coordinates[0].tolist() == [0, 0, 0, 0]
    assert smoothed.coordinates[2].tolist() == [2, 0, 0, 0]
    assert smoothed.edges.tolist() == zigzag.edges.tolist()  # topology untouched
