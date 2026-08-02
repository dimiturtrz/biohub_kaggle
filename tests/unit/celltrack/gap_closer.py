import numpy as np

from celltrack.gap_closer import GapCloser
from core.data.tracks import TrackGraph
from core.geometry import Spacing


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2) if edges else np.empty((0, 2), dtype=np.int64),
    )


def closer(**overrides: float) -> GapCloser:
    settings: dict[str, float] = {"gate_um": 10.0, "reuse_um": 5.0, "max_added_fraction": 1.0}
    return GapCloser(spacing=Spacing(z=1.0, y=1.0, x=1.0), **(settings | overrides))


def test_transform():
    """A track ending at t, an isolated node at t+1 near the midpoint, and a start at t+2 are bridged."""
    # nodes: P->E (end at t1), M isolated at t2, S start at t3; all colinear at the origin.
    gapped = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 0]], [[0, 1]])
    bridged = closer().transform(gapped).edges.tolist()
    assert [10, 30] in bridged and [30, 20] in bridged


def test_transform_leaves_a_gap_open_without_a_reusable_node():
    """No isolated node near the midpoint means the gap is left open, not filled with an invention."""
    far_middle = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 50]], [[0, 1]])
    assert closer().transform(far_middle).edges.tolist() == [[0, 10]]


def test_transform_rejects_an_end_and_start_beyond_the_gate():
    """Two track fragments too far apart to be the same cell are not joined."""
    far_apart = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 50], [2, 0, 0, 25]], [[0, 1]])
    assert closer().transform(far_apart).edges.tolist() == [[0, 10]]


def test_transform_honours_the_cap():
    """A zero budget adds no bridges however reconnectable the gap is."""
    gapped = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 0]], [[0, 1]])
    assert closer(max_added_fraction=0.0).transform(gapped).edges.tolist() == [[0, 10]]
