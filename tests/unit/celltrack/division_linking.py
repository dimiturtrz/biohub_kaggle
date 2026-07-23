import numpy as np

from celltrack.division_linking import DivisionAwareLinker
from core.data.tracks import TrackGraph
from core.geometry import Spacing

LINKER = DivisionAwareLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, division_distance_um=10.0)


def detections(coordinates: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_link():
    """A single chain links exactly as nearest-neighbour would — no spurious division edges."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 1, 0]]))
    assert graph.edges.tolist() == [[0, 10]]


def test_link_recovers_a_division():
    """One parent with two daughters gets both edges — the second is what nearest-neighbour drops."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 3, 0], [1, 0, -3, 0]]))
    assert sorted(graph.edges.tolist()) == [[0, 10], [0, 20]]


def test_link_caps_a_parent_at_two_children():
    """A cell has at most two daughters — a third nearby unmatched cell gets no edge."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 2, 0], [1, 0, -2, 0], [1, 0, 4, 0]]))
    assert sum(1 for source, _ in graph.edges.tolist() if source == 0) == 2


def test_link_ignores_a_daughter_beyond_the_division_radius():
    """An unmatched cell too far from any parent is left unlinked, not forced into a division."""
    linker = DivisionAwareLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=5.0, division_distance_um=5.0)
    graph = linker.link(detections([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 50, 0]]))
    assert graph.edges.tolist() == [[0, 10]]
