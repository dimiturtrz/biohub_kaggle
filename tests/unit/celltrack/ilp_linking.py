import numpy as np

from celltrack.ilp_linking import ILPLinker
from core.data.tracks import TrackGraph
from core.geometry import Spacing

LINKER = ILPLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, division=True)


def detections(coordinates: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_link():
    """A single chain links exactly as nearest-neighbour would — one edge, no spurious division."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 1, 0]]))
    assert graph.edges.tolist() == [[0, 10]]


def test_link_recovers_a_division():
    """One parent with two daughters gets both edges — the topology a one-to-one assignment cannot express."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 3, 0], [1, 0, -3, 0]]))
    assert sorted(graph.edges.tolist()) == [[0, 10], [0, 20]]


def test_link_ignores_a_target_beyond_the_gate():
    """A successor further than the gate is left unlinked rather than forced into an edge."""
    linker = ILPLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=5.0, division=True)
    graph = linker.link(detections([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 50, 0]]))
    assert graph.edges.tolist() == [[0, 10]]
