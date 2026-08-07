import numpy as np

from celltrack.linkers.linking import NearestNeighbourLinker
from core.data.tracks import TrackGraph
from core.geometry import Spacing

LINKER = NearestNeighbourLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0)


def detections(coordinates: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_link():
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 1, 0]]))
    assert graph.edges.tolist() == [[0, 10]]


def test_link_returns_the_detections_unchanged():
    """The linker owns the edges, never the nodes — detections pass through untouched."""
    source = detections([[0, 0, 0, 0], [1, 0, 1, 0]])
    linked = LINKER.link(source)
    assert linked.node_ids.tolist() == source.node_ids.tolist()
    assert linked.coordinates.tolist() == source.coordinates.tolist()


def test_link_gates_at_the_physical_radius():
    """A successor beyond the gate is left unlinked rather than joined across the frame."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 50, 0]]))
    assert graph.edges.tolist() == []


def test_link_measures_the_gate_in_micrometres():
    """Five voxels of z is 20 um under 4 um voxels — outside a 10 um gate, unlike the same gap in y."""
    linker = NearestNeighbourLinker(spacing=Spacing(z=4.0, y=1.0, x=1.0), max_distance_um=10.0)
    assert linker.link(detections([[0, 0, 0, 0], [1, 5, 0, 0]])).edges.tolist() == []
    assert linker.link(detections([[0, 0, 0, 0], [1, 0, 5, 0]])).edges.tolist() == [[0, 10]]


def test_link_is_one_to_one():
    """Two cells cannot both claim the same successor — the assignment keeps the cheaper pairing."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [0, 0, 8, 0], [1, 0, 1, 0]]))
    assert graph.edges.tolist() == [[0, 20]]


def test_link_assigns_optimally_not_greedily():
    """Greedy nearest would steal the shared target; the optimal assignment pairs both cells."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [0, 0, 4, 0], [1, 0, 3, 0], [1, 0, 6, 0]]))
    assert sorted(graph.edges.tolist()) == [[0, 20], [10, 30]]


def test_link_leaves_a_single_frame_without_edges():
    assert LINKER.link(detections([[0, 0, 0, 0], [0, 0, 5, 0]])).edges.tolist() == []
