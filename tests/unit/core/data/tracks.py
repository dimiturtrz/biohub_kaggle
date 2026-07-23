from pathlib import Path

import numpy as np

from core.data.tracks import Adjacency, AnnotatedTracks, TrackGraph
from core.geometry import Spacing

UNIT = Spacing(z=1.0, y=1.0, x=1.0)


def test_timepoints(graph: TrackGraph):
    assert graph.timepoints().tolist() == [0, 1]


def test_positions(graph: TrackGraph):
    assert graph.positions().tolist() == [[0, 0, 0], [0, 4, 0]]


def test_rows_of(graph: TrackGraph):
    """Node ids are sparse arbitrary integers, so a lookup is a search rather than an index."""
    assert graph.rows_of(np.array([9, 7])).tolist() == [1, 0]


def test_division_parents(graph: TrackGraph):
    assert graph.division_parents().tolist() == []


def test_division_parents_finds_a_fork(geff_store: Path):
    assert AnnotatedTracks.from_geff(geff_store).graph.division_parents().tolist() == [100]


def test_link_displacements(graph: TrackGraph):
    assert graph.link_displacements(UNIT).tolist() == [4.0]


def test_link_displacements_are_per_timepoint(graph: TrackGraph):
    """A link spanning two timepoints moved half as fast as the same gap spanning one."""
    gapped = TrackGraph(node_ids=graph.node_ids, coordinates=np.array([[0, 0, 0, 0], [2, 0, 4, 0]]), edges=graph.edges)
    assert gapped.link_displacements(UNIT).tolist() == [2.0]


def test_link_displacements_scale_anisotropically(graph: TrackGraph):
    assert graph.link_displacements(Spacing(z=1.0, y=0.5, x=1.0)).tolist() == [2.0]


def test_from_geff(geff_store: Path):
    tracks = AnnotatedTracks.from_geff(geff_store)
    assert tracks.graph.node_ids.tolist() == [100, 200, 300]
    assert tracks.graph.coordinates.tolist() == [[0, 0, 0, 0], [1, 0, 4, 0], [1, 1, 0, 4]]
    assert tracks.graph.edges.tolist() == [[100, 200], [100, 300]]
    assert tracks.estimated_node_count == 30


def test_edge_rows(graph: TrackGraph):
    """Edges arrive as node ids; row indices are what actually index the coordinate arrays."""
    assert graph.edge_rows().tolist() == [[0, 1]]


def test_without_edges(graph: TrackGraph):
    """Dropping the links keeps the nodes — the shape a detector hands the linker."""
    edgeless = graph.without_edges()
    assert edgeless.node_ids.tolist() == graph.node_ids.tolist()
    assert edgeless.edges.shape == (0, 2)


def forked() -> TrackGraph:
    return TrackGraph(
        node_ids=np.array([5, 6, 7, 8]),
        coordinates=np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 9, 0], [0, 0, 40, 0]]),
        edges=np.array([[5, 6], [5, 7]]),
    )


def test_of():
    adjacency = Adjacency.of(forked())
    assert adjacency.successors[0] == (1, 2)
    assert adjacency.predecessors[2] == (0,)
    assert adjacency.out_degrees.tolist() == [2, 0, 0, 0]
    assert adjacency.in_degrees.tolist() == [0, 1, 1, 0]


def test_dividing_rows():
    assert Adjacency.of(forked()).dividing_rows().tolist() == [0]


def test_components():
    """The unlinked fourth node is its own lineage fragment — division evidence cannot cross fragments."""
    labels = Adjacency.of(forked()).components().tolist()
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_annotated_fraction(geff_store: Path):
    """Three labelled nodes against thirty estimated real cells — the supervision is a thin sample."""
    assert AnnotatedTracks.from_geff(geff_store).annotated_fraction == 0.1
