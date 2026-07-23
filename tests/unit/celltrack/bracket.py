from pathlib import Path

import numpy as np

from celltrack.bracket import BracketResult, Ceiling, ValidationFold
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.geometry import Spacing
from core.metrics.edges import EdgeCounts
from core.paths import DataRoot

SPACING = Spacing(z=1.0, y=1.0, x=1.0)


def a_track(length: int) -> AnnotatedTracks:
    """A single cell moving one voxel per frame — the linker should recover every step."""
    coordinates = np.array([[t, 0, t, 0] for t in range(length)])
    node_ids = np.arange(length)
    edges = np.stack([node_ids[:-1], node_ids[1:]], axis=1)
    graph = TrackGraph(node_ids=node_ids, coordinates=coordinates, edges=edges)
    return AnnotatedTracks(graph=graph, estimated_node_count=length * 10)


def test_edge_jaccard():
    result = BracketResult(label="x", edges=EdgeCounts(tp=8, fp=1, fn=1), predicted_nodes=9, annotated_nodes=9)
    assert result.edge_jaccard() == 0.8


def test_report():
    result = BracketResult(label="ceiling", edges=EdgeCounts(tp=8, fp=1, fn=1), predicted_nodes=9, annotated_nodes=9)
    assert "ceiling" in result.report()
    assert "edge_jaccard=0.8000" in result.report()


def test_with_gate():
    ceiling = Ceiling.with_gate(SPACING, gate_um=12.0)
    assert ceiling.linker.max_distance_um == 12.0


def test_over():
    """On clean, well-separated tracks the ceiling recovers every link — the linker is not the limit."""
    ceiling = Ceiling.with_gate(SPACING, gate_um=10.0)
    result = ceiling.over([a_track(5), a_track(4)])
    assert result.edges.fn == 0
    assert result.edge_jaccard() == 1.0
    assert result.predicted_nodes == result.annotated_nodes == 9


def test_over_cannot_recover_a_division():
    """A one-to-one linker gives a parent one child, so the second daughter edge is always a false negative."""
    parent = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 9, 0]])
    dividing = TrackGraph(node_ids=np.array([0, 1, 2]), coordinates=parent, edges=np.array([[0, 1], [0, 2]]))
    truth = AnnotatedTracks(graph=dividing, estimated_node_count=30)
    result = Ceiling.with_gate(SPACING, gate_um=10.0).over([truth])
    assert result.edges.fn == 1


def test_load(dataset_root: Path):
    """The fold reads its held-out annotations and the voxel spacing off disk in one place."""
    fold = ValidationFold.load(DataRoot(dataset_root), fold=0)
    assert len(fold.annotations) == 2
    assert fold.spacing == Spacing(z=1.625, y=0.40625, x=0.40625)
