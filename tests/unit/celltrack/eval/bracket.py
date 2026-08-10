from pathlib import Path

import numpy as np

from celltrack.eval.bracket import ArmResult, Ceiling, Floor
from celltrack.postproc.division_recovery import DivisionRecovery
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.divisions import DivisionCounts
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import VideoMetrics

SPACING = Spacing(z=1.0, y=1.0, x=1.0)
MATCHER = DistanceMatcher(spacing=SPACING)


def a_track(length: int) -> AnnotatedTracks:
    """A single cell moving one voxel per frame — the linker should recover every step."""
    coordinates = np.array([[t, 0, t, 0] for t in range(length)])
    node_ids = np.arange(length)
    edges = np.stack([node_ids[:-1], node_ids[1:]], axis=1)
    graph = TrackGraph(node_ids=node_ids, coordinates=coordinates, edges=edges)
    return AnnotatedTracks(graph=graph, estimated_node_count=length * 10)


def a_metric(edges: EdgeCounts, predicted: int, estimated: int) -> VideoMetrics:
    return VideoMetrics(
        edges=edges, divisions=DivisionCounts(tp=0, fp=0, fn=0), predicted_nodes=predicted, estimated_nodes=estimated
    )


def test_edge_jaccard():
    arm = ArmResult(label="x", metrics=(a_metric(EdgeCounts(tp=8, fp=1, fn=1), 9, 90),))
    assert arm.edge_jaccard() == 0.8


def test_score():
    arm = ArmResult(label="x", metrics=(a_metric(EdgeCounts(tp=9, fp=0, fn=1), 100, 100),))
    assert arm.score().score == 0.9


def test_predicted_nodes():
    arm = ArmResult(label="x", metrics=(a_metric(EdgeCounts(tp=1, fp=0, fn=0), 40, 90),))
    assert arm.predicted_nodes() == 40


def test_estimated_nodes():
    arm = ArmResult(label="x", metrics=(a_metric(EdgeCounts(tp=1, fp=0, fn=0), 40, 90),))
    assert arm.estimated_nodes() == 90


def test_report():
    arm = ArmResult(label="ceiling", metrics=(a_metric(EdgeCounts(tp=8, fp=1, fn=1), 9, 90),))
    assert "ceiling" in arm.report()
    assert "edge_jaccard=0.8000" in arm.report()


def test_with_gate():
    ceiling = Ceiling.with_gate(SPACING, gate_um=12.0)
    assert ceiling.linker.max_distance_um == 12.0


def test_with_motion():
    """The motion ceiling labels itself by whether a division-recovery pass follows the linker."""
    plain = Ceiling.with_motion(SPACING, recovery=None)
    assert plain.label == "ceiling-motion"
    assert plain.recovery is None
    divided = Ceiling.with_motion(SPACING, recovery=DivisionRecovery(SPACING, 10.5, 8.0, 0.02))
    assert divided.label == "ceiling-motiondiv"


def test_ceiling_over():
    """On clean, well-separated tracks the ceiling recovers every link — the linker is not the limit."""
    result = Ceiling.with_gate(SPACING, gate_um=10.0).over([a_track(5), a_track(4)])
    assert result.edge_jaccard() == 1.0
    assert result.predicted_nodes() == 9


def test_ceiling_over_cannot_recover_a_division():
    """A one-to-one linker gives a parent one child, so the second daughter edge is always a false negative."""
    parent = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 9, 0]])
    dividing = TrackGraph(node_ids=np.array([0, 1, 2]), coordinates=parent, edges=np.array([[0, 1], [0, 2]]))
    truth = AnnotatedTracks(graph=dividing, estimated_node_count=30)
    result = Ceiling.with_gate(SPACING, gate_um=10.0).over([truth])
    assert result.metrics[0].edges.fn == 1


def test_with_scale_and_gate():
    floor = Floor.with_scale_and_gate(SPACING, scale_um=4.0, gate_um=12.0)
    assert floor.detector.scale_um == 4.0
    assert floor.linker.max_distance_um == 12.0


def test_floor_over(video_store: Path):
    """The floor detects, links and scores end to end — it produces a real score without touching ground truth."""
    cell = CellVideo.from_ome_zarr(video_store)
    truth = a_track(3)
    result = Floor.with_scale_and_gate(SPACING, scale_um=2.0, gate_um=10.0).over([(cell, truth)])
    assert result.label == "floor"
    assert len(result.metrics) == 1
    assert result.metrics[0].estimated_nodes == truth.estimated_node_count
