from pathlib import Path

import pytest

from celltrack.linking import NearestNeighbourLinker
from celltrack.operating_point import OperatingPointSweep, SweepPoint
from celltrack.training.model import Checkpoint, DetectionUNet
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.divisions import DivisionCounts
from core.metrics.edges import EdgeCounts
from core.metrics.score import VideoMetrics

ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)


def a_checkpoint() -> Checkpoint:
    return Checkpoint(state_dict=DetectionUNet(window=1, width=4).state_dict(), strides=[[2, 2, 2]], window=1, width=4)


def a_point(ratio: float, prefix: str, jaccard_tp: int) -> SweepPoint:
    metrics = VideoMetrics(
        edges=EdgeCounts(tp=jaccard_tp, fp=0, fn=10 - jaccard_tp),
        divisions=DivisionCounts(tp=0, fp=0, fn=0),
        predicted_nodes=100,
        estimated_nodes=100,
    )
    return SweepPoint(ratio=ratio, prefix=prefix, metrics=metrics)


def test_from_checkpoint():
    """The sweep pairs the checkpoint-loaded detector with the nearest-neighbour linker."""
    sweep = OperatingPointSweep.from_checkpoint(ISOTROPIC, a_checkpoint(), gate_um=15.0, device="cpu")
    assert isinstance(sweep.linker, NearestNeighbourLinker)
    assert sweep.linker.max_distance_um == 15.0


def test_at(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """One budget produces one scored point tagged with its ratio and prefix."""
    sweep = OperatingPointSweep.from_checkpoint(ISOTROPIC, a_checkpoint(), gate_um=10.0, device="cpu")
    point = sweep.at(0.5, "6bba", CellVideo.from_ome_zarr(video_store), in_bounds_tracks)
    assert point.ratio == 0.5
    assert point.prefix == "6bba"


def test_report(caplog: pytest.LogCaptureFixture):
    """The report names the best-scoring budget per acquisition."""
    points = [a_point(0.5, "6bba", 4), a_point(1.0, "6bba", 8)]
    with caplog.at_level("INFO"):
        OperatingPointSweep.report(points)
    assert "best" in caplog.text
