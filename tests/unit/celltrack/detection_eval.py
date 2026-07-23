from pathlib import Path

import pytest

from celltrack.detection_eval import Checkpoint, DetectionComparison, RecallComparison
from celltrack.training.model import DetectionUNet
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.recall import DetectionRecall

ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)


def a_checkpoint() -> Checkpoint:
    return Checkpoint(state_dict=DetectionUNet(window=1, width=4).state_dict(), strides=[[2, 2, 2]], window=1, width=4)


def test_from_checkpoint():
    """A comparison rebuilds the learned detector from a checkpoint's structure and weights."""
    comparison = DetectionComparison.from_checkpoint(ISOTROPIC, a_checkpoint(), device="cpu")
    assert comparison.learned.window == 1
    assert comparison.classical.scale_um > 0


def test_compare(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """One video is scored by both detectors at its estimated node count, tagged with its prefix."""
    comparison = DetectionComparison.from_checkpoint(ISOTROPIC, a_checkpoint(), device="cpu")
    result = comparison.compare("6bba", CellVideo.from_ome_zarr(video_store), in_bounds_tracks)
    assert result.prefix == "6bba"
    assert result.learned.total == len(in_bounds_tracks.graph.node_ids)


def test_report(caplog: pytest.LogCaptureFixture):
    """The report names the winner per acquisition."""
    comparisons = [RecallComparison("6bba", DetectionRecall(found=1, total=2), DetectionRecall(found=2, total=2))]
    with caplog.at_level("INFO"):
        DetectionComparison.report(comparisons)
    assert "learned wins" in caplog.text
