from pathlib import Path

from celltrack.inference import LearnedDetector
from celltrack.peaks import PeakExtractor
from celltrack.training.model import DetectionUNet
from core.data.video import CellVideo
from core.geometry import Spacing


def a_detector() -> LearnedDetector:
    return LearnedDetector(
        model=DetectionUNet(window=1, width=4),
        peaks=PeakExtractor(spacing=Spacing(z=1.0, y=1.0, x=1.0), scale_um=2.0),
        window=1,
    )


def test_detect(video_store: Path):
    """The learned detector runs the network over every timepoint and returns an edgeless centre graph."""
    detections = a_detector().detect(CellVideo.from_ome_zarr(video_store), keep=6)
    assert detections.edges.shape == (0, 2)
    assert detections.coordinates.shape[1] == 4
    assert set(detections.timepoints().tolist()) <= set(range(CellVideo.from_ome_zarr(video_store).timepoint_count))
