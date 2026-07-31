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


def test_detect_above(video_store: Path):
    """Threshold detection returns an edgeless centre graph; the node count emerges from the threshold."""
    detections = a_detector().detect_above(CellVideo.from_ome_zarr(video_store), probability_threshold=0.5)
    assert detections.edges.shape == (0, 2)
    assert detections.coordinates.shape[1] == 4


def test_responses(video_store: Path):
    """One probability volume per timepoint comes back, each a `(z, y, x)` grid the peak-picker reads."""
    video = CellVideo.from_ome_zarr(video_store)
    volumes = list(a_detector().responses(video))
    assert len(volumes) == video.timepoint_count
    assert all(volume.ndim == 3 for volume in volumes)
