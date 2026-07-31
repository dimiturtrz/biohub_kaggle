from pathlib import Path

import numpy as np

from core.data.video import CellVideo, IntensityQuantiles

QUANTILES = IntensityQuantiles(levels={0.001: 10.0, 0.999: 210.0})


def test_from_ome_zarr(video_store: Path):
    """Spacing comes from the store's own OME metadata rather than being assumed."""
    opened = CellVideo.from_ome_zarr(video_store)
    assert (opened.spacing.z, opened.spacing.y, opened.spacing.x) == (1.625, 0.40625, 0.40625)


def test_timepoint_count(video_store: Path):
    assert CellVideo.from_ome_zarr(video_store).timepoint_count == 3


def test_volume_shape(video_store: Path):
    assert CellVideo.from_ome_zarr(video_store).volume_shape == (2, 4, 4)


def test_frame(video_store: Path):
    frame = CellVideo.from_ome_zarr(video_store).frame(1)
    assert frame.shape == (2, 4, 4)
    assert frame[0, 0, 0] == 32


def test_window(video_store: Path):
    """A window is the sub-volume at an origin — the same voxels a full-frame crop would keep."""
    video = CellVideo.from_ome_zarr(video_store)
    window = video.window(1, origin=(0, 1, 1), size=(2, 2, 2))
    assert window.shape == (2, 2, 2)
    assert window.tolist() == video.frame(1)[0:2, 1:3, 1:3].tolist()


def test_normalised_frames(video_store: Path):
    """Every timepoint is yielded once, each clipped to [0, 1] by the shipped quantiles."""
    frames = list(CellVideo.from_ome_zarr(video_store).normalised_frames(0.001, 0.999))
    assert len(frames) == 3
    assert all(frame.min() >= 0.0 and frame.max() <= 1.0 for frame in frames)


def test_robust_range():
    assert QUANTILES.robust_range(0.001, 0.999) == (10.0, 210.0)


def test_normalise():
    frame = np.array([[[10, 110, 210]]], dtype=np.uint16)
    assert QUANTILES.normalise(frame, 0.001, 0.999).ravel().tolist() == [0.0, 0.5, 1.0]


def test_normalise_clips_the_outlier_tails():
    """The raw uint16 range is dominated by a few bright voxels, so the tails clip rather than rescale."""
    frame = np.array([[[0, 4000]]], dtype=np.uint16)
    assert QUANTILES.normalise(frame, 0.001, 0.999).ravel().tolist() == [0.0, 1.0]
