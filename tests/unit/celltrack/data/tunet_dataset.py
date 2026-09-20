"""Unit test for the detector training frame stream — one item is a normalised frame and its GT centres."""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.data.tunet_dataset import FrameDataset, FrameTarget
from core.data.video import ImageStatistics
from core.paths import DataRoot


def test_getitem(video_store: Path):
    """An item is the quantile-normalised, non-negative frame volume paired with its GT voxel centres."""
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    target = FrameTarget(
        zarr_path=video_store,
        timepoint=0,
        q_low=float(quantiles["0.001"]),
        q_high=float(quantiles["0.999"]),
        coords=np.array([[1, 2, 2]], dtype=np.int64),
    )
    dataset = FrameDataset([target], steps=5, downsample=(1, 1, 1), seed=0)
    assert len(dataset) == 5
    frame, coords = dataset[0]
    assert frame.ndim == 3
    assert torch.all(frame >= 0.0)
    assert coords.tolist() == [[1, 2, 2]]


def test_stream(video_store: Path):
    """The threaded stream yields every item once, in index order, matching direct indexing."""
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    target = FrameTarget(
        zarr_path=video_store,
        timepoint=0,
        q_low=float(quantiles["0.001"]),
        q_high=float(quantiles["0.999"]),
        coords=np.array([[1, 2, 2]], dtype=np.int64),
    )
    dataset = FrameDataset([target], steps=6, downsample=(1, 1, 1), seed=0)
    streamed = list(dataset.stream(threads=3, prefetch=2))
    assert len(streamed) == 6
    for index, (frame, _coords) in enumerate(streamed):
        assert torch.equal(frame, dataset[index][0])


def test_batches(video_store: Path):
    """Batches stack same-shape frames into (b, z, y, x) with ragged per-frame centres; a short tail is kept."""
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    target = FrameTarget(
        zarr_path=video_store,
        timepoint=0,
        q_low=float(quantiles["0.001"]),
        q_high=float(quantiles["0.999"]),
        coords=np.array([[1, 2, 2]], dtype=np.int64),
    )
    dataset = FrameDataset([target], steps=5, downsample=(1, 1, 1), seed=0)
    batches = list(dataset.batches(threads=3, prefetch=2, batch_size=2))
    assert [frames.shape[0] for frames, _ in batches] == [2, 2, 1]  # 5 frames -> 2 + 2 + short 1
    frames, centres = batches[0]
    assert frames.ndim == 4  # (b, z, y, x)
    assert len(centres) == 2 and centres[0].tolist() == [[1, 2, 2]]


def test_index(video_store: Path, geff_store: Path):
    """Indexing a video yields one target per annotated timepoint, centres already in the downsampled grid."""
    targets = FrameTarget.index(DataRoot(geff_store.parent), [video_store], downsample=(1, 2, 2))

    assert [target.timepoint for target in targets] == [0, 1]
    assert targets[0].coords.tolist() == [[0, 0, 0]]  # node 100 at (z, y, x) = (0, 0, 0)
    assert targets[1].coords.tolist() == [[0, 2, 0], [1, 0, 2]]  # (0, 4, 0) and (1, 0, 4) halved in y/x
    assert (targets[0].q_low, targets[0].q_high) == (10.0, 210.0)
