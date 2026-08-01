"""Unit test for the detector training frame stream — one item is a normalised frame and its GT centres."""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.training.tunet_dataset import FrameDataset, FrameTarget
from core.data.video import ImageStatistics


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
