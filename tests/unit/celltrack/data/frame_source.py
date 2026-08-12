"""Unit tests for the competition corpus's frame reader — the strategy every other corpus is measured against."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from celltrack.data.frame_source import DOWNSAMPLE_ATTR, FrameSource, PooledFrames, ZarrFrames


def test_zarr_frames_frame(video_store: Path):
    """A frame is read at the asked-for stride, quantile-windowed and clamped non-negative."""
    frames = ZarrFrames(video_store, q_low=10.0, q_high=210.0)

    full = frames.frame(0, (1, 1, 1))
    strided = frames.frame(0, (1, 2, 2))

    assert tuple(full.shape) == (2, 4, 4)  # the tiny fixture's (Z, Y, X)
    assert tuple(strided.shape) == (2, 2, 2)
    assert torch.equal(strided, full[:, ::2, ::2])
    assert float(full.min()) >= 0.0  # the below-window voxels are clamped, never negative
    assert isinstance(frames, FrameSource)


def test_pooled_frames_frame(tmp_path: Path):
    """A pooled store returns exactly what striding the raw store returns — same voxels, same normalisation.

    This is the property the whole pre-pooling idea rests on: the store is a SPEED change and must not be a
    numerical one, so the two sources are pinned equal rather than merely both plausible.
    """
    raw = np.arange(2 * 4 * 8 * 8, dtype=np.uint16).reshape(2, 4, 8, 8)
    store = tmp_path / "pooled.zarr"
    array = zarr.create_array(store=str(store), shape=(2, 4, 2, 2), chunks=(1, 4, 2, 2), dtype=np.uint16)
    array[:] = raw[:, ::1, ::4, ::4]
    array.attrs[DOWNSAMPLE_ATTR] = [1, 4, 4]

    pooled = PooledFrames.of(store, q_low=10.0, q_high=210.0)
    expected = (torch.from_numpy(raw[0, ::1, ::4, ::4].astype(np.float32)) - 10.0) / (200.0 + 1e-6)

    assert torch.equal(pooled.frame(0, (1, 4, 4)), expected.clamp(0.0))


def test_store():
    """The grid comes from the STORE, so a reader cannot be told the wrong one."""
    assert PooledFrames.store(Path("proc"), (1, 4, 4), "vid") == Path("proc/pooled_1x4x4/vid.zarr")


def test_directory():
    """The grid is in the directory NAME, so two downsamples can never share a store."""
    assert PooledFrames.directory(Path("proc"), (1, 4, 4)).name == "pooled_1x4x4"
    assert PooledFrames.directory(Path("proc"), (1, 2, 2)).name == "pooled_1x2x2"


def test_pooled_frames_refuses_another_grid(tmp_path: Path):
    """Asked for a grid it was not pooled at, it raises instead of re-striding voxels that ARE the stride."""
    store = tmp_path / "pooled.zarr"
    array = zarr.create_array(store=str(store), shape=(1, 2, 2, 2), chunks=(1, 2, 2, 2), dtype=np.uint16)
    array.attrs[DOWNSAMPLE_ATTR] = [1, 4, 4]

    with pytest.raises(ValueError, match="cannot serve"):
        PooledFrames.of(store, 0.0, 1.0).frame(0, (1, 2, 2))


def test_of(tmp_path: Path):
    """`of` takes the grid from the STORE's own provenance, so a reader can never be told the wrong one."""
    store = tmp_path / "provenance.zarr"
    array = zarr.create_array(store=str(store), shape=(1, 1, 1, 1), chunks=(1, 1, 1, 1), dtype=np.uint16)
    array.attrs[DOWNSAMPLE_ATTR] = [1, 4, 4]

    assert PooledFrames.of(store, 0.0, 1.0).downsample == (1, 4, 4)
