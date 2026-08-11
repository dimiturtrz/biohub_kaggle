"""Unit tests for the competition corpus's frame reader — the strategy every other corpus is measured against."""

from pathlib import Path

import torch

from celltrack.data.frame_source import FrameSource, ZarrFrames


def test_frame(video_store: Path):
    """A frame is read at the asked-for stride, quantile-windowed and clamped non-negative."""
    frames = ZarrFrames(video_store, q_low=10.0, q_high=210.0)

    full = frames.frame(0, (1, 1, 1))
    strided = frames.frame(0, (1, 2, 2))

    assert tuple(full.shape) == (2, 4, 4)  # the tiny fixture's (Z, Y, X)
    assert tuple(strided.shape) == (2, 2, 2)
    assert torch.equal(strided, full[:, ::2, ::2])
    assert float(full.min()) >= 0.0  # the below-window voxels are clamped, never negative
    assert isinstance(frames, FrameSource)
