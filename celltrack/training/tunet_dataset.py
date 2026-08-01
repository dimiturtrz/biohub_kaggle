"""The frame stream the temporal-U-Net detector trains on — one annotated frame per item, read lazily.

Each item is a single quantile-normalised, downsampled frame paired with its ground-truth cell centres in
the downsampled grid. Frames are opened per worker at access time (no full video kept in RAM), and the
stream length is a fixed step count rather than the corpus size, so an epoch is a number of optimiser steps.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import torch
import zarr
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameTarget:
    """One training frame: where to read it from and the downsampled voxel centres it must fire on."""

    zarr_path: Path
    timepoint: int
    q_low: float
    q_high: float
    coords: Int[np.ndarray, "n 3"]  # downsampled (z, y', x')


class FrameDataset(Dataset[tuple[Tensor, Tensor]]):
    """A stream of `steps` frames sampled uniformly from the annotated frames, opened lazily per worker."""

    def __init__(self, targets: list[FrameTarget], steps: int, downsample: tuple[int, int, int], seed: int) -> None:
        self._targets = targets
        self._steps = steps
        self._downsample = downsample
        self._seed = seed

    def __len__(self) -> int:
        return self._steps

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Sample one frame (seeded by index) and return its normalised volume and GT voxel centres."""
        rng = np.random.default_rng(self._seed + index)
        target = self._targets[int(rng.integers(len(self._targets)))]
        dz, dy, dx = self._downsample
        array = zarr.open_array(target.zarr_path / "0")
        raw = np.asarray(array[target.timepoint, ::dz, ::dy, ::dx], dtype=np.float32)
        normed = (torch.from_numpy(raw) - target.q_low) / (target.q_high - target.q_low + 1e-6)
        return normed.clamp(0.0), torch.from_numpy(target.coords)
