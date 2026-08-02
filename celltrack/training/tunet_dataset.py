"""The frame stream the temporal-U-Net detector trains on — one annotated frame per item, read lazily.

Each item is a single quantile-normalised, downsampled frame paired with its ground-truth cell centres in
the downsampled grid. Frames are opened per worker at access time (no full video kept in RAM), and the
stream length is a fixed step count rather than the corpus size, so an epoch is a number of optimiser steps.
"""

from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import override

import numpy as np
import torch
import zarr
from jaxtyping import Float, Int
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


@dataclass(frozen=True)
class Augmentation:
    """Label-preserving frame perturbations — intensity jitter and axis flips — the pilkwang recipe has and ours lacks.

    A crop centred on a cell teaches "fire here"; without variation the detector memorises the exact intensity
    and orientation it saw. Multiplicative + additive intensity jitter breaks the brightness dependence, and a
    random flip along each eligible axis multiplies the effective orientations — coordinates are mirrored with
    the volume so a flipped frame stays correctly labelled. Defaults are the identity, so a run that does not
    ask for augmentation is byte-for-byte the un-augmented baseline.
    """

    brightness: float = 0.0  # multiplicative jitter half-range: gain in [1 - b, 1 + b]
    offset: float = 0.0  # additive jitter half-range: bias in [-o, o]
    flip_axes: tuple[int, ...] = ()  # frame axes (0=z, 1=y, 2=x) each flipped independently

    _FLIP_PROBABILITY = 0.5

    def apply(
        self, frame: Float[Tensor, "z y x"], coords: Int[Tensor, "n 3"], rng: np.random.Generator
    ) -> tuple[Float[Tensor, "z y x"], Int[Tensor, "n 3"]]:
        """Return the jittered, randomly-flipped frame with its centres mirrored to match."""
        gain = 1.0 + rng.uniform(-self.brightness, self.brightness)
        bias = rng.uniform(-self.offset, self.offset)
        frame = (frame * gain + bias).clamp(0.0)
        for axis in self.flip_axes:
            if rng.random() < self._FLIP_PROBABILITY:
                frame = torch.flip(frame, dims=(axis,))
                coords = coords.clone()
                coords[:, axis] = frame.shape[axis] - 1 - coords[:, axis]
        return frame, coords


_NO_AUGMENTATION = Augmentation()  # frozen identity default; a module singleton avoids a call in arg defaults


class FrameDataset(Dataset[tuple[Tensor, Tensor]]):
    """A stream of `steps` frames sampled uniformly from the annotated frames, opened lazily per worker."""

    def __init__(
        self,
        targets: list[FrameTarget],
        steps: int,
        downsample: tuple[int, int, int],
        seed: int,
        augmentation: Augmentation = _NO_AUGMENTATION,
    ) -> None:
        self._targets = targets
        self._steps = steps
        self._downsample = downsample
        self._seed = seed
        self._augmentation = augmentation

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
        return self._augmentation.apply(normed.clamp(0.0), torch.from_numpy(target.coords), rng)

    def stream(self, threads: int, prefetch: int) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield every item once, decompressed by a pool of `threads`, keeping `prefetch` frames in flight.

        zarr's blosc/zstd decompression releases the GIL, so worker *threads* parallelise it across cores with
        no process spawn — the throughput of DataLoader workers without the Windows worker-pool deadlock, and
        with the GPU overlapping the reads instead of stalling on them. Items are yielded in index order (so a
        run stays reproducible) even though they finish decompressing out of order.
        """
        with ThreadPoolExecutor(max_workers=threads) as pool:
            indices = iter(range(self._steps))
            pending: deque[Future[tuple[Tensor, Tensor]]] = deque()
            for index in islice(indices, prefetch):
                pending.append(pool.submit(self.__getitem__, index))
            for index in indices:
                item = pending.popleft().result()
                pending.append(pool.submit(self.__getitem__, index))
                yield item
            while pending:
                yield pending.popleft().result()
