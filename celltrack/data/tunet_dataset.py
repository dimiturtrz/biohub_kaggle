"""The frame stream the temporal-U-Net detector trains on — one annotated frame per item, read lazily.

Each item is a single quantile-normalised, downsampled frame paired with its ground-truth cell centres in
the downsampled grid. Frames are opened per worker at access time (no full video kept in RAM), and the
stream length is a fixed step count rather than the corpus size, so an epoch is a number of optimiser steps.
"""

from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import cast, override

import numpy as np
import torch
import zarr
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from celltrack.data.augmentation import _NO_AUGMENTATION, Augmentation
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.paths import DataRoot


@dataclass(frozen=True)
class FrameTarget:
    """One training frame: where to read it from and the downsampled voxel centres it must fire on."""

    zarr_path: Path
    timepoint: int
    q_low: float
    q_high: float
    coords: Int[np.ndarray, "n 3"]  # downsampled (z, y', x')

    @classmethod
    def index(cls, root: DataRoot, paths: Iterable[Path], downsample: tuple[int, int, int]) -> list["FrameTarget"]:
        """Every annotated frame of `paths`, with its centres already in the downsampled grid the network sees.

        Shared by the trainer and by anything measuring a trained head on the same frames, so a diagnostic
        cannot silently index its inputs differently from the run it is diagnosing.
        """
        targets: list[FrameTarget] = []
        for path in paths:
            quantiles = cast(ImageStatistics, zarr.open_group(path, mode="r").attrs["image_statistics"])["quantiles"]
            q_low, q_high = float(quantiles["0.001"]), float(quantiles["0.999"])
            coordinates = AnnotatedTracks.from_geff(root.track_store(path)).graph.coordinates
            for timepoint in np.unique(coordinates[:, 0]):
                frame_coords = coordinates[coordinates[:, 0] == timepoint][:, 1:] // np.array(downsample)
                targets.append(cls(path, int(timepoint), q_low, q_high, frame_coords.astype(np.int64)))
        return targets


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

    def batches(
        self, threads: int, prefetch: int, batch_size: int
    ) -> Iterator[tuple[Float[Tensor, "b z y x"], list[Int[Tensor, "n 3"]]]]:
        """Group the frame stream into fixed-size batches — frames stacked (one shape), centres left ragged.

        Every fold-train frame is the same downsampled shape, so the volumes stack into a single `(b, z, y, x)`
        tensor the detector forwards in one pass; the per-frame centre counts differ, so those stay a list the
        loss scatters per item. A short final batch is yielded as-is rather than dropped.
        """
        frames: list[Float[Tensor, "z y x"]] = []
        centres: list[Int[Tensor, "n 3"]] = []
        for frame, coords in self.stream(threads, prefetch):
            frames.append(frame)
            centres.append(coords)
            if len(frames) == batch_size:
                yield torch.stack(frames), centres
                frames = []  # a new list, not clear(): the yielded `centres` is held by reference downstream
                centres = []
        if frames:
            yield torch.stack(frames), centres
