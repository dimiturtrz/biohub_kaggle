"""Where a training pair's frames come from — the seam a corpus other than the competition's own plugs into.

`PairTarget` used to hold a zarr path and two quantiles, which is a STORAGE recipe wearing the clothes of a
training pair. The pair shape the joint trainer consumes says nothing about storage, and it must not: a corpus
whose candidate sets are COMPLETE — where every crowded near-neighbour is a true negative in the edge matrix
rather than an unannotated detection outside it — is what the measured association failure calls for, and it
is stored as pooled `.npz` volumes rather than OME-Zarr (`celltrack.data.synthetic_pairs`). One protocol, so
the loop cannot tell the two corpora apart.

`downsample` is a parameter of the READ rather than of the source because it is the run's choice, not the
corpus's: `ZarrFrames` strides a full-resolution store by it, while a corpus that ships pre-pooled volumes
refuses any value but the pooling it already carries.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import zarr
from jaxtyping import Float
from torch import Tensor

from celltrack.data.normalisation import HIGH_QUANTILE, LOW_QUANTILE, Normalisation, QuantileWindow


@runtime_checkable
class FrameSource(Protocol):
    """One video's frames, already normalised — the only thing a pair needs to know about how it is stored."""

    def frame(self, timepoint: int, downsample: tuple[int, int, int]) -> Float[Tensor, "z y x"]:
        """That timepoint's normalised, non-negative volume on the grid `downsample` names."""
        ...


DOWNSAMPLE_ATTR = "downsample"
"""The grid a pooled store was written on — read back so it can refuse to serve any other."""


@dataclass(frozen=True)
class PooledFrames:
    """A pre-pooled store's frames: the same normalisation, read off voxels that were strided once, offline.

    `ZarrFrames` strides the raw store on every access, and the raw chunk IS a whole frame (8.39 MB), so a
    downsampled read decompresses sixteen times what it returns — every epoch of every run. This reads a store
    written once at that grid instead: 11.2 ms/frame becomes 5.2 ms, and the store is a sixteenth the voxels.

    It holds the POOLED RAW integers, not the normalised floats, so the normalisation stays in ONE place and a
    change to the quantiles does not invalidate the store. The formula below is therefore the same expression
    `ZarrFrames` applies, and the two are pinned byte-identical by test.

    `downsample` is what the store was WRITTEN at, and any other grid is refused rather than silently served:
    the array's voxels already are the stride, so re-striding them would return a different grid under the same
    name. `SyntheticPairs` refuses the same way for the same reason.
    """

    array_path: Path
    q_low: float
    q_high: float
    downsample: tuple[int, int, int]
    # The preprocessing layer, selectable per run. Defaults to the reference's own window, so a caller that
    # does not ask reads exactly what every arm before it read — and what the MOUNTED weights were trained on.
    normalisation: Normalisation = field(default_factory=QuantileWindow)

    @classmethod
    def of(cls, store: Path, q_low: float, q_high: float) -> "PooledFrames":
        """Open a pooled store, taking the grid from the store itself rather than from the caller."""
        written = zarr.open_array(store).attrs[DOWNSAMPLE_ATTR]
        return cls(store, q_low, q_high, tuple(int(axis) for axis in written))  # type: ignore[arg-type]

    @staticmethod
    def directory(processed: Path, downsample: tuple[int, int, int]) -> Path:
        """Where stores for one grid live — the grid is IN THE NAME so two grids cannot be confused.

        The convention lives here rather than in the tool that writes it, so the reader and the writer cannot
        disagree about where a store is, and so `celltrack` never has to import from `tools`.
        """
        return processed / f"pooled_{'x'.join(str(axis) for axis in downsample)}"

    @staticmethod
    def store(processed: Path, downsample: tuple[int, int, int], stem: str) -> Path:
        """One video's pooled store path — present only if it has been written for this grid."""
        return PooledFrames.directory(processed, downsample) / f"{stem}.zarr"

    def frame(self, timepoint: int, downsample: tuple[int, int, int]) -> Float[Tensor, "z y x"]:
        """One quantile-normalised frame — refusing any grid but the one this store was pooled at."""
        if tuple(downsample) != self.downsample:
            message = (
                f"pooled store at {self.array_path.name} holds the {self.downsample} grid, so it cannot serve "
                f"{tuple(downsample)}; pool a store at that grid or read the raw video"
            )
            raise ValueError(message)
        raw = np.asarray(zarr.open_array(self.array_path)[timepoint], dtype=np.float32)
        return torch.from_numpy(self.normalisation.apply(raw, self._quantiles()))

    def _quantiles(self) -> dict[str, float]:
        """This video's shipped intensity quantiles, as the strategy layer's contract spells them."""
        return {LOW_QUANTILE: self.q_low, HIGH_QUANTILE: self.q_high}


@dataclass(frozen=True)
class ZarrFrames:
    """The competition corpus's frames: an OME-Zarr store read through the video's own intensity quantiles."""

    zarr_path: Path
    q_low: float
    q_high: float
    normalisation: Normalisation = field(default_factory=QuantileWindow)

    def frame(self, timepoint: int, downsample: tuple[int, int, int]) -> Float[Tensor, "z y x"]:
        """One quantile-normalised, strided, non-negative frame — the detector's own preprocessing recipe."""
        dz, dy, dx = downsample
        array = zarr.open_array(self.zarr_path / "0")
        raw = np.asarray(array[timepoint, ::dz, ::dy, ::dx], dtype=np.float32)
        return torch.from_numpy(self.normalisation.apply(raw, self._quantiles()))

    def _quantiles(self) -> dict[str, float]:
        """This video's shipped intensity quantiles, as the strategy layer's contract spells them."""
        return {LOW_QUANTILE: self.q_low, HIGH_QUANTILE: self.q_high}
