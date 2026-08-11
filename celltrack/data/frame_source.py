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

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import zarr
from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class FrameSource(Protocol):
    """One video's frames, already normalised — the only thing a pair needs to know about how it is stored."""

    def frame(self, timepoint: int, downsample: tuple[int, int, int]) -> Float[Tensor, "z y x"]:
        """That timepoint's normalised, non-negative volume on the grid `downsample` names."""
        ...


@dataclass(frozen=True)
class ZarrFrames:
    """The competition corpus's frames: an OME-Zarr store read through the video's own intensity quantiles."""

    zarr_path: Path
    q_low: float
    q_high: float

    def frame(self, timepoint: int, downsample: tuple[int, int, int]) -> Float[Tensor, "z y x"]:
        """One quantile-normalised, strided, non-negative frame — the detector's own preprocessing recipe."""
        dz, dy, dx = downsample
        array = zarr.open_array(self.zarr_path / "0")
        raw = np.asarray(array[timepoint, ::dz, ::dy, ::dx], dtype=np.float32)
        normed = (torch.from_numpy(raw) - self.q_low) / (self.q_high - self.q_low + 1e-6)
        return normed.clamp(0.0)
