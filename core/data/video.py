"""Reading one 3D+time light-sheet video from its OME-Zarr store."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import zarr
from jaxtyping import Float, UInt16
from zarr.core.metadata import ArrayV3Metadata

from core.geometry import Spacing

_FULL_RESOLUTION = "0"


class ScaleTransformation(TypedDict):
    """An OME-NGFF coordinate transformation — the per-axis voxel size, `(t, z, y, x)`."""

    scale: list[float]


class Multiscale(TypedDict):
    """The one resolution level this dataset ships, with its transformation."""

    datasets: list[dict[str, list[ScaleTransformation]]]


class ImageStatistics(TypedDict):
    """Per-video intensity quantiles, keyed by quantile level as a string."""

    quantiles: dict[str, float]


@dataclass(frozen=True)
class IntensityQuantiles:
    """Per-video intensity quantiles shipped in the store metadata.

    These are what normalisation is derived from: the raw uint16 range is dominated by a handful of
    bright outliers, so a fixed divisor would be a magic number and a per-batch statistic would make
    training depend on batching.
    """

    levels: dict[float, float]

    def robust_range(self, low: float, high: float) -> tuple[float, float]:
        """The intensity interval between two shipped quantile levels."""
        return self.levels[low], self.levels[high]

    def normalise(self, frame: UInt16[np.ndarray, "z y x"], low: float, high: float) -> Float[np.ndarray, "z y x"]:
        """Map a frame onto roughly [0, 1] using the shipped quantiles, clipping the outlier tails."""
        floor, ceiling = self.robust_range(low, high)
        return np.clip((frame.astype(np.float32) - floor) / (ceiling - floor), 0.0, 1.0)


class CellVideo:
    """A `(T, Z, Y, X)` uint16 acquisition, chunked one timepoint per chunk."""

    def __init__(
        self,
        frames: zarr.Array[ArrayV3Metadata],
        spacing: Spacing,
        quantiles: IntensityQuantiles,
    ) -> None:
        self._frames = frames
        self.spacing = spacing
        self.quantiles = quantiles

    @classmethod
    def from_ome_zarr(cls, store: Path) -> "CellVideo":
        """Open a video, taking voxel spacing and intensity statistics from the OME metadata rather than assuming."""
        attributes = zarr.open_group(store, mode="r").attrs
        multiscale = cast(list[Multiscale], attributes["multiscales"])[0]
        scale = multiscale["datasets"][0]["coordinateTransformations"][0]["scale"]
        statistics = cast(ImageStatistics, attributes["image_statistics"])["quantiles"]
        quantiles = {float(level): float(value) for level, value in statistics.items()}
        return cls(
            frames=zarr.open_array(store / _FULL_RESOLUTION),
            spacing=Spacing(z=float(scale[1]), y=float(scale[2]), x=float(scale[3])),
            quantiles=IntensityQuantiles(levels=quantiles),
        )

    @property
    def timepoint_count(self) -> int:
        """How many timepoints the acquisition holds."""
        return int(self._frames.shape[0])

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        """The `(z, y, x)` shape of a single timepoint."""
        z, y, x = self._frames.shape[1:]
        return int(z), int(y), int(x)

    def frame(self, timepoint: int) -> UInt16[np.ndarray, "z y x"]:
        """One timepoint as a volume — a whole chunk, so this is a single decompress."""
        return np.asarray(self._frames[timepoint])
