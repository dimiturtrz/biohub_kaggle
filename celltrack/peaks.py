"""Recovering cell centres from a per-voxel response, with an anisotropy-aware suppression radius.

Both detectors — the classical LoG and the learned U-Net — end the same way: a per-voxel "cellness"
volume from which centres are the suppressed local maxima. A voxel is a centre when it equals the maximum
over its anisotropic neighbourhood. Two readouts share that suppression: the strongest `keep` maxima (a
fixed node budget), or every maximum above a probability threshold (an emergent count — the high-precision
mode a classification detector is read out with). The neighbourhood is a physical cell radius expressed per
axis in voxels, so on the 4x-anisotropic grid it reaches four times as far in y and x as in z, and two
peaks closer than a cell cannot both survive.

Suppression is a strided-1 max pool: `response == max_pool(response)` marks every local maximum in one
pass, and it runs on the extractor's device — handed a response already on the GPU (the learned detector's
logits) it never leaves, so the whole read-out is GPU-resident with only the final short coordinate list
copied back. That both keeps the RTX busy on large volumes and holds under a competition kernel's time cap.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

from core.geometry import Spacing

_Response = Float[np.ndarray, "z y x"] | Float[Tensor, "z y x"]


@dataclass(frozen=True)
class PeakExtractor:
    """Non-maximum suppression over a response volume at a physical cell scale, on a chosen device."""

    spacing: Spacing
    scale_um: float
    device: str = "cpu"

    def centres(self, response: _Response, keep: int) -> Int[np.ndarray, "k 3"]:
        """The `keep` strongest suppressed maxima of a response, as `(z, y, x)` voxel indices."""
        volume = self._volume(response)
        # A flat region equals its own max, so exclude the response floor — otherwise the dark background,
        # all at the minimum, would count as a field of local maxima (peak_local_max's default threshold).
        coordinates, values = self._local_maxima(volume, floor=float(volume.amin()))
        strongest = torch.argsort(values, descending=True)[:keep]
        return coordinates[strongest].cpu().numpy().astype(np.int64)

    def above_threshold(self, response: _Response, threshold: float) -> Int[np.ndarray, "k 3"]:
        """Every suppressed maximum above `threshold`, as `(z, y, x)` voxel indices — the count is emergent."""
        coordinates, _ = self._local_maxima(self._volume(response), floor=threshold)
        return coordinates.cpu().numpy().astype(np.int64)

    def _volume(self, response: _Response) -> Float[Tensor, "1 1 z y x"]:
        """The response as a batched float tensor on the extractor's device — a no-op if it is already there."""
        return torch.as_tensor(response, dtype=torch.float32, device=self.device)[None, None]

    def _local_maxima(
        self, volume: Float[Tensor, "1 1 z y x"], floor: float
    ) -> tuple[Int[Tensor, "k 3"], Float[Tensor, "k"]]:
        """Coordinates and values of every voxel equal to its anisotropic-neighbourhood max and above `floor`."""
        radius = np.ceil(self.spacing.anisotropic_radius(self.scale_um)).astype(int)
        kernel = tuple(int(2 * axis + 1) for axis in radius)
        padding = tuple(int(axis) for axis in radius)
        pooled = functional.max_pool3d(volume, kernel_size=kernel, stride=1, padding=padding)
        is_peak = ((volume == pooled) & (volume > floor))[0, 0]
        coordinates = torch.nonzero(is_peak)
        return coordinates, volume[0, 0][is_peak]
