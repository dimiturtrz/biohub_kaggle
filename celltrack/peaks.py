"""Recovering cell centres from a per-voxel response, with an anisotropy-aware suppression radius.

Both detectors — the classical LoG and the learned U-Net — end the same way: a per-voxel "cellness"
volume from which centres are the suppressed local maxima. A voxel is a centre when it equals the maximum
over its anisotropic neighbourhood, and the strongest such maxima are kept. The neighbourhood is a
physical cell radius expressed per axis in voxels, so on the 4x-anisotropic grid it reaches four times as
far in y and x as in z, and two peaks closer than a cell cannot both survive.

Suppression is a strided-1 max pool: `response == max_pool(response)` marks every local maximum in one
pass. This is the same operation a dense maximum filter computes, but the pooling kernel is separable and
optimised, so it is orders of magnitude faster than a voxel-by-voxel filter over a several-thousand-voxel
footprint — which matters when the detector has to run inside a competition kernel under a time cap.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from torch.nn import functional

from core.geometry import Spacing


@dataclass(frozen=True)
class PeakExtractor:
    """Non-maximum suppression over a response volume at a physical cell scale."""

    spacing: Spacing
    scale_um: float

    def centres(self, response: Float[np.ndarray, "z y x"], keep: int) -> Int[np.ndarray, "k 3"]:
        """The `keep` strongest suppressed maxima of a response, as `(z, y, x)` voxel indices."""
        radius = np.ceil(self.spacing.anisotropic_radius(self.scale_um)).astype(int)
        kernel = tuple(int(2 * axis + 1) for axis in radius)
        padding = tuple(int(axis) for axis in radius)
        volume = torch.from_numpy(np.ascontiguousarray(response, dtype=np.float32))[None, None]
        pooled = functional.max_pool3d(volume, kernel_size=kernel, stride=1, padding=padding)
        # A flat region equals its own max, so exclude the response floor — otherwise the dark background,
        # all at the minimum, would count as a field of local maxima (this is what peak_local_max's default
        # `threshold_abs = image.min()` does).
        is_peak = ((volume == pooled) & (volume > volume.amin()))[0, 0]
        coordinates = torch.nonzero(is_peak)
        strongest = torch.argsort(volume[0, 0][is_peak], descending=True)[:keep]
        return coordinates[strongest].numpy().astype(np.int64)
