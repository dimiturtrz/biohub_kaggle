"""Recovering cell centres from a per-voxel response, with an anisotropy-aware suppression radius.

Both detectors — the classical LoG and the learned U-Net — end the same way: a per-voxel "cellness"
volume from which centres are the suppressed local maxima. The suppression radius is a physical cell
radius expressed per axis in voxels, so on the 4x-anisotropic grid it reaches four times as far in y and x
as in z, and two peaks closer than a cell cannot both survive. Keeping this in one place is what lets the
learned detector be compared to the classical one at a matched operating point rather than through two
subtly different peak-pickers.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from skimage.feature import peak_local_max

from core.geometry import Spacing


@dataclass(frozen=True)
class PeakExtractor:
    """Non-maximum suppression over a response volume at a physical cell scale."""

    spacing: Spacing
    scale_um: float

    def centres(self, response: Float[np.ndarray, "z y x"], keep: int) -> Int[np.ndarray, "k 3"]:
        """The `keep` strongest suppressed maxima of a response, as `(z, y, x)` voxel indices."""
        radius = np.ceil(self.spacing.anisotropic_radius(self.scale_um)).astype(int)
        footprint = np.ones(2 * radius + 1, dtype=bool)
        return peak_local_max(response, footprint=footprint, num_peaks=keep).astype(np.int64)
