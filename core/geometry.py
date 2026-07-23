"""Voxel/physical-space conversion.

The imaging is 4x anisotropic (1.625 um in z against 0.40625 um in y and x), while every stored
coordinate is a voxel index and every scoring distance is in micrometres. Mixing the two spaces is the
easiest way to silently mis-score, so the conversion lives in one place.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int


@dataclass(frozen=True)
class Spacing:
    """Physical size of one voxel, in micrometres, along z, y and x."""

    z: float
    y: float
    x: float

    def as_array(self) -> Float[np.ndarray, "3"]:
        """The spacing as a (z, y, x) vector, for broadcasting against coordinate arrays."""
        return np.array([self.z, self.y, self.x], dtype=np.float64)

    def to_micrometres(self, voxels: Int[np.ndarray, "n 3"]) -> Float[np.ndarray, "n 3"]:
        """Scale voxel-index coordinates into the micrometre space the metric measures in."""
        return voxels.astype(np.float64) * self.as_array()

    def to_voxels(self, micrometres: Float[np.ndarray, "n 3"]) -> Float[np.ndarray, "n 3"]:
        """Scale micrometre coordinates back into voxel indices."""
        return micrometres / self.as_array()

    def anisotropic_radius(self, micrometres: float) -> Float[np.ndarray, "3"]:
        """A physical radius expressed per axis in voxels — an isotropic sphere is an ellipsoid in index space."""
        return micrometres / self.as_array()
