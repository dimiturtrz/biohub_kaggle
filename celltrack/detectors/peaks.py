"""Recovering cell centres from a per-voxel response, with an anisotropy-aware suppression radius.

Both detectors — the classical LoG and the learned U-Net — end the same way: a per-voxel "cellness"
volume from which centres are the suppressed local maxima. A voxel is a centre when it equals the maximum
over its anisotropic neighbourhood. Two readouts share that suppression: the strongest `keep` maxima (a
fixed node budget), or every maximum above a probability threshold (an emergent count — the high-precision
mode a classification detector is read out with). The neighbourhood is a physical cell radius expressed per
axis in voxels, so on the 4x-anisotropic grid it reaches four times as far in y and x as in z, and two
peaks closer than a cell cannot both survive.

Suppression is a strided-1 max pool on the extractor's device: `response == max_pool(response)` marks every
local maximum in one GPU pass. A flat region equals its own max everywhere, though, so that mask alone turns
a saturated blob into thousands of "maxima"; the mask is then reduced on the host by connected-component
labelling to one centre — the strongest voxel — per plateau. The heavy pooling stays on the RTX; only the
small boolean mask crosses to the CPU for the labelling, which holds under a competition kernel's time cap.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from scipy import ndimage
from torch import Tensor
from torch.nn import functional

from core.geometry import Spacing

# Full 26-connectivity: a plateau is one blob however its voxels touch, so face-, edge-, and corner-adjacent
# maxima all collapse into a single component rather than splitting into several centres.
_PLATEAU_CONNECTIVITY = np.ones((3, 3, 3), dtype=np.int64)

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

    def maxima(self, response: _Response, floor: float) -> tuple[Int[np.ndarray, "k 3"], Float[np.ndarray, "k"]]:
        """Every suppressed maximum above `floor` with its value, so a threshold sweep filters instead of re-pooling.

        The max-pool NMS is the expensive step and is threshold-independent — only the floor cut varies. Extract
        once at the lowest floor of interest, then any higher threshold is a value mask over this small array.
        """
        coordinates, values = self._local_maxima(self._volume(response), floor=floor)
        return coordinates.cpu().numpy().astype(np.int64), values.cpu().numpy()

    def _volume(self, response: _Response) -> Float[Tensor, "1 1 z y x"]:
        """The response as a batched float tensor on the extractor's device — a no-op if it is already there."""
        return torch.as_tensor(response, dtype=torch.float32, device=self.device)[None, None]

    def _local_maxima(
        self, volume: Float[Tensor, "1 1 z y x"], floor: float
    ) -> tuple[Int[Tensor, "k 3"], Float[Tensor, "k"]]:
        """One centre per connected plateau of maxima above `floor` — the max-value voxel of each component.

        `volume == max_pool(volume)` marks every voxel of a *flat* region as a maximum, so a saturated detector
        (large plateaus at probability one) would emit tens of thousands of "peaks" for a single blob, exploding
        the linker's pairwise cost matrix. Collapsing each connected component of maxima to its single
        strongest voxel recovers one centre per cell whatever the response looks like.
        """
        # Suppression radius = half the rounded cell *diameter* (2 * scale_um) in voxels — the reference
        # read-out (predict_unet_transformer.py: `k = round(diameter / spacing); pad = k // 2`, an effective
        # radius of `k // 2`). Rounding the diameter first then halving, rather than ceiling a half-diameter
        # radius, keeps the window from inflating a whole voxel per axis: at 1.625um isotropic a 5um cell gives
        # radius 1 (a 3^3 kernel), where `ceil(radius)` gave radius 2 (5^3) and merged cells two voxels apart.
        # Doubling then halving keeps the kernel odd, so stride-1 symmetric padding is size-preserving (an even
        # kernel would grow the pooled volume by one voxel). Integer spacing ratios are unchanged by the round.
        radius = np.rint(2.0 * self.spacing.anisotropic_radius(self.scale_um)).astype(int) // 2
        kernel = tuple(int(2 * axis + 1) for axis in radius)
        padding = tuple(int(axis) for axis in radius)
        pooled = functional.max_pool3d(volume, kernel_size=kernel, stride=1, padding=padding)
        is_peak = ((volume == pooled) & (volume > floor))[0, 0]
        values = volume[0, 0].detach().cpu().numpy()
        labels, count = ndimage.label(is_peak.cpu().numpy(), structure=_PLATEAU_CONNECTIVITY)
        if count == 0:
            return torch.empty((0, 3), dtype=torch.long), torch.empty((0,))
        positions = np.asarray(
            ndimage.maximum_position(values, labels, np.arange(1, count + 1)), dtype=np.int64
        ).reshape(-1, 3)
        peaks = torch.from_numpy(positions)
        return peaks, torch.from_numpy(values[positions[:, 0], positions[:, 1], positions[:, 2]])
