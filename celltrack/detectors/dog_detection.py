"""A multi-scale Difference-of-Gaussians blob detector, run on the GPU.

A Laplacian-of-Gaussian at a single scale (the classical floor, `BlobDetector`) finds cells of one size;
real nuclei span a range, and the response of a scale-matched filter falls off sharply away from its
tuned radius. A Difference-of-Gaussians is the standard blob detector that fixes both: `G(sigma) -
G(k*sigma)` is a band-pass that approximates the scale-normalised LoG when `k = 1.6` (Lowe), and taking
the per-voxel maximum over a small octave-spaced band of scales gives one response that peaks at a cell of
*any* size in the band. Dim, deep-z nuclei that a single scale misses survive because a different scale in
the band answers for them.

Every Gaussian is a separable convolution, so the whole response is three 1-D convolutions per axis per
scale on the GPU — orders of magnitude faster than a CPU `gaussian_laplace`, which matters both for local
iteration on the RTX and for a competition kernel under a time cap. The per-axis sigma is a physical cell
radius in voxels (the same anisotropy-aware conversion the LoG detector uses), so an isotropic cell in
tissue is the ellipsoid it actually is on the 4x-anisotropic grid.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from torch.nn import functional

from celltrack.detectors.detection import CELL_SCALE_UM, BlobDetector
from celltrack.detectors.peaks import PeakExtractor
from core.data.tracks import TrackGraph
from core.geometry import Spacing

# The Difference-of-Gaussians ratio that approximates the scale-normalised Laplacian-of-Gaussian (Lowe,
# "Distinctive Image Features from Scale-Invariant Keypoints", 2004). Not a tunable — it is *the* value
# that makes a DoG a LoG.
_LOG_APPROXIMATING_RATIO = 1.6

# The octave-spaced band of scales around the cell radius: one octave below, at, and above, so a cell up to
# sqrt(2) larger or smaller than the nominal radius still lands on a tuned scale. A default band, not a
# constant — pass `scales_um` directly to detect a different size range.
_OCTAVE = float(np.sqrt(2.0))
_DEFAULT_SCALES_UM = (CELL_SCALE_UM / _OCTAVE, CELL_SCALE_UM, CELL_SCALE_UM * _OCTAVE)

# A Gaussian truncated at three standard deviations captures >99% of its mass; wider only adds cost.
_TRUNCATE = 3.0


@dataclass(frozen=True)
class DoGDetector:
    """Multi-scale Difference-of-Gaussians blob detection at a band of physical cell scales, on the GPU."""

    spacing: Spacing
    scales_um: tuple[float, ...] = _DEFAULT_SCALES_UM
    device: str = "cuda"

    def detect(self, frames: Iterable[Float[np.ndarray, "z y x"]], frame_count: int, keep: int) -> TrackGraph:
        """Detect the `keep` strongest cell centres across a video's frames, as an edgeless track graph."""
        return BlobDetector.detect_over_frames(self, frames, frame_count, keep)

    def response(self, frame_normalised: Float[np.ndarray, "z y x"]) -> Float[np.ndarray, "z y x"]:
        """The per-voxel maximum Difference-of-Gaussians response over the band — high at a blob of any scale in it."""
        volume = torch.from_numpy(np.ascontiguousarray(frame_normalised, dtype=np.float32))[None, None]
        volume = volume.to(self.device)
        band = torch.stack([self._difference_of_gaussians(volume, scale_um) for scale_um in self.scales_um])
        return band.amax(dim=0)[0, 0].cpu().numpy()

    def centres(self, response: Float[np.ndarray, "z y x"], keep: int) -> Int[np.ndarray, "k 3"]:
        """The `keep` strongest suppressed maxima of a response, as `(z, y, x)` voxel indices."""
        return PeakExtractor(self.spacing, self.scales_um[len(self.scales_um) // 2]).centres(response, keep)

    def _difference_of_gaussians(
        self, volume: Float[torch.Tensor, "1 1 z y x"], scale_um: float
    ) -> Float[torch.Tensor, "1 1 z y x"]:
        """`G(sigma) - G(k*sigma)` at one physical scale, scale-normalised so responses compare across the band."""
        sigma = self.spacing.anisotropic_radius(scale_um)
        inner = self._blur(volume, sigma)
        outer = self._blur(volume, sigma * _LOG_APPROXIMATING_RATIO)
        return (inner - outer) * (scale_um**2)

    def _blur(
        self, volume: Float[torch.Tensor, "1 1 z y x"], sigma_zyx: Float[np.ndarray, "3"]
    ) -> Float[torch.Tensor, "1 1 z y x"]:
        """An anisotropic 3-D Gaussian blur as three separable 1-D convolutions, one per axis, on the GPU."""
        blurred = volume
        for axis, sigma in enumerate(sigma_zyx):
            blurred = self._blur_axis(blurred, float(sigma), axis)
        return blurred

    def _blur_axis(
        self, volume: Float[torch.Tensor, "1 1 z y x"], sigma: float, axis: int
    ) -> Float[torch.Tensor, "1 1 z y x"]:
        """Convolve one spatial axis with a truncated 1-D Gaussian, reflecting at the borders.

        The truncation radius is capped to the axis length so a wide Gaussian on a thin axis (the few-slice
        z of an anisotropic crop) stays inside what a reflecting pad allows.
        """
        radius = min(max(1, int(_TRUNCATE * sigma + 0.5)), volume.shape[2 + axis] - 1)
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        padding = [0, 0, 0]
        padding[axis] = radius
        pad = (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
        padded = functional.pad(volume, pad, mode="reflect")
        return functional.conv3d(padded, kernel.reshape(shape))
