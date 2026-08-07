"""A classical blob detector — the floor of the detection family.

Cells are bright roughly-spherical blobs on a dark background, so a Laplacian-of-Gaussian tuned to the
cell scale peaks at their centres. The scale is a physical length (a cell radius in micrometres, measured
from the LoG response at annotated centres — see tools/derive_scale.py), converted to a per-axis voxel
sigma because the imaging is 4x anisotropic: an isotropic blob in tissue is an ellipsoid in index space.

Centres come from non-maximum suppression over the response, with a suppression radius set from the same
physical scale so two peaks closer than a cell cannot both survive. How many peaks to keep is the
detector's operating point, and the metric ties it to the estimated true cell count — see the sweep task.
This is deliberately the simplest detector that respects the physics; the gap between what it reaches and
what a learned detector reaches is the budget the model has to win.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jaxtyping import Float, Int
from scipy.ndimage import gaussian_laplace

from celltrack.detectors.peaks import PeakExtractor
from core.data.tracks import TrackGraph
from core.geometry import Spacing

# The quantile levels the frame is clipped to before detection — the shipped robust range, tails removed.
LOW, HIGH = 0.001, 0.999

# The cell scale, measured by tools/derive_scale.py: the LoG sigma with peak response at annotated centres
# is 4.0 um for 42 % of them (median 4.0) across four videos. It is a default, not a constant — pass a
# different scale to detect cells of a different size.
CELL_SCALE_UM = 4.0


class _FrameResponder(Protocol):
    """A response-and-suppress blob detector — the two steps `detect_over_frames` drives per frame."""

    def response(self, frame_normalised: Float[np.ndarray, "z y x"]) -> Float[np.ndarray, "z y x"]:
        """The per-voxel cellness response for one normalised frame."""
        ...

    def centres(self, response: Float[np.ndarray, "z y x"], keep: int) -> Int[np.ndarray, "k 3"]:
        """The `keep` strongest suppressed maxima of a response, as `(z, y, x)` voxel indices."""
        ...


def detect_over_frames(
    detector: _FrameResponder, frames: Iterable[Float[np.ndarray, "z y x"]], frame_count: int, keep: int
) -> TrackGraph:
    """Detect the `keep` strongest cell centres across a video's frames, as an edgeless track graph.

    `keep` is spread evenly over the timepoints — the annotation gives a per-video budget, not a per-frame
    one, and cells are present throughout, so an even split is the assumption-free default. Shared by every
    response-and-suppress detector (LoG, DoG); the network detector caches its response instead.
    """
    per_frame = max(1, round(keep / frame_count))
    coordinates = [
        np.column_stack([np.full(len(centres), timepoint), centres])
        for timepoint, frame in enumerate(frames)
        for centres in [detector.centres(detector.response(frame), per_frame)]
    ]
    stacked = np.concatenate(coordinates).astype(np.int64) if coordinates else np.empty((0, 4), np.int64)
    return TrackGraph(
        node_ids=np.arange(len(stacked), dtype=np.int64),
        coordinates=stacked,
        edges=np.empty((0, 2), dtype=np.int64),
    )


@dataclass(frozen=True)
class BlobDetector:
    """Single-scale Laplacian-of-Gaussian blob detection at a physical cell scale."""

    spacing: Spacing
    scale_um: float = CELL_SCALE_UM

    def detect(self, frames: Iterable[Float[np.ndarray, "z y x"]], frame_count: int, keep: int) -> TrackGraph:
        """Detect the `keep` strongest cell centres across a video's frames, as an edgeless track graph."""
        return detect_over_frames(self, frames, frame_count, keep)

    def response(self, frame_normalised: Float[np.ndarray, "z y x"]) -> Float[np.ndarray, "z y x"]:
        """The scale-normalised LoG response — high at a bright blob of about the cell scale.

        The Laplacian is negative at a bright peak, so it is negated; scaling by the physical scale
        squared makes responses comparable to a multi-scale search even though only one scale is used.
        """
        sigma = self.spacing.anisotropic_radius(self.scale_um)
        return -gaussian_laplace(frame_normalised, sigma=sigma) * (self.scale_um**2)

    def centres(self, response: Float[np.ndarray, "z y x"], keep: int) -> Int[np.ndarray, "k 3"]:
        """The `keep` strongest suppressed maxima of a response, as `(z, y, x)` voxel indices."""
        return PeakExtractor(self.spacing, self.scale_um).centres(response, keep)
