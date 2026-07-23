"""Turning sparse annotated cell centres into a supervised detection target.

The ground truth annotates ~6 % of the cells, and this is the fact the whole training scheme turns on: an
unlabelled bright blob is a real cell, not background, so it must not be scored as a negative. The target
is a per-voxel "cellness" heatmap — a Gaussian bump at every annotated centre — and a *mask* decides where
the loss is even allowed to look:

- **near an annotated centre** the answer is known (a cell is there) → supervise, positive;
- **in clearly dark tissue** far from any centre the answer is known (nothing is there) → supervise, negative;
- **everywhere else** — bright, but unlabelled — the answer is unknown (likely an un-annotated cell) → excluded.

The dark-background negatives are what stop the model predicting cells everywhere; excluding the ambiguous
bright regions is what stops it being punished for finding real cells the annotation happened to skip.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from core.geometry import Spacing

# A voxel is a safe negative only well below the intensity the dimmest annotated cells reach, so a genuine
# but unlabelled cell is never mistaken for background. Calibrated against the shipped quantile range,
# which normalisation maps to [0, 1]; the dimmest true cells sit well above this.
_BACKGROUND_MAX_INTENSITY = 0.15


@dataclass(frozen=True)
class DetectionTarget:
    """A detection heatmap and the mask that says where its loss is trustworthy."""

    heatmap: Float[np.ndarray, "z y x"]
    mask: Bool[np.ndarray, "z y x"]

    @classmethod
    def build(
        cls,
        centres: Int[np.ndarray, "n 3"],
        volume: Float[np.ndarray, "z y x"],
        spacing: Spacing,
        scale_um: float,
    ) -> "DetectionTarget":
        """A Gaussian heatmap at the annotated centres, masked into known-positive, known-negative, unknown.

        `centres` are the `(z, y, x)` voxel indices of the cells annotated in this volume; `volume` is the
        normalised intensity the negatives are read from; `scale_um` is the physical cell radius the
        Gaussian and the positive radius are derived from.
        """
        sigma = spacing.anisotropic_radius(scale_um)
        heatmap = cls._gaussian_peaks(centres, volume.shape, sigma)
        near_a_centre = heatmap > np.exp(-0.5)  # within ~1 sigma of some annotated centre
        is_dark = volume <= _BACKGROUND_MAX_INTENSITY
        mask = near_a_centre | (is_dark & ~near_a_centre)
        return cls(heatmap=heatmap, mask=mask)

    def supervised_fraction(self) -> float:
        """What share of the volume carries loss — the rest is ambiguous and excluded."""
        return float(self.mask.mean())

    @staticmethod
    def _gaussian_peaks(
        centres: Int[np.ndarray, "n 3"],
        shape: tuple[int, int, int],
        sigma: Float[np.ndarray, "3"],
    ) -> Float[np.ndarray, "z y x"]:
        """The maximum over anisotropic Gaussian bumps placed at each centre — peak 1.0, decaying per axis.

        Written as a max of per-centre bumps rather than a sum so two nearby cells do not add to more than
        one, and evaluated only inside a few-sigma box per centre so the cost scales with the cell count,
        not the volume.
        """
        heatmap = np.zeros(shape, dtype=np.float32)
        radius = np.ceil(3 * sigma).astype(int)
        for centre in centres:
            lo = np.maximum(centre - radius, 0)
            hi = np.minimum(centre + radius + 1, shape)
            grids = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))]
            squared = sum(((grid - c) / s) ** 2 for grid, c, s in zip(grids, centre, sigma, strict=True))
            window = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
            np.maximum(heatmap[window], np.exp(-0.5 * squared), out=heatmap[window])
        return heatmap
