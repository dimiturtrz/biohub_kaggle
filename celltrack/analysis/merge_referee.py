"""GT-free blob-volume merge referee.

The (1,4,4) decode fuses two crowded cells into one over-large heatmap basin; the (1,2,2) finer
decode is claimed to separate them. GT can't referee this — the annotation enforces >4um inter-cell
spacing, so the merge regime holds *zero* labelled pairs. The physics referee instead:

- a single zebrafish cell occupies a roughly constant volume V0;
- each detected centre sits in a connected basin of voxels above a probability floor;
- a (1,4,4)-merged two-cell detection is ONE basin of ~2*V0.

So: threshold the sigmoid prob volume, label connected components, and read the basin-volume
distribution. If the finer decode un-merges, its component COUNT rises and the >1.5*V0 mass fraction
falls, at matched physical voxel size and matched weights. No GT, no card -- pure numpy over the
cached logit volumes.

`MergeReferee.analyse` is the pure compute (loaded arrays in, `GridStats` out); `main` does the disk I/O.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from scipy import ndimage

logger = logging.getLogger(__name__)

_CACHE = Path("D:/data/volumetric/microscopy/processed/biohub_cell_tracking/cache/responses")
# full-res pixels spanned by one decoded voxel in y,x for each downsample grid (z is 1:1 both).
_YX_PIXELS = {"ds1x4x4": 4, "ds1x2x2": 2}
_CONNECTIVITY = ndimage.generate_binary_structure(3, 1)  # 6-connected, no diagonal bleed
_MERGE_FACTOR = 1.5  # a basin above this multiple of V0 holds more than one cell
_FRAGMENT_FACTOR = 0.5  # a basin below this multiple of V0 is noise, not a cell
_COARSE_GRID = "ds1x4x4"  # V0 reference: isolated cells are cleanest single blobs here
_INFLATION_TOL_PCT = 15  # matched-grid peak gap this small = grids agree; larger = finer field confounded


@dataclass(frozen=True)
class Band:
    """A volume band: how many basins fall in it and what fraction of total mass they carry."""

    n: int
    mass_frac: float


@dataclass(frozen=True)
class GridStats:
    """One decode grid's basin/peak readout, all fields shared against the coarse grid's V0."""

    n_components: int
    median_vol: float
    voxel_pixels_yx: int
    peaks_native: int
    peaks_matched_1_6um: int
    merged_mass_fraction: float
    v0_used: float
    fragment: Band  # < 0.5 V0: noise
    single: Band  # 0.5-1.5 V0: one real cell
    merged: Band  # > 1.5 V0: more than one cell


class MergeReferee:
    """Compare two decode grids' basin/peak statistics to decide if the finer grid un-merges cells."""

    @staticmethod
    def _prob(logit: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
        return 1.0 / (1.0 + np.exp(-logit))

    @staticmethod
    def _components(frame_prob: Float[np.ndarray, "z y x"], floor: float, voxel_vol: float) -> Float[np.ndarray, "n"]:
        """Physical volumes of every connected basin above `floor`, one entry per component."""
        labels, count = ndimage.label(frame_prob > floor, structure=_CONNECTIVITY)
        if count == 0:
            return np.empty(0, dtype=np.float64)
        sizes = np.bincount(labels.ravel())[1:]  # drop background label 0
        return sizes.astype(np.float64) * voxel_vol

    @staticmethod
    def _peak_count(frame_prob: Float[np.ndarray, "z y x"], floor: float, min_sep_yx: int) -> int:
        """Local maxima above `floor` after NMS at a PHYSICAL yx separation -- the detector's own count.

        Replicates PeakExtractor: a voxel is a peak iff it equals the max over a footprint and clears
        the floor. Footprint yx scales with the grid so `min_sep_yx` is a fixed physical distance across
        decodes -- that is what makes the peak COUNT (unlike basin volume) immune to the finer field.
        """
        footprint = np.ones((3, 2 * min_sep_yx + 1, 2 * min_sep_yx + 1), dtype=bool)
        pooled = ndimage.maximum_filter(frame_prob, footprint=footprint, mode="nearest")
        is_peak = (frame_prob == pooled) & (frame_prob > floor)
        # collapse plateaus (equal-valued adjacent peak voxels) to one centre each
        _, count = ndimage.label(is_peak, structure=_CONNECTIVITY)
        return int(count)

    @classmethod
    def _grid_stats(
        cls, prob: Float[np.ndarray, "t z y x"], yx: int, floor: float
    ) -> tuple[Float[np.ndarray, "n"], int, int]:
        """Per-grid basin volumes plus peak counts at native and coarse-matched (1.6um) NMS."""
        voxel_vol = float(yx * yx)  # z factor is 1 and cancels in every ratio downstream
        per_frame = [cls._components(prob[t], floor, voxel_vol) for t in range(prob.shape[0])]
        vols = np.concatenate(per_frame) if per_frame else np.empty(0)
        native_sep, matched_sep = 1, max(1, 4 // yx)
        peaks_native = sum(cls._peak_count(prob[t], floor, native_sep) for t in range(prob.shape[0]))
        peaks_matched = sum(cls._peak_count(prob[t], floor, matched_sep) for t in range(prob.shape[0]))
        return vols, peaks_native, peaks_matched

    @classmethod
    def analyse(cls, logit_by_grid: dict[str, np.ndarray], floor: float) -> dict[str, GridStats]:
        """Basin/peak/band statistics for each decode grid, given its loaded logit volume.

        Pure compute: no disk, no card. `logit_by_grid` maps a grid key in `_YX_PIXELS` to its
        (T, Z, Y, X) logit volume. V0 (single-cell volume) is taken from the coarse grid's median
        basin and shared, so the merged-mass fraction is a fair cross-grid comparison.
        """
        vols_by_grid: dict[str, np.ndarray] = {}
        peaks_by_grid: dict[str, tuple[int, int]] = {}
        for grid, logit in logit_by_grid.items():
            vols, peaks_native, peaks_matched = cls._grid_stats(cls._prob(logit), _YX_PIXELS[grid], floor)
            vols_by_grid[grid] = vols
            peaks_by_grid[grid] = (peaks_native, peaks_matched)

        coarse_vols = vols_by_grid[_COARSE_GRID]
        v0 = float(np.median(coarse_vols)) if coarse_vols.size else float("nan")

        out: dict[str, GridStats] = {}
        for grid, vols in vols_by_grid.items():
            total = float(vols.sum()) or 1.0
            peaks_native, peaks_matched = peaks_by_grid[grid]
            # band-resolve count AND mass: fragments (<0.5 V0) are noise, single (0.5-1.5 V0) are real
            # cells, merged (>1.5 V0) hold more than one. Un-merge = gain in the single band, at conserved
            # mass; fragmentation = gain in the fragment band, carrying negligible mass.
            frag_mask = vols < _FRAGMENT_FACTOR * v0
            single_mask = (vols >= _FRAGMENT_FACTOR * v0) & (vols <= _MERGE_FACTOR * v0)
            merged_mask = vols > _MERGE_FACTOR * v0
            merged_mass = float(vols[merged_mask].sum())
            out[grid] = GridStats(
                n_components=int(vols.size),
                median_vol=float(np.median(vols)) if vols.size else float("nan"),
                voxel_pixels_yx=_YX_PIXELS[grid],
                peaks_native=peaks_native,
                peaks_matched_1_6um=peaks_matched,
                merged_mass_fraction=merged_mass / float(vols.sum()) if vols.sum() else float("nan"),
                v0_used=v0,
                fragment=Band(int(frag_mask.sum()), float(vols[frag_mask].sum()) / total),
                single=Band(int(single_mask.sum()), float(vols[single_mask].sum()) / total),
                merged=Band(int(merged_mask.sum()), float(vols[merged_mask].sum()) / total),
            )
        return out


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-key", default="pilkwang_seed1_logits")
    parser.add_argument("--video-key", default="44b6_e57ff5c6.zarr")
    parser.add_argument("--floor", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logit_by_grid = {
        grid: np.load(_CACHE / args.weights_key / f"{args.video_key}.logit.{grid}_tta1_pc0.npy") for grid in _YX_PIXELS
    }
    result = MergeReferee.analyse(logit_by_grid, args.floor)
    coarse, fine = result["ds1x4x4"], result["ds1x2x2"]
    logger.info("weights=%s video=%s floor=%s", args.weights_key, args.video_key, args.floor)
    logger.info(
        "  (1,4,4): %6d basins | median V %.1f | merged-mass frac %.4f",
        coarse.n_components,
        coarse.median_vol,
        coarse.merged_mass_fraction,
    )
    logger.info(
        "  (1,2,2): %6d basins | median V %.1f | merged-mass frac %.4f",
        fine.n_components,
        fine.median_vol,
        fine.merged_mass_fraction,
    )
    d_count = fine.n_components - coarse.n_components
    d_merge = fine.merged_mass_fraction - coarse.merged_mass_fraction
    logger.info(
        "  DELTA: basins %+d (%+.1f%%) | merged-frac %+.4f",
        d_count,
        100 * d_count / max(coarse.n_components, 1),
        d_merge,
    )
    for stats, tag in [(coarse, "(1,4,4)"), (fine, "(1,2,2)")]:
        logger.info(
            "  %s bands: frag n=%6d m=%.3f | single n=%6d m=%.3f | merged n=%6d m=%.3f",
            tag,
            stats.fragment.n,
            stats.fragment.mass_frac,
            stats.single.n,
            stats.single.mass_frac,
            stats.merged.n,
            stats.merged.mass_frac,
        )
    logger.info(
        "  PEAKS  (1,4,4): native(1.6um)=%6d  matched(1.6um)=%6d", coarse.peaks_native, coarse.peaks_matched_1_6um
    )
    logger.info("  PEAKS  (1,2,2): native(0.8um)=%6d  matched(1.6um)=%6d", fine.peaks_native, fine.peaks_matched_1_6um)
    # noise control: at matched 1.6um the two grids should agree; if fine-matched >> coarse-native the
    # finer field is inflating peaks. the un-merge count is fine-native minus fine-matched (cells the
    # finer grid resolves ONLY below 1.6um), and it is trustworthy only if the matched pair agrees.
    noise_inflation = fine.peaks_matched_1_6um - coarse.peaks_native
    unmerge_gain = fine.peaks_native - fine.peaks_matched_1_6um
    infl_pct = 100 * noise_inflation / max(coarse.peaks_native, 1)
    logger.info(
        "  noise control: matched-grid peak gap %+d (%+.1f%%)  | sub-1.6um un-merge peaks %+d",
        noise_inflation,
        infl_pct,
        unmerge_gain,
    )
    if abs(infl_pct) < _INFLATION_TOL_PCT and unmerge_gain > 0:
        verdict = f"UN-MERGES: +{unmerge_gain} real sub-1.6um cells, matched grids agree ({infl_pct:+.1f}%)"
    elif infl_pct >= _INFLATION_TOL_PCT:
        verdict = f"CONFOUNDED: finer field inflates peaks {infl_pct:+.1f}% even at matched resolution"
    else:
        verdict = "NO un-merge signal"
    logger.info("  VERDICT: %s", verdict)


if __name__ == "__main__":
    main()
