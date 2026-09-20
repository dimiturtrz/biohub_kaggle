"""Is a GT cell the detector 'missed' actually absent, or is it its own detection placed a few microns off?

Node recall read at cell separation (2.87 um) is far below recall read at the official matcher (7 um), and
the gap has been explained as missing cells -- a GT node borrowing a NEIGHBOURING cell's detection while
every edge through it dies. That explanation and the benign one make different, measurable predictions:

    missing      nothing sits near the GT cell, or the thing that does belongs to another GT cell nearby
    displaced    the cell's OWN detection sits just past the tolerance, with no second GT cell in reach

Three readings separate them, and the third is the referee. Proximity alone proves nothing: with ~850
detections in a ~104 um cube, something is always a few microns away. So the same distance is measured
against a MATCHED NULL -- the GT cells of frame t scored against the detections of frame (t + shift), which
preserves the detection count, the spatial distribution and the density exactly and destroys only the
cell-to-detection identity. If the real distance does not beat the null, the nearby detection is a
coincidence and the 'displaced' reading is dead.

    .venv/Scripts/python kaggle/detection_localization_null.py --cache <dir of *.npz> --train <dir of *.geff>

Reported as distributions, never means: a mean over a skewed per-axis error hides the asymmetry that is
the only part of this surviving its own selection effect.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from core.data.tracks import AnnotatedTracks
from core.geometry import Spacing

log = logging.getLogger(__name__)

SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)
SEPARATION_UM = 2.87
ABSORBED_UM = 5.0
BLIND_UM = 10.0
NULL_SHIFT_FRAMES = 50
HORIZON = 100
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
VOXEL_HISTOGRAM_RANGE = 5


def _nearest(query_um, reference_um):
    if len(reference_um) == 0:
        return np.full(len(query_um), np.inf), np.zeros(len(query_um), dtype=int)
    return cKDTree(reference_um).query(query_um, k=1)


def measure(cache_dir: Path, train_dir: Path):
    """Per missed GT cell: distance to its nearest detection, to its nearest GT neighbour, and the null."""
    scale = SPACING.as_array()
    real, null, neighbour_missed, neighbour_found, offsets, offsets_all = [], [], [], [], [], []

    for path in sorted(cache_dir.glob("*.npz")):
        geff = train_dir / f"{path.stem}.geff"
        if not geff.exists():
            continue
        graph = AnnotatedTracks.from_geff(geff).graph
        times, positions = graph.timepoints(), graph.positions()
        coords = np.load(path)["coords"].astype(np.float64)

        for t in np.unique(times):
            gt_voxel = positions[times == t].astype(np.float64)
            detected = coords[coords[:, 0] == t][:, 1:]
            shuffled = coords[coords[:, 0] == (t + NULL_SHIFT_FRAMES) % HORIZON][:, 1:]
            if len(detected) == 0 or len(gt_voxel) == 0:
                continue

            distance, index = _nearest(gt_voxel * scale, detected * scale)
            missed = distance > SEPARATION_UM
            offsets_all.append(detected[index] - gt_voxel)
            offsets.append(detected[index][missed] - gt_voxel[missed])
            real.append(distance[missed])
            null.append(_nearest(gt_voxel[missed] * scale, shuffled * scale)[0])

            if len(gt_voxel) > 1:
                own, _ = cKDTree(gt_voxel * scale).query(gt_voxel * scale, k=2)
                neighbour_missed.append(own[missed, 1])
                neighbour_found.append(own[~missed, 1])
        log.info("  %s done", path.stem)

    return {
        "real": np.concatenate(real),
        "null": np.concatenate(null),
        "neighbour_missed": np.concatenate(neighbour_missed),
        "neighbour_found": np.concatenate(neighbour_found),
        "offsets": np.concatenate(offsets),
        "offsets_all": np.concatenate(offsets_all),
    }


def report(m):
    real, null = m["real"], m["null"]
    log.info("")
    log.info("missed GT cells (no detection within %.2f um)   n=%d", SEPARATION_UM, len(real))
    for label, values in (
        ("missed -> nearest detection", real),
        ("missed -> NULL detection (frame t+%d)" % NULL_SHIFT_FRAMES, null),
        ("missed -> nearest GT cell", m["neighbour_missed"]),
        ("found  -> nearest GT cell", m["neighbour_found"]),
    ):
        finite = values[np.isfinite(values)]
        spread = " ".join(f"{q:6.2f}" for q in np.quantile(finite, QUANTILES))
        log.info("  %-38s um q10/25/50/75/90 = %s", label, spread)

    log.info("")
    log.info("  detection within %.1f um of a missed cell (DISPLACED)  %.3f", ABSORBED_UM, (real <= ABSORBED_UM).mean())
    log.info("  nothing within %.1f um of it (BLIND)                   %.3f", BLIND_UM, (real > BLIND_UM).mean())
    log.info("  own-frame CLOSER than the null                        %.3f   (0.500 = chance)", (real < null).mean())

    log.info("")
    log.info("per-axis offset (detection - GT) in VOXELS, over ALL GT -- unconditioned, so no selection effect")
    for axis, name in enumerate("zyx"):
        rounded = np.round(m["offsets_all"][:, axis]).astype(int)
        bins = range(-VOXEL_HISTOGRAM_RANGE, VOXEL_HISTOGRAM_RANGE + 1)
        body = "  ".join(f"{v:+d}:{(rounded == v).mean() * 100:.1f}%" for v in bins if (rounded == v).any())
        log.info("  d%s  %s", name, body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached detector *.npz (coords)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching train *.geff")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(measure(args.cache, args.train))


if __name__ == "__main__":
    main()
