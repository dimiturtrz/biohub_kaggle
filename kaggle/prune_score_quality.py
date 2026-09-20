"""Can any score we ALREADY have rank a detection as real-vs-FP well enough to prune on?

E47 set the spec: a pruner must reach FP-recall ~0.75 at <=2 % collateral before the axis pays, and
argued that a threshold fails because its decision variable is intensity-ordered. That was an argument.
This measures it, for the two rankings the cache and the volumes already provide:

    INTENSITY   the raw center-voxel value at the detection, frame-normalised -- what any detector
                threshold is ultimately ordered by.
    AFFINITY    the best incident edge affinity -- the model-derived score, and the obvious proposal
                ("delete detections nothing wants to link to").

Reported three ways: separability (AUC of real over FP), the trajectory across the E47 grid, and the
number that decides it -- FP-recall AT the required collateral, where the cut is placed on the real
detections' own quantile so collateral is exact by construction.

    .venv/Scripts/python kaggle/prune_score_quality.py --cache <dir of *.npz> --train <dir of *.geff + *.zarr>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import zarr
from fragment_selection_gate import OFFICIAL_UM, matched_detections

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

AFFINITY_COLUMN = 2
COLLATERAL_TARGETS = (0.02, 0.05, 0.10)
REQUIRED_RECALL = 0.75
AUC_SAMPLE = 20000


def brightness(volume: Path, coords):
    """Frame-normalised center-voxel intensity at every detection."""
    array = zarr.open(str(volume))["0"]
    values = np.zeros(len(coords))
    for time in np.unique(coords[:, 0]):
        rows = np.where(coords[:, 0] == time)[0]
        frame = np.asarray(array[int(time)]).astype(np.float64)
        here = coords[rows]
        values[rows] = frame[here[:, 1], here[:, 2], here[:, 3]] / max(frame.max(), 1)
    return values


def best_affinity(edges, count):
    """Per detection, the strongest affinity on any edge touching it."""
    score = np.zeros(count)
    affinity = edges[:, AFFINITY_COLUMN]
    np.maximum.at(score, edges[:, 0].astype(int), affinity)
    np.maximum.at(score, edges[:, 1].astype(int), affinity)
    return score


def collect(cache_dir: Path, train_dir: Path, ruler_um: float):
    intensity, affinity, is_real = [], [], []
    for path in sorted(cache_dir.glob("*.npz")):
        geff, volume = train_dir / f"{path.stem}.geff", train_dir / f"{path.stem}.zarr"
        if not geff.exists() or not volume.exists():
            continue
        cached = np.load(path)
        coords = cached["coords"].astype(int)
        graph = AnnotatedTracks.from_geff(geff).graph
        matched = matched_detections(graph, coords.astype(np.float64), ruler_um)
        real = np.zeros(len(coords), dtype=bool)
        real[[index for index in matched if index >= 0]] = True

        intensity.append(brightness(volume, coords))
        affinity.append(best_affinity(cached["edges"], len(coords)))
        is_real.append(real)
    return np.concatenate(intensity), np.concatenate(affinity), np.concatenate(is_real)


def separability(score, real, rng):
    """P(a real detection outranks a random FP) -- 0.5 is a score that knows nothing."""
    drawn = rng.integers(0, int((~real).sum()), AUC_SAMPLE)
    return (score[real][:, None] > score[~real][drawn]).mean()


def report(intensity, affinity, real, seed):
    rng = np.random.default_rng(seed)
    log.info("detections n=%d   real %d   FP %d", len(real), int(real.sum()), int((~real).sum()))
    for name, score in (("center-voxel intensity", intensity), ("best incident affinity", affinity)):
        log.info("%s: AUC(real > FP) %.4f", name, separability(score, real, rng))
        for target in COLLATERAL_TARGETS:
            cut = np.quantile(score[real], target)
            log.info(
                "    collateral %.2f -> FP-recall %.4f   (E47 requires %.2f at 0.02)",
                target,
                (score[~real] < cut).mean(),
                REQUIRED_RECALL,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached candidate graphs (*.npz)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching *.geff and *.zarr")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    parser.add_argument("--seed", type=int, default=0, help="seed for the AUC sample")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(*collect(args.cache, args.train, args.ruler), args.seed)


if __name__ == "__main__":
    main()
