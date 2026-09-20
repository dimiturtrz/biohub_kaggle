"""Can ANY combination of cache-available features prune? The ceiling of the offline prune axis.

E48 killed two scores individually -- center-voxel intensity (AUC 0.5221) and best incident affinity
(AUC 0.6380) -- both an order of magnitude short of E47's operating point (FP-recall 0.75 at 2 %
collateral). Its strong reading was that real-vs-FP is not recoverable from anything DOWNSTREAM of the
detector. Two scores do not establish that. This gives the claim its best shot: every feature the cache
and volumes hold, fed jointly to a gradient-boosted model, scored on HELD-OUT MOVIES so nothing leaks.

Features are deliberately exhaustive rather than clever -- intensity, the affinity field around a
detection (best in, best out, margin to the runner-up, degree), local crowding, and temporal support
(how close the nearest detection sits in the previous and next frame). If the joint model still misses
the operating point, the offline prune axis is closed and the lever is the detector itself; if it
clears it, the winning feature names the build.

    .venv/Scripts/python kaggle/prune_feature_ceiling.py --cache <dir of *.npz> --train <dir of *.geff + *.zarr>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from fragment_selection_gate import OFFICIAL_UM, matched_detections
from prune_score_quality import AFFINITY_COLUMN, REQUIRED_RECALL, brightness
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

SUBSTRATE = np.array([1.0, 4.0, 4.0])
UM_PER_VOXEL = 1.625
CROWD_RADIUS_UM = 5.0
COLLATERAL_TARGETS = (0.02, 0.05, 0.10)
NAMES = ("intensity", "best_out", "best_in", "margin", "degree", "crowding", "support_prev", "support_next")


def affinity_field(edges, count):
    """Per detection: strongest outgoing, strongest incoming, margin to the runner-up, and degree."""
    source, target, affinity = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, AFFINITY_COLUMN]
    best_out, best_in, runner_up, degree = (np.zeros(count) for _ in range(4))
    np.maximum.at(best_out, source, affinity)
    np.maximum.at(best_in, target, affinity)
    np.add.at(degree, source, 1)
    np.add.at(degree, target, 1)
    for a, score in zip(source, affinity, strict=True):
        if score < best_out[a]:
            runner_up[a] = max(runner_up[a], score)
    return best_out, best_in, best_out - runner_up, degree


def neighbourhood(coords):
    """Per detection: how crowded its own frame is, and how close the nearest cell sits in t-1 and t+1."""
    positions = coords[:, 1:] / SUBSTRATE * UM_PER_VOXEL
    times = coords[:, 0]
    trees = {int(t): cKDTree(positions[times == t]) for t in np.unique(times)}
    rows = {int(t): np.where(times == t)[0] for t in np.unique(times)}

    crowding, support = np.zeros(len(coords)), np.full((len(coords), 2), np.inf)
    for time, here in rows.items():
        crowding[here] = trees[time].query_ball_point(positions[here], CROWD_RADIUS_UM, return_length=True) - 1
        for column, step in enumerate((-1, 1)):
            neighbour = trees.get(time + step)
            if neighbour is not None:
                support[here, column] = neighbour.query(positions[here])[0]
    return crowding, np.nan_to_num(support[:, 0], posinf=0.0), np.nan_to_num(support[:, 1], posinf=0.0)


def collect(cache_dir: Path, train_dir: Path, ruler_um: float):
    """Feature rows, labels, and the movie each row came from (the split axis)."""
    rows, labels, movies = [], [], []
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

        features = (brightness(volume, coords), *affinity_field(cached["edges"], len(coords)), *neighbourhood(coords))
        rows.append(np.column_stack(features))
        labels.append(real)
        movies.append(np.full(len(coords), len(movies)))
    return np.vstack(rows), np.concatenate(labels), np.concatenate(movies)


def operating_points(score, real):
    """FP-recall at each required collateral, the cut placed on the real detections' own quantile."""
    return [((score[~real] < np.quantile(score[real], target)).mean(), target) for target in COLLATERAL_TARGETS]


def report(rows, labels, movies, seed):
    held_out = movies % 2 == 1
    model = HistGradientBoostingClassifier(random_state=seed).fit(rows[~held_out], labels[~held_out])
    score = model.predict_proba(rows[held_out])[:, 1]
    real = labels[held_out]

    log.info("train %d rows / test %d rows on held-out movies   real %d", (~held_out).sum(), len(real), real.sum())
    log.info("JOINT model: AUC(real > FP) %.4f", roc_auc_score(real, score))
    for recall, target in operating_points(score, real):
        log.info("    collateral %.2f -> FP-recall %.4f   (E47 requires %.2f at 0.02)", target, recall, REQUIRED_RECALL)
    for name, column in zip(NAMES, rows[held_out].T, strict=True):
        log.info("  alone  %-13s AUC %.4f", name, roc_auc_score(real, column))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached candidate graphs (*.npz)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching *.geff and *.zarr")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    parser.add_argument("--seed", type=int, default=0, help="seed for the gradient-boosted model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(*collect(args.cache, args.train, args.ruler), args.seed)


if __name__ == "__main__":
    main()
