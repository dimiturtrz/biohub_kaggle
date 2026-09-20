"""Does a contour HIERARCHY carry information the detected candidate set does not? The `g89y` precondition.

E49 closed the offline prune axis: eight features of our existing detections (position, intensity,
crowding, temporal support, and the whole affinity field) reach AUC 0.6622 jointly and FP-recall 0.0846
at the 2 % collateral E47 requires, against 0.75. bd `g89y` survives that only because Ultrack's
segment-SELECTION ILP does not score the detections we hold -- it scores nested contour hypotheses the
candidate set never contained. That is an argument about scope, and it is cheap to test.

Two numbers decide whether the GPU build is worth anyone's wall-clock:

    CONTAINMENT   what fraction of GT cells has a hierarchy segment within the cell-separation ruler.
                  Our detector reaches 0.824 at 2.87 um (E36). A hierarchy that contains cells the
                  detector misses is offering new candidates, which is the half `g89y` claims.
    SEPARABILITY  whether a hierarchy-INTRINSIC score -- contour frontier strength, segment height in
                  the ultrametric tree, area -- reaches E47's operating point. These are not functions
                  of any of E49's eight features: they describe how a segment survives across merge
                  thresholds, which a fixed detection has no way to express.

The contour map here is a raw-intensity proxy (edge = 1 - normalised frame), NOT a trained contour
head, so the asymmetry is deliberate and must be respected when reading the result: a PASS licenses the
build, while a NULL is confounded by the proxy and only says "not with this contour map".

The proxy is not even valid everywhere. A movie with a bright hazy background (raw median 1115 vs 114)
puts its GT cells BELOW the foreground cut, so containment there is zero by construction and measures
the threshold, not the hierarchy. `proxy_validity` refuses such a movie up front rather than letting it
average into the pooled number -- Otsu admits those cells but inflates foreground from 2 % to 21-48 %,
costing >3 min/frame against 5.5 s, so there is no threshold that is both valid and affordable here.

    .venv/Scripts/python kaggle/hierarchy_precondition.py --train <dir of *.geff + *.zarr> --movies 6
"""

import argparse
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import zarr
from fragment_selection_gate import OFFICIAL_UM, SPACING
from prune_score_quality import REQUIRED_RECALL
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from ultrack.core.segmentation.hierarchy import create_hierarchies

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

SEPARATION_UM = 2.87
FOREGROUND_QUANTILE = 0.98
MIN_PROXY_VALIDITY = 0.5
MIN_AREA = 40
MAX_AREA = 4000
MIN_AREA_FACTOR = 4.0
COLLATERAL_TARGETS = (0.02, 0.05, 0.10)
NAMES = ("frontier", "height", "area", "intensity")


def segments(frame):
    """Every nested contour hypothesis in one frame, with the scores the hierarchy itself assigns."""
    normalised = (frame - frame.min()) / max(frame.max() - frame.min(), 1e-6)
    foreground = normalised > np.quantile(normalised, FOREGROUND_QUANTILE)
    nodes = []
    for hierarchy in create_hierarchies(
        foreground,
        1.0 - normalised,
        min_area=MIN_AREA,
        max_area=MAX_AREA,
        min_frontier=0.0,
        min_area_factor=MIN_AREA_FACTOR,
    ):
        nodes.extend(hierarchy.nodes)
    if not nodes:
        return np.zeros((0, 3)), np.zeros((0, len(NAMES)))
    centroids = np.array([node.centroid for node in nodes], dtype=float)
    scores = np.array(
        [(node.frontier, node.height, node.area, normalised[tuple(node.centroid)]) for node in nodes],
        dtype=float,
    )
    return centroids, scores


def label_against_truth(centroids, truth, ruler_um):
    """A segment is REAL if a GT cell has no closer segment; containment is the GT's own nearest hit."""
    scale = SPACING.as_array()
    if len(centroids) == 0 or len(truth) == 0:
        return np.zeros(len(centroids), dtype=bool), np.zeros(len(truth))
    distance, nearest = cKDTree(centroids * scale).query(truth * scale, k=1)
    real = np.zeros(len(centroids), dtype=bool)
    real[nearest[distance <= ruler_um]] = True
    return real, distance


def proxy_validity(frame, truth):
    """Fraction of GT cells the proxy can see; below the foreground cut, containment is 0 by construction."""
    normalised = (frame - frame.min()) / max(frame.max() - frame.min(), 1e-6)
    voxels = np.clip(np.rint(truth).astype(int), 0, np.array(normalised.shape) - 1)
    return float((normalised[tuple(voxels.T)] > np.quantile(normalised, FOREGROUND_QUANTILE)).mean())


def collect(train_dir: Path, stems, ruler_um: float, stride: int):
    scores, labels, reach = [], [], []
    for stem in stems:
        geff, volume = train_dir / f"{stem}.geff", train_dir / f"{stem}.zarr"
        if not volume.exists():
            continue
        array = zarr.open(str(volume))["0"]
        graph = AnnotatedTracks.from_geff(geff).graph
        times, positions = graph.timepoints(), graph.positions().astype(float)
        sampled = np.unique(times)[::stride]
        validity = proxy_validity(np.asarray(array[int(sampled[0])]).astype(np.float32), positions[times == sampled[0]])
        if validity < MIN_PROXY_VALIDITY:
            log.warning(
                "  %s SKIPPED: proxy sees %.3f of GT (< %.2f) -- intensity is not a contour cue here",
                stem,
                validity,
                MIN_PROXY_VALIDITY,
            )
            continue
        for time in sampled:
            truth = positions[times == time]
            started = perf_counter()
            centroids, score = segments(np.asarray(array[int(time)]).astype(np.float32))
            real, distance = label_against_truth(centroids, truth, ruler_um)
            scores.append(score)
            labels.append(real)
            reach.append(distance)
            log.info(
                "  %s t=%d segments %d  GT %d  contained %.3f  %.1fs",
                stem,
                time,
                len(score),
                len(truth),
                (distance <= ruler_um).mean() if len(truth) else float("nan"),
                perf_counter() - started,
            )
    return np.vstack(scores), np.concatenate(labels), np.concatenate(reach)


def report(scores, labels, reach):
    log.info("segments n=%d   real %d   GT cells %d", len(labels), int(labels.sum()), len(reach))
    for ruler in (SEPARATION_UM, OFFICIAL_UM):
        log.info("CONTAINMENT: GT with a segment within %.2f um: %.4f", ruler, (reach <= ruler).mean())
    for name, column in zip(NAMES, scores.T, strict=True):
        log.info("  %-10s AUC(real > FP) %.4f", name, roc_auc_score(labels, column))
        for target in COLLATERAL_TARGETS:
            cut = np.quantile(column[labels], target)
            log.info(
                "      collateral %.2f -> FP-recall %.4f   (E47 requires %.2f at 0.02)",
                target,
                (column[~labels] < cut).mean(),
                REQUIRED_RECALL,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="directory of *.geff and matching *.zarr")
    parser.add_argument("--stems", nargs="+", required=True, help="densely annotated movie stems")
    parser.add_argument("--ruler", type=float, default=SEPARATION_UM, help="match radius in um")
    parser.add_argument("--stride", type=int, default=10, help="sample every Nth frame")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    logging.getLogger("ultrack").setLevel(logging.WARNING)
    report(*collect(args.train, args.stems, args.ruler, args.stride))


if __name__ == "__main__":
    main()
