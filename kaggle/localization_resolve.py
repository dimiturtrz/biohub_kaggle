"""If the detector placed its nodes correctly, how much of the score would the LINKER then win back?

E57 measured that a broken GT edge's endpoints sit a median 4.08 um from the cell against 1.46 um for a
recovered one, and E54b's fourth row showed that distance to the target's TRUE position ranks the true
successor first 76/76. Both are rankings, not scores, and E57 flagged why a score cannot be read off them:
snapping a node onto its ground-truth coordinate is a no-op for the metric by construction, because the
broken/recovered verdict depends only on the node matching, which the snap improves directly, and the
solver has already run. Causality needs a RE-SOLVE.

So this re-solves. The arms differ in one thing: the coordinates the LINKER is allowed to see. Every arm
is scored on the ORIGINAL predicted positions, so the node matching, the node-count term and the estimated
cell count are byte-identical across arms and the whole delta lands on edges. `--shrink 1.0` hands the
linker perfect coordinates for every node the scorer can match; 0.5 hands it half the error, which is the
dose-response question a centre-offset head actually has to answer -- a head that halves the error is a
realistic target, a head that eliminates it is not.

The linker is a plain per-frame Hungarian on distance, the same in every arm. It is deliberately weaker
than the champion's ILP: the number to read is the DELTA between arms, never the level. The champion's own
edges are scored alongside as the reference level.

Only TRAIN crops have ground truth, so this runs nowhere else.

    .venv/Scripts/python kaggle/localization_resolve.py --submission <submission.csv> --train <geff dir>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from error_budget import match_bipartite
from fragment_selection_gate import OFFICIAL_UM, SPACING
from scipy.optimize import linear_sum_assignment
from short_component_prune import read_submission

from core.data.tracks import AnnotatedTracks, TrackGraph
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics

log = logging.getLogger(__name__)

LINK_GATE_UM = 14.0
"""The champion's own `OUTPUT_EDGE_MAX_UM` -- see E57, where using our 10.0 invented fifteen unreachable edges."""

SHRINK_ARMS = (0.0, 0.5, 1.0)
"""The fraction of each node's localization error handed to the linker: none, half, all."""

UNREACHABLE = 1.0e9


def link_by_distance(positions_um, timepoints, gate_um):
    """One-to-one nearest-partner links between consecutive frames, as row-index pairs."""
    links = []
    frames = np.unique(timepoints)
    rows_at = {int(frame): np.flatnonzero(timepoints == frame) for frame in frames}
    for frame in frames.tolist():
        source_rows, target_rows = rows_at.get(int(frame)), rows_at.get(int(frame) + 1)
        if target_rows is None or len(source_rows) == 0 or len(target_rows) == 0:
            continue
        cost = np.linalg.norm(positions_um[source_rows][:, None, :] - positions_um[target_rows][None, :, :], axis=-1)
        gated = np.where(cost <= gate_um, cost, UNREACHABLE)
        for row, column in zip(*linear_sum_assignment(gated), strict=True):
            if gated[row, column] < UNREACHABLE:
                links.append((int(source_rows[row]), int(target_rows[column])))
    return np.array(links, dtype=np.int64).reshape(-1, 2)


def shrunk_positions(predicted, truth, pred_to_gt, shrink):
    """The predicted voxel coordinates with each matched node moved `shrink` of the way onto its GT cell."""
    moved = predicted.astype(np.float64).copy()
    for pred_row, gt_row in pred_to_gt.items():
        moved[pred_row, 1:] += shrink * (truth[gt_row, 1:] - predicted[pred_row, 1:])
    return np.rint(moved).astype(np.int64)


def graph_of(coordinates, edge_rows):
    """A scorable graph whose node ids are its own row indices."""
    node_ids = np.arange(len(coordinates), dtype=np.int64)
    return TrackGraph(node_ids=node_ids, coordinates=coordinates.astype(np.int64), edges=node_ids[edge_rows])


def measure(submission: Path, train_dir: Path, ruler_um: float, gate_um: float):
    """Per arm, the competition score of a re-solve that saw coordinates corrected by that fraction."""
    nodes, edges = read_submission(submission)
    scale = SPACING.as_array()
    matcher = DistanceMatcher(spacing=SPACING, max_distance_um=ruler_um)
    samples: dict[str, list[VideoMetrics]] = {}
    for dataset, table in sorted(nodes.items()):
        geff = train_dir / f"{dataset}.geff"
        if not geff.exists():
            continue
        order = list(table)
        index = {node_id: row for row, node_id in enumerate(order)}
        predicted = np.array([table[node_id] for node_id in order], dtype=np.int64)
        annotated = AnnotatedTracks.from_geff(geff)
        truth = np.column_stack([annotated.graph.timepoints(), annotated.graph.positions()])
        pred_to_gt, _ = match_bipartite(predicted, truth, ruler_um)

        champion_rows = np.array([(index[source], index[target]) for source, target in edges[dataset]], dtype=np.int64)
        arms = {"champion edges": champion_rows.reshape(-1, 2)}
        for shrink in SHRINK_ARMS:
            seen = shrunk_positions(predicted, truth, pred_to_gt, shrink)
            arms[f"re-solve shrink {shrink:.1f}"] = link_by_distance(seen[:, 1:] * scale, seen[:, 0], gate_um)
        for arm, edge_rows in arms.items():
            metrics = VideoMetrics.of(graph_of(predicted, edge_rows), annotated, matcher)
            samples.setdefault(arm, []).append(metrics)
        log.info("%s: %d nodes, %d matched to GT", dataset, len(predicted), len(pred_to_gt))
    return samples


def report(samples):
    """Every arm's score and broken-edge count, and what correcting the coordinates bought the linker."""
    scores = {arm: SplitScore.of(videos) for arm, videos in samples.items()}
    for arm, split in scores.items():
        pooled = EdgeCounts.pooled(video.edges for video in samples[arm])
        log.info("%-22s GT edges recovered %d, broken %d", arm, pooled.tp, pooled.fn)
        log.info(
            "%-22s score %.4f (edge jaccard %.4f, division %.4f)",
            arm,
            split.score,
            split.adjusted_edge_jaccard,
            split.division_jaccard,
        )
    baseline = scores.get(f"re-solve shrink {SHRINK_ARMS[0]:.1f}")
    if baseline is None:
        return
    for shrink in SHRINK_ARMS[1:]:
        arm = f"re-solve shrink {shrink:.1f}"
        gain = scores[arm].score - baseline.score
        log.info("shrink %.1f buys the linker %+.4f over its own uncorrected re-solve", shrink, gain)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="a cached champion submission.csv")
    parser.add_argument("--train", type=Path, required=True, help="directory of *.geff ground truth")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    parser.add_argument("--gate", type=float, default=LINK_GATE_UM, help="link search radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    report(measure(args.submission, args.train, args.ruler, args.gate))


if __name__ == "__main__":
    main()
