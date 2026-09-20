"""Where is the champion's remaining score, once only METRIC-VISIBLE errors are counted?

E53 established that an edge between two unannotated detections is neither TP nor FP -- the scorer's
confusion only sees edges with a matched endpoint. Every error budget we have drawn before counted
candidate-set quantities instead, so none of them priced an axis in the units the leaderboard pays in.

This scores the cached champion submission with the official rule, faithfully: bipartite Hungarian node
matching at 7 um on the real voxel scale, the same TP/FP test the scorer uses, the signed node-count
term, and the champion's own error decomposition (detection / fragmentation / wrong-association). The
output says how many points each axis is worth if it were solved PERFECTLY, which is the number every
future proposal has to be cheaper than.

    .venv/Scripts/python kaggle/error_budget.py --submission <submission.csv> --train <geff dir>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from fragment_selection_gate import OFFICIAL_UM, SPACING
from scipy.optimize import linear_sum_assignment
from short_component_prune import read_submission

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

NODE_COUNT_PENALTY_A = 0.1
UNREACHABLE = 1e6


def match_bipartite(predicted, truth, ruler_um):
    """The scorer's own matching: one-to-one Hungarian per frame, gated at the ruler."""
    scale = SPACING.as_array()
    pred_to_gt, gt_to_pred = {}, {}
    for frame in np.unique(truth[:, 0]):
        pred_rows = np.flatnonzero(predicted[:, 0] == frame)
        gt_rows = np.flatnonzero(truth[:, 0] == frame)
        if len(pred_rows) == 0 or len(gt_rows) == 0:
            continue
        difference = predicted[pred_rows, 1:][:, None, :] * scale - truth[gt_rows, 1:][None, :, :] * scale
        cost = np.sqrt((difference**2).sum(axis=-1))
        gated = np.where(cost <= ruler_um, cost, UNREACHABLE)
        for row, column in zip(*linear_sum_assignment(gated), strict=True):
            if gated[row, column] < UNREACHABLE:
                pred_to_gt[int(pred_rows[row])] = int(gt_rows[column])
                gt_to_pred[int(gt_rows[column])] = int(pred_rows[row])
    return pred_to_gt, gt_to_pred


def confusion(pred_edges, gt_edges, pred_to_gt):
    """TP/FP/FN exactly as the scorer counts them -- an edge with no matched endpoint counts nowhere."""
    outgoing, incoming = {}, {}
    for source, target in gt_edges:
        outgoing.setdefault(source, set()).add(target)
        incoming[target] = source

    true_positive, false_positive, invisible, matched = 0, 0, 0, set()
    for source, target in pred_edges:
        mapped_source, mapped_target = pred_to_gt.get(source), pred_to_gt.get(target)
        if mapped_source is not None and mapped_target in outgoing.get(mapped_source, ()):
            true_positive += 1
            matched.add((mapped_source, mapped_target))
        elif (mapped_target is not None and mapped_target in incoming) or bool(outgoing.get(mapped_source, ())):
            false_positive += 1
        else:
            invisible += 1
    return true_positive, false_positive, len(set(gt_edges) - matched), invisible


def decompose(gt_edges, pred_edges, gt_to_pred, pred_to_gt):
    """The champion's split of the error mass: detection vs fragmentation vs wrong association."""
    pred_edge_set = set(pred_edges)
    outgoing: dict[int, set[int]] = {}
    for source, target in gt_edges:
        outgoing.setdefault(source, set()).add(target)

    recovered = fragmented = lost = 0
    for source, target in set(gt_edges):
        mapped_source, mapped_target = gt_to_pred.get(source), gt_to_pred.get(target)
        if mapped_source is None or mapped_target is None:
            lost += 1
        elif (mapped_source, mapped_target) in pred_edge_set:
            recovered += 1
        else:
            fragmented += 1

    wrong = sum(
        1
        for source, target in pred_edge_set
        if (mapped := pred_to_gt.get(source)) is not None
        and pred_to_gt.get(target) not in (None, *outgoing.get(mapped, ()))
    )
    return recovered, fragmented, lost, wrong


def measure(submission: Path, train_dir: Path, ruler_um: float):
    nodes, edges = read_submission(submission)
    rows = []
    for dataset, table in nodes.items():
        geff = train_dir / f"{dataset}.geff"
        if not geff.exists():
            continue
        order = list(table)
        index = {node_id: position for position, node_id in enumerate(order)}
        predicted = np.array([table[node_id] for node_id in order], dtype=float)
        annotated = AnnotatedTracks.from_geff(geff)
        graph = annotated.graph
        truth = np.column_stack([graph.timepoints(), graph.positions().astype(float)])

        pred_to_gt, gt_to_pred = match_bipartite(predicted, truth, ruler_um)
        pred_edges = [(index[source], index[target]) for source, target in edges[dataset]]
        gt_edges = [(int(source), int(target)) for source, target in graph.edge_rows()]

        true_positive, false_positive, false_negative, invisible = confusion(pred_edges, gt_edges, pred_to_gt)
        rows.append(
            (
                dataset,
                true_positive,
                false_positive,
                false_negative,
                invisible,
                len(order),
                annotated.estimated_node_count,
                decompose(gt_edges, pred_edges, gt_to_pred, pred_to_gt),
            )
        )
    return rows


def report(rows):
    weighted, total_weight = 0.0, 0
    for dataset, tp, fp, fn, invisible, n_pred, t_true, parts in rows:
        jaccard = tp / max(tp + fp + fn, 1)
        multiplier = 1.0 if not t_true else 1.0 - NODE_COUNT_PENALTY_A * (n_pred - t_true) / t_true
        weight = tp + fp + fn
        weighted += max(0.0, jaccard * multiplier) * weight
        total_weight += weight
        recovered, fragmented, lost, wrong = parts
        log.info(
            "%s  jac %.4f x %.4f = %.4f | tp %5d fp %5d fn %5d | invisible edges %6d (%.1f%% of pred)",
            dataset,
            jaccard,
            multiplier,
            max(0.0, jaccard * multiplier),
            tp,
            fp,
            fn,
            invisible,
            100 * invisible / max(tp + fp + invisible, 1),
        )
        log.info(
            "    GT edges: recovered %5d  fragmented %5d  lost-to-detection %5d | wrong-association %5d",
            recovered,
            fragmented,
            lost,
            wrong,
        )

    tp = sum(row[1] for row in rows)
    fp = sum(row[2] for row in rows)
    fn = sum(row[3] for row in rows)
    recovered, fragmented, lost, wrong = (sum(row[7][column] for row in rows) for column in range(4))
    log.info("POOLED weighted adjusted jaccard %.4f", weighted / max(total_weight, 1))
    log.info("  perfect association (fp->0, wrong->0): %.4f", tp / max(tp + fn, 1))
    log.info("  perfect fragmentation (frag->tp):      %.4f", (tp + fragmented) / max(tp + fp + fn, 1))
    log.info("  perfect detection (lost->tp):          %.4f", (tp + lost) / max(tp + fp + fn, 1))
    log.info("  current pooled jaccard:                %.4f", tp / max(tp + fp + fn, 1))
    log.info(
        "  fragmented %d / lost %d / wrong %d of %d GT edges", fragmented, lost, wrong, recovered + fragmented + lost
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="a cached champion submission.csv")
    parser.add_argument("--train", type=Path, required=True, help="directory of *.geff ground truth")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    report(measure(args.submission, args.train, args.ruler))


if __name__ == "__main__":
    main()
