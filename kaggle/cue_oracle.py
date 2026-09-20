"""Does ANY available cue rank the true successor first, where the champion picked a rival?

E54 showed 76 of the champion's 105 broken GT edges are pure selection: every endpoint detected, the
candidate present, another partner ranked above it. The displacement gap (broken edges move 2.8x further
than recovered ones) suggests multi-frame motion continuity as the missing cue.

This tests that with an ORACLE. For each broken edge it ranks every detection in the next frame inside
the association gate by (a) plain distance from the source, (b) velocity extrapolated from the PREDICTED
tracklet, (c) velocity extrapolated from the cell's TRUE GT trajectory, and (d) distance to the true GT
position of the target, which is the sanity ceiling. If (c) does not beat (a), no velocity model, Kalman
filter or tracklet second pass recovers these edges -- the estimate is not the problem.

    .venv/Scripts/python kaggle/cue_oracle.py --submission <submission.csv> --train <geff dir>
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from error_budget import match_bipartite
from fragment_selection_gate import OFFICIAL_UM, SPACING
from short_component_prune import read_submission

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

ASSOCIATION_GATE_UM = 10.0
HISTORY_FRAMES = 4
MINIMUM_CHAIN = 2


def walk_back(node, predecessors, positions, times):
    """The chain of unambiguous predecessors, and the constant velocity it implies."""
    chain, current = [node], node
    while len(chain) < HISTORY_FRAMES:
        previous = predecessors.get(current)
        if previous is None or len(previous) != 1:
            break
        current = previous[0]
        chain.append(current)
    span = times[chain[0]] - times[chain[-1]]
    if len(chain) < MINIMUM_CHAIN or span == 0:
        return np.zeros(positions.shape[1])
    return (positions[chain[0]] - positions[chain[-1]]) / span


def gt_predecessors(gt_edges):
    parents, children = {}, {}
    for source, target in gt_edges:
        parents[target] = [source]
        children.setdefault(source, []).append(target)
    return parents, children


def rank_of(target_row, candidates, reference, positions):
    """How many candidates the reference point ranks ABOVE the true successor."""
    distances = np.linalg.norm(positions[candidates] - reference, axis=1)
    return int((distances < distances[target_row]).sum())


def probe(submission: Path, train_dir: Path, ruler_um: float, gate_um: float):
    nodes, edges = read_submission(submission)
    scale = SPACING.as_array()
    tally: Counter[str] = Counter()
    for dataset, table in nodes.items():
        geff = train_dir / f"{dataset}.geff"
        if not geff.exists():
            continue
        order = list(table)
        index = {node_id: position for position, node_id in enumerate(order)}
        predicted = np.array([table[node_id] for node_id in order], dtype=float)
        pred_xyz, pred_t = predicted[:, 1:] * scale, predicted[:, 0].astype(int)

        graph = AnnotatedTracks.from_geff(geff).graph
        truth = np.column_stack([graph.timepoints(), graph.positions().astype(float)])
        gt_xyz, gt_t = truth[:, 1:] * scale, truth[:, 0].astype(int)
        _, gt_to_pred = match_bipartite(predicted, truth, ruler_um)

        gt_edges = [(int(source), int(target)) for source, target in graph.edge_rows()]
        gt_parents, gt_children = gt_predecessors(gt_edges)
        pred_edges = {(index[source], index[target]) for source, target in edges[dataset]}
        pred_parents: dict[int, list[int]] = {}
        for source, target in pred_edges:
            pred_parents.setdefault(target, []).append(source)
        frames = {frame: np.flatnonzero(pred_t == frame) for frame in np.unique(pred_t)}

        for source, target in gt_edges:
            mapped_source, mapped_target = gt_to_pred.get(source), gt_to_pred.get(target)
            if mapped_source is None or mapped_target is None:
                continue
            if (mapped_source, mapped_target) in pred_edges or len(gt_children[source]) > 1:
                continue
            next_frame = frames.get(pred_t[mapped_source] + 1)
            if next_frame is None:
                continue
            reachable = next_frame[np.linalg.norm(pred_xyz[next_frame] - pred_xyz[mapped_source], axis=1) <= gate_um]
            listed = reachable.tolist()
            if mapped_target not in listed:
                tally["true successor outside the gate"] += 1
                continue
            row = listed.index(mapped_target)
            tally["selection failures ranked"] += 1
            tally["distance"] += rank_of(row, reachable, pred_xyz[mapped_source], pred_xyz) == 0
            predicted_velocity = walk_back(mapped_source, pred_parents, pred_xyz, pred_t)
            tally["predicted-tracklet velocity"] += (
                rank_of(row, reachable, pred_xyz[mapped_source] + predicted_velocity, pred_xyz) == 0
            )
            true_velocity = walk_back(source, gt_parents, gt_xyz, gt_t)
            tally["ORACLE true-trajectory velocity"] += (
                rank_of(row, reachable, pred_xyz[mapped_source] + true_velocity, pred_xyz) == 0
            )
            tally["ORACLE true target position (ceiling)"] += rank_of(row, reachable, gt_xyz[target], pred_xyz) == 0
    return tally


def report(tally):
    total = tally["selection failures ranked"]
    log.info("%5d  selection failures ranked", total)
    for cue in (
        "distance",
        "predicted-tracklet velocity",
        "ORACLE true-trajectory velocity",
        "ORACLE true target position (ceiling)",
    ):
        log.info("%5d / %d ranked first by %s", tally[cue], total, cue)
    log.info(
        "%5d  true successor outside the gate (unreachable by any solver)", tally["true successor outside the gate"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="a cached champion submission.csv")
    parser.add_argument("--train", type=Path, required=True, help="directory of *.geff ground truth")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    parser.add_argument("--gate", type=float, default=ASSOCIATION_GATE_UM, help="association gate in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    report(probe(args.submission, args.train, args.ruler, args.gate))


if __name__ == "__main__":
    main()
