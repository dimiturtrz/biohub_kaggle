"""Does the champion already HOLD a broken GT edge, between nodes the matcher happened not to pick?

The scorer represents each GT cell by exactly one predicted node -- the one a global Hungarian assignment
gives it, gated at 7 um. E57 showed the node chosen for a broken edge sits median 4.08 um from the cell,
three times further than for a recovered one. That leaves a question the displacement table cannot answer:
is the champion's edge really absent, or does it exist between a DIFFERENT pair of nodes that are also
inside the ruler of the same two GT cells, and merely lost the assignment?

The two cases want opposite work. A genuinely absent edge needs a better tracker. An edge that exists but
is represented by the wrong node is a node-SELECTION defect: the tracker is right, the submission simply
offers the scorer a worse representative, and thinning or re-centring detections could hand over the one
that is already linked -- no re-solve, no training.

So for every GT edge the champion failed to recover, this collects every predicted node within the ruler
of the GT source and of the GT target, and asks whether ANY predicted edge joins the two sets. Recovered
edges run through the same test as a control: they must come back at 100%, since their own matched pair
qualifies.

    .venv/Scripts/python kaggle/representative_alternates.py --submission <submission.csv> --train <geff dir>
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from error_budget import match_bipartite
from fragment_selection_gate import OFFICIAL_UM, SPACING
from scipy.spatial import cKDTree
from short_component_prune import read_submission

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)


def in_ruler(tree, point_um, ruler_um, rows):
    """The predicted rows within the ruler of one GT position."""
    return rows[tree.query_ball_point(point_um, ruler_um)]


def probe(submission: Path, train_dir: Path, ruler_um: float):
    nodes, edges = read_submission(submission)
    scale = SPACING.as_array()
    tally: Counter[str] = Counter()
    alternates: list[int] = []

    for dataset, table in nodes.items():
        geff = train_dir / f"{dataset}.geff"
        if not geff.exists():
            continue
        order = list(table)
        index = {node_id: position for position, node_id in enumerate(order)}
        predicted = np.array([table[node_id] for node_id in order], dtype=float)
        pred_um, pred_t = predicted[:, 1:] * scale, predicted[:, 0].astype(int)

        graph = AnnotatedTracks.from_geff(geff).graph
        truth = np.column_stack([graph.timepoints(), graph.positions().astype(float)])
        gt_um, gt_t = truth[:, 1:] * scale, truth[:, 0].astype(int)
        _, gt_to_pred = match_bipartite(predicted, truth, ruler_um)

        pred_edges = {(index[source], index[target]) for source, target in edges[dataset]}
        degree: Counter[int] = Counter()
        for endpoints in pred_edges:
            degree.update(endpoints)
        frames = {frame: np.flatnonzero(pred_t == frame) for frame in np.unique(pred_t)}
        trees = {frame: cKDTree(pred_um[rows]) for frame, rows in frames.items()}

        for source, target in graph.edge_rows():
            source, target = int(source), int(target)
            mapped_source, mapped_target = gt_to_pred.get(source), gt_to_pred.get(target)
            if mapped_source is None or mapped_target is None:
                tally["endpoint unmatched (out of scope)"] += 1
                continue
            source_frame, target_frame = int(gt_t[source]), int(gt_t[target])
            if source_frame not in trees or target_frame not in trees:
                continue
            near_source = in_ruler(trees[source_frame], gt_um[source], ruler_um, frames[source_frame])
            near_target = in_ruler(trees[target_frame], gt_um[target], ruler_um, frames[target_frame])
            bucket = "recovered" if (mapped_source, mapped_target) in pred_edges else "broken"
            tally[bucket] += 1
            tally[f"{bucket}: candidate nodes near source"] += len(near_source)

            held = {(a, b) for a in near_source.tolist() for b in near_target.tolist()} & pred_edges
            tally[f"{bucket}: an in-ruler pair IS linked"] += bool(held)
            if bucket == "broken" and held:
                alternates.append(len(held))
                tally["decoy source edges"] += degree[mapped_source]
                tally["decoy target edges"] += degree[mapped_target]
                tally["decoy source is isolated"] += degree[mapped_source] == 0
                tally["decoy target is isolated"] += degree[mapped_target] == 0

        tally["predicted nodes"] += len(predicted)
        tally["nodes with a same-frame neighbour inside the ruler"] += sum(
            int((trees[frame].query_ball_point(pred_um[rows], ruler_um, return_length=True) > 1).sum())
            for frame, rows in frames.items()
        )
    return tally, np.array(alternates, dtype=float)


def report(tally, alternates, ruler_um):
    log.info("ruler %.1f um", ruler_um)
    for bucket in ("recovered", "broken"):
        total = max(tally[bucket], 1)
        log.info(
            "%5d %-10s  %4d have an in-ruler pair already linked (%.3f), %.2f candidate nodes near the source",
            tally[bucket],
            bucket,
            tally[f"{bucket}: an in-ruler pair IS linked"],
            tally[f"{bucket}: an in-ruler pair IS linked"] / total,
            tally[f"{bucket}: candidate nodes near source"] / total,
        )
    log.info("%5d GT edges out of scope (an endpoint matched nothing)", tally["endpoint unmatched (out of scope)"])
    if len(alternates):
        log.info("broken edges that ARE held elsewhere carry %.2f such links each on average", alternates.mean())
    log.info(
        "  the %d decoys that win the assignment carry %.2f / %.2f edges (source / target); %d and %d are isolated",
        len(alternates),
        tally["decoy source edges"] / max(len(alternates), 1),
        tally["decoy target edges"] / max(len(alternates), 1),
        tally["decoy source is isolated"],
        tally["decoy target is isolated"],
    )
    log.info(
        "%6d of %d predicted nodes have a same-frame neighbour inside the ruler (%.3f) -- the collateral base",
        tally["nodes with a same-frame neighbour inside the ruler"],
        tally["predicted nodes"],
        tally["nodes with a same-frame neighbour inside the ruler"] / max(tally["predicted nodes"], 1),
    )
    log.info("recovered must read 1.000 -- its own matched pair qualifies; anything less is a bug in this probe.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="a cached champion submission.csv")
    parser.add_argument("--train", type=Path, required=True, help="directory of *.geff ground truth")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    report(*probe(args.submission, args.train, args.ruler), args.ruler)


if __name__ == "__main__":
    main()
