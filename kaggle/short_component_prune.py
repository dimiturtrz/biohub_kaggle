"""Is SHORT5 -- deleting every track component of <= N nodes -- a free prune or a recall tax?

`howonkang/biohub-0947-short5-prepp-r1` applies this to the RAW geff before the champion's own
postprocessing: treat predicted edges as undirected, drop every connected component with <= 5 nodes,
touch nothing else. It is distinct from the champion's `OUTPUT_MIN_TRACK_LEN = 6`, which acts after
postprocessing on tracks and is held back by `OUTPUT_KEEP_DIVISION_COMPONENTS = 1`.

Every prune answers to E47's spec, which is ASYMMETRIC because a deleted real cell destroys the true
edge on BOTH its sides: FP-recall ~0.75 at <= 2 % collateral earns +0.035, while at 5 % collateral even a
perfect pruner nets only +0.021. E49 already measured per-detection temporal support as a NULL (AUC
0.484), which is the same intuition one level down -- but component size in the LINKED graph is not that
feature: it is a property of the solver's output, and a solver that gave a node no partners is making a
different statement than a detector that gave it no continuation.

So the question is one sweep over a cached champion submission: at each component-size cut, what fraction
of the FP nodes does it remove, and what fraction of the REAL ones does it take with them.

    .venv/Scripts/python kaggle/short_component_prune.py --submission <submission.csv> --train <geff dir>
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from fragment_selection_gate import OFFICIAL_UM, SPACING
from scipy.spatial import cKDTree

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

SEPARATION_UM = 2.87
SIZE_CUTS = (2, 3, 5, 8, 12)
SHORT5_CUT = 5
COLLATERAL_BUDGET = 0.02
REQUIRED_RECALL = 0.75


def read_submission(path: Path):
    """Nodes and undirected edges per dataset, exactly as the champion wrote them."""
    nodes, edges = defaultdict(dict), defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            if row["row_type"] == "node":
                nodes[dataset][int(row["node_id"])] = (
                    float(row["t"]),
                    float(row["z"]),
                    float(row["y"]),
                    float(row["x"]),
                )
            else:
                edges[dataset].append((int(row["source_id"]), int(row["target_id"])))
    return nodes, edges


def component_sizes(node_ids, edges):
    """Union-find over undirected edges; returns each node's component size."""
    parent = dict.fromkeys(node_ids)
    for node_id in node_ids:
        parent[node_id] = node_id

    def find(node_id):
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    for source, target in edges:
        left, right = find(source), find(target)
        if left != right:
            parent[left] = right
    counts: dict[int, int] = defaultdict(int)
    roots = {node_id: find(node_id) for node_id in node_ids}
    for root in roots.values():
        counts[root] += 1
    return np.array([counts[roots[node_id]] for node_id in node_ids])


def label_real(positions, times, truth_positions, truth_times, ruler_um):
    """A predicted node is REAL if some GT cell in its own frame lies within the ruler."""
    scale = SPACING.as_array()
    real = np.zeros(len(positions), dtype=bool)
    for frame in np.unique(times):
        predicted = np.flatnonzero(times == frame)
        actual = truth_positions[truth_times == frame]
        if len(actual) == 0 or len(predicted) == 0:
            continue
        distance, _ = cKDTree(actual * scale).query(positions[predicted] * scale, k=1)
        real[predicted] = distance <= ruler_um
    return real


def collect(submission: Path, train_dir: Path, ruler_um: float):
    nodes, edges = read_submission(submission)
    sizes, labels = [], []
    for dataset, table in nodes.items():
        geff = train_dir / f"{dataset}.geff"
        if not geff.exists():
            log.warning("  %s has no GT in --train; skipped", dataset)
            continue
        node_ids = list(table)
        stacked = np.array([table[node_id] for node_id in node_ids], dtype=float)
        graph = AnnotatedTracks.from_geff(geff).graph
        real = label_real(stacked[:, 1:], stacked[:, 0], graph.positions().astype(float), graph.timepoints(), ruler_um)
        size = component_sizes(node_ids, edges[dataset])
        log.info(
            "  %s nodes %d  real %.4f  components<=%d hold %d nodes",
            dataset,
            len(node_ids),
            real.mean(),
            SHORT5_CUT,
            int((size <= SHORT5_CUT).sum()),
        )
        sizes.append(size)
        labels.append(real)
    return np.concatenate(sizes), np.concatenate(labels)


def report(sizes, labels, ruler_um):
    log.info(
        "pooled nodes %d   real %d (%.4f)   ruler %.2f um",
        len(labels),
        int(labels.sum()),
        labels.mean(),
        ruler_um,
    )
    for cut in SIZE_CUTS:
        removed = sizes <= cut
        log.info(
            "  component <= %2d: removes %5d nodes | FP-recall %.4f | collateral %.4f %s",
            cut,
            int(removed.sum()),
            (removed & ~labels).sum() / max((~labels).sum(), 1),
            (removed & labels).sum() / max(labels.sum(), 1),
            "<= E47 budget" if (removed & labels).sum() / max(labels.sum(), 1) <= COLLATERAL_BUDGET else "",
        )
    log.info("E47 requires FP-recall %.2f at <= %.2f collateral", REQUIRED_RECALL, COLLATERAL_BUDGET)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True, help="a cached champion submission.csv")
    parser.add_argument("--train", type=Path, required=True, help="directory of *.geff ground truth")
    parser.add_argument("--ruler", type=float, default=SEPARATION_UM, help="match radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    sizes, labels = collect(args.submission, args.train, args.ruler)
    report(sizes, labels, args.ruler)
    if args.ruler != OFFICIAL_UM:
        sizes, labels = collect(args.submission, args.train, OFFICIAL_UM)
        report(sizes, labels, OFFICIAL_UM)


if __name__ == "__main__":
    main()
