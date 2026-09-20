"""Does making track termination EXPENSIVE recover fragmented edges, or cost them?

E43 showed edge fragmentation is a SELECTION failure: 99.51 % of fragmented GT edges already have a
proposed candidate carrying an affinity, so the solver saw the pair and declined it in favour of ending
the track. The obvious next move is to raise the termination (disappearance) cost until continuing a
track is cheaper than ending it. This sweeps that knob on the cached candidate graph, with no detector
pass and no GPU.

The cached graph is pure dt=1 -- every candidate links consecutive frames -- so the faithful model of the
linker is a per-frame-pair global assignment where every node may instead pay a constant `term` to go
unmatched. Costs are -log(affinity); the termination option is a square augmentation. The assignment
decomposes over the connected components OF EACH FRAME PAIR (not of the movie -- dt=1 edges chain all
frames into one component and the decomposition buys nothing).

DENOMINATORS, because the naive ones lie here:

    recall      true edges taken / all GT edges. Honest.
    mislinks    counted ONLY over links whose two endpoints are both GT-matched detections. The naive
                version -- every taken link that is not a GT edge -- sits at 0.98 for every value of
                `term` and measures nothing, because the candidate field carries ~980 FP per frame
                against ~17 GT cells, so almost every link is FP-to-FP and was never a mistake about a
                real cell.

The level is not comparable to the real tracker (this model has no distance gate or division term); the
SHAPE across `term` is the result.

    .venv/Scripts/python kaggle/termination_cost_sweep.py --cache <dir of *.npz> --train <dir of *.geff>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from fragment_selection_gate import OFFICIAL_UM, matched_detections
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

FORBIDDEN = 1e6
AFFINITY_COLUMN = 2
DEFAULT_TERMS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def truth_and_nodes(geff: Path, coords, ruler_um: float):
    """GT edges rewritten as detection-index pairs, plus the set of detections a GT cell matched."""
    graph = AnnotatedTracks.from_geff(geff).graph
    matched = matched_detections(graph, coords, ruler_um)
    row = {int(node): i for i, node in enumerate(np.asarray(graph.node_ids))}
    edges = set()
    for head, tail in np.asarray(graph.edges):
        source, target = matched[row[int(head)]], matched[row[int(tail)]]
        if source >= 0 and target >= 0:
            edges.add((int(source), int(target)))
    return edges, set(matched[matched >= 0].tolist()), len(np.asarray(graph.edges))


def components(coords, source, target):
    """Yield the candidate-edge indices of each connected component of each consecutive frame pair.

    The candidate graph is pure dt=1, so components taken over the WHOLE movie chain every frame into one
    blob and buy nothing; they have to be taken per frame pair.
    """
    frame = coords[:, 0].astype(int)
    order = np.argsort(frame[source], kind="stable")
    bound = np.searchsorted(frame[source][order], np.arange(frame.max() + 2))
    for time in range(frame.max() + 1):
        block = order[bound[time] : bound[time + 1]]
        if len(block) == 0:
            continue
        rows = np.unique(source[block], return_inverse=True)[1]
        columns = np.unique(target[block], return_inverse=True)[1]
        width = rows.max() + 1
        size = width + columns.max() + 1
        pairs = coo_matrix((np.ones(len(block)), (rows, width + columns)), shape=(size, size))
        label = connected_components(pairs, directed=False)[1][rows]
        for component in np.unique(label):
            yield block[label == component]


def assign(costs, term: float):
    """Global assignment with a per-node termination option, as a square augmentation."""
    n_rows, n_columns = costs.shape
    size = n_rows + n_columns
    square = np.zeros((size, size))
    square[:n_rows, :n_columns] = costs
    square[:n_rows, n_columns:] = FORBIDDEN
    square[n_rows:, :n_columns] = FORBIDDEN
    np.fill_diagonal(square[:n_rows, n_columns:], term)
    np.fill_diagonal(square[n_rows:, :n_columns].T, term)
    rows, columns = linear_sum_assignment(square)
    keep = (rows < n_rows) & (columns < n_columns)
    return rows[keep], columns[keep]


def sweep(cache_dir: Path, train_dir: Path, ruler_um: float, terms):
    taken = dict.fromkeys(terms, 0)
    true = dict.fromkeys(terms, 0)
    judged = dict.fromkeys(terms, 0)
    gt_edges = 0

    for path in sorted(cache_dir.glob("*.npz")):
        geff = train_dir / f"{path.stem}.geff"
        if not geff.exists():
            continue
        cached = np.load(path)
        coords = cached["coords"].astype(np.float64)
        proposed = cached["edges"]
        source, target = proposed[:, 0].astype(int), proposed[:, 1].astype(int)
        cost = -np.log(np.clip(proposed[:, AFFINITY_COLUMN], 1e-6, 1 - 1e-6))

        edges, gt_nodes, count = truth_and_nodes(geff, coords, ruler_um)
        gt_edges += count

        for block in components(coords, source, target):
            heads, rows = np.unique(source[block], return_inverse=True)
            tails, columns = np.unique(target[block], return_inverse=True)
            grid = np.full((len(heads), len(tails)), FORBIDDEN)
            grid[rows, columns] = cost[block]
            for term in terms:
                for row, column in zip(*assign(grid, term), strict=True):
                    if grid[row, column] >= FORBIDDEN / 2:
                        continue
                    a, b = int(heads[row]), int(tails[column])
                    taken[term] += 1
                    if a in gt_nodes and b in gt_nodes:
                        judged[term] += 1
                        true[term] += (a, b) in edges

    return gt_edges, taken, true, judged


def report(gt_edges, taken, true, judged, terms):
    log.info("GT edges %d", gt_edges)
    for term in terms:
        wrong = judged[term] - true[term]
        log.info(
            "  term=%-5s taken %7d  recall %.4f  judged(GT-matched both ends) %6d  mislinks %5d (%.4f)",
            term,
            taken[term],
            true[term] / max(gt_edges, 1),
            judged[term],
            wrong,
            wrong / max(judged[term], 1),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached candidate graphs (*.npz)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching train *.geff")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    parser.add_argument("--terms", type=float, nargs="+", default=list(DEFAULT_TERMS), help="termination costs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    terms = tuple(args.terms)
    report(*sweep(args.cache, args.train, args.ruler, terms), terms=terms)


if __name__ == "__main__":
    main()
