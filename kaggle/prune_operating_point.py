"""What must an FP pruner achieve before it pays? Prices bd `g89y` before anyone builds it.

E46 found the detection that outranks a true edge is an unmatched FP in 0.9698 of contests, which makes
the outranked band PRUNABLE in principle. This asks what "in principle" costs, on CPU, from the cache:

    ORACLE      delete every unmatched detection -- the ceiling of the whole axis (row r=1.00, q=0.00).
    OPERATING   delete a fraction r of FPs while collaterally deleting a fraction q of REAL cells,
                which is what any real pruner does.

The verdict lives in the COLLATERAL column, not the recall one: deleting a real cell destroys the true
edges on BOTH sides of it, so q enters at roughly twice its own size while r only rescues a band that is
0.1195 wide. Read `net`, never `mutual-best` alone.

Errors are drawn uniformly at random, which FLATTERS a real pruner: its mistakes concentrate on faint
and crowded cells (E41), the same population the rescued edges live in. Treat the gains as upper bounds
at a given q.

    .venv/Scripts/python kaggle/prune_operating_point.py --cache <dir of *.npz> --train <dir of *.geff>
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from fragment_selection_gate import OFFICIAL_UM, matched_detections

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

AFFINITY_COLUMN = 2
FP_RECALLS = (0.25, 0.50, 0.75, 0.90, 1.00)
COLLATERALS = (0.0, 0.02, 0.05)


def grid():
    return [(recall, collateral) for recall in FP_RECALLS for collateral in COLLATERALS]


def true_edges(graph, matched, row, weight):
    """GT edges whose endpoints are both detected AND carry a proposed candidate, with its affinity."""
    found = []
    for head, tail in np.asarray(graph.edges):
        source, target = matched[row[int(head)]], matched[row[int(tail)]]
        if source < 0 or target < 0:
            continue
        score = weight.get((int(source), int(target)))
        if score is not None:
            found.append((int(source), int(target), score))
    return found


def survivors(real, recall, collateral, rng):
    """Which detections a pruner of this quality leaves standing."""
    roll = rng.random(len(real))
    return np.where(real, roll >= collateral, roll >= recall)


def collect(cache_dir: Path, train_dir: Path, ruler_um: float, seed: int):
    rng = np.random.default_rng(seed)
    kept, destroyed = defaultdict(int), defaultdict(int)
    total, baseline = 0, 0

    for path in sorted(cache_dir.glob("*.npz")):
        geff = train_dir / f"{path.stem}.geff"
        if not geff.exists():
            continue
        cached = np.load(path)
        coords = cached["coords"].astype(np.float64)
        proposed = cached["edges"]
        source = proposed[:, 0].astype(int)
        target = proposed[:, 1].astype(int)
        affinity = proposed[:, AFFINITY_COLUMN]
        weight = {(int(a), int(b)): s for a, b, s in zip(source, target, affinity, strict=True)}

        graph = AnnotatedTracks.from_geff(geff).graph
        matched = matched_detections(graph, coords, ruler_um)
        row = {int(node): i for i, node in enumerate(np.asarray(graph.node_ids))}
        real = np.zeros(len(coords), dtype=bool)
        real[[index for index in matched if index >= 0]] = True

        truth = true_edges(graph, matched, row, weight)
        total += len(truth)
        baseline += sum(mutual_best(truth, source, target, affinity, np.ones(len(coords), dtype=bool)))

        for recall, collateral in grid():
            alive = survivors(real, recall, collateral, rng)
            wins = mutual_best(truth, source, target, affinity, alive)
            for (a, b, _), won in zip(truth, wins, strict=True):
                if not (alive[a] and alive[b]):
                    destroyed[recall, collateral] += 1
                elif won:
                    kept[recall, collateral] += 1
    return kept, destroyed, total, baseline


def mutual_best(truth, source, target, affinity, alive):
    """Per true edge, whether it is the best proposal out of its source AND into its target."""
    out_of, into = defaultdict(list), defaultdict(list)
    for a, b, score in zip(source, target, affinity, strict=True):
        if alive[a] and alive[b]:
            out_of[a].append(score)
            into[b].append(score)
    return [alive[a] and alive[b] and score >= max(out_of[a]) and score >= max(into[b]) for a, b, score in truth]


def report(kept, destroyed, total, baseline):
    reference = baseline / total
    log.info("true edges n=%d   baseline mutual-best %.4f (outranked band %.4f)", total, reference, 1 - reference)
    for recall, collateral in grid():
        rescued = kept[recall, collateral] / total
        log.info(
            "  FP-recall %.2f  collateral %.2f   mutual-best %.4f  lost-to-prune %.4f  net %+.4f",
            recall,
            collateral,
            rescued,
            destroyed[recall, collateral] / total,
            rescued - reference,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached candidate graphs (*.npz)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching train *.geff")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    parser.add_argument("--seed", type=int, default=0, help="seed for the pruner's error draw")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(*collect(args.cache, args.train, args.ruler, args.seed))


if __name__ == "__main__":
    main()
