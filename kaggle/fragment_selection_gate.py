"""Is edge fragmentation a CANDIDATE failure or a SELECTION failure?

A GT edge whose two endpoints are both detected and both matched, yet carries no predicted link, is
*fragmented* -- an omission, not a confusion. That single word hides two completely different bugs, and
they have opposite fixes:

    CANDIDATE   the linker never saw the pair -- no edge between those two detections was ever proposed.
                Then no solver knob reaches it; the fix is candidate generation (a wider gate).
    SELECTION   the pair was proposed, with an affinity, and the solver declined it in favour of
                terminating the track. Then it is reachable by the termination/disappearance cost, and
                the affinity head's ranking says how much of it is reachable at all.

This separates them on the cached candidate graph, which already stores every proposed edge together with
its affinity, so the question costs CPU minutes and no detector pass. Two numbers come out:

    the fraction of matched GT edges that HAVE a candidate at all        -> candidate vs selection
    the fraction whose candidate is MUTUAL-BEST for both endpoints       -> how much selection can reach

The second matters because a true edge the affinity head already ranks first is one a solver loses only to
cheap termination, whereas a true edge outranked by a rival is an affinity-quality problem that no
termination cost repairs. The margin by which the outranked ones lose says whether a *global* solver can
still flip them -- a thin margin is exactly what global optimisation buys over greedy matching.

    .venv/Scripts/python kaggle/fragment_selection_gate.py --cache <dir of *.npz> --train <dir of *.geff>
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from core.data.tracks import AnnotatedTracks
from core.geometry import Spacing

log = logging.getLogger(__name__)

SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)
OFFICIAL_UM = 7.0
COIN_FLIP = 0.5
AFFINITY_COLUMN = 2


def matched_detections(graph, coords, ruler_um):
    """Per GT node: the index of the detection matched to it in its own frame, or -1 if none is in reach."""
    scale = SPACING.as_array()
    times, positions = graph.timepoints(), graph.positions().astype(np.float64)
    matched = np.full(len(positions), -1)
    for time in np.unique(times):
        frame = times == time
        candidates = np.flatnonzero(coords[:, 0] == time)
        if len(candidates) == 0:
            continue
        distance, nearest = cKDTree(coords[candidates][:, 1:] * scale).query(positions[frame] * scale, k=1)
        matched[frame] = np.where(distance <= ruler_um, candidates[nearest], -1)
    return matched


def measure(cache_dir: Path, train_dir: Path, ruler_um: float):
    total = both_matched = has_candidate = 0
    missing_um, true_affinity, rival_affinity, mutual_best = [], [], [], []

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
        out_of, into = defaultdict(list), defaultdict(list)
        for a, b, s in zip(source, target, affinity, strict=True):
            out_of[a].append(s)
            into[b].append(s)

        graph = AnnotatedTracks.from_geff(geff).graph
        matched = matched_detections(graph, coords, ruler_um)
        row = {int(node): i for i, node in enumerate(np.asarray(graph.node_ids))}

        for head, tail in np.asarray(graph.edges):
            total += 1
            a, b = matched[row[int(head)]], matched[row[int(tail)]]
            if a < 0 or b < 0:
                continue
            both_matched += 1
            score = weight.get((int(a), int(b)))
            if score is None:
                missing_um.append(float(np.linalg.norm((coords[b, 1:] - coords[a, 1:]) * SPACING.as_array())))
                continue
            has_candidate += 1
            forward_best = max(out_of[a])
            true_affinity.append(score)
            rival_affinity.append(forward_best)
            mutual_best.append(score >= forward_best and score >= max(into[b]))

    return {
        "total": total,
        "both_matched": both_matched,
        "has_candidate": has_candidate,
        "missing_um": np.array(missing_um),
        "true_affinity": np.array(true_affinity),
        "rival_affinity": np.array(rival_affinity),
        "mutual_best": np.array(mutual_best),
    }


def report(result):
    total, matched, candidate = result["total"], result["both_matched"], result["has_candidate"]
    log.info("GT edges %d   both endpoints matched %d (%.4f)", total, matched, matched / total)
    log.info(
        "  candidate edge EXISTS %d = %.4f of matched   MISSING %d = %.4f of all GT edges",
        candidate,
        candidate / matched,
        matched - candidate,
        (matched - candidate) / total,
    )
    missing = result["missing_um"]
    if len(missing):
        spread = np.percentile(missing, [50, 90]).round(2)
        log.info("  displacement um of the MISSING q50/q90/max = %s %.2f", spread, missing.max())

    true_affinity, rival, mutual = result["true_affinity"], result["rival_affinity"], result["mutual_best"]
    band = np.percentile(true_affinity, [10, 50]).round(4)
    log.info("  MUTUAL-BEST %.4f of true edges   true affinity q10/q50 %s", mutual.mean(), band)
    outranked = ~mutual
    if outranked.any():
        log.info(
            "  outranked %d (%.4f): true %.4f vs rival %.4f, margin q50 %.4f",
            outranked.sum(),
            outranked.mean(),
            np.median(true_affinity[outranked]),
            np.median(rival[outranked]),
            np.median(rival[outranked] - true_affinity[outranked]),
        )
    weak = true_affinity < COIN_FLIP
    log.info(
        "  true affinity below coin-flip %d (%.4f), of which mutual-best %.4f",
        weak.sum(),
        weak.mean(),
        mutual[weak].mean(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached candidate graphs (*.npz)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching train *.geff")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(measure(args.cache, args.train, args.ruler))


if __name__ == "__main__":
    main()
