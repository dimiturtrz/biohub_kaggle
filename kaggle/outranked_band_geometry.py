"""Can any GEOMETRIC cue rescue the true edges the affinity head ranks below a rival?

E43 left one live block: 11.95 % of true candidate edges are outranked by a rival, by a median affinity
margin of only 0.0985. A margin that thin invites a tie-breaker, and the cache already carries a
displacement column nothing in the cost reads. This asks three questions, all on CPU, in order of how
cheaply they can kill the idea:

    1. WHAT IS THE BAND?   displacement of outranked vs mutual-best vs all true edges, and the
                           conditional P(outranked | large displacement) against its marginal.
    2. DOES DISTANCE HELP? in the band, is the true edge closer than the rival that beat it?
    3. DOES VELOCITY?      using GT positions to build x_t + (x_t - x_{t-1}) -- an ORACLE, so it upper
                           bounds any learned motion feature.

Question 3 measures from the GT source position to detections, and the true detection was itself chosen
as the one nearest the GT tail, so that frame is biased TOWARD the true edge. Read the velocity-minus-
static DELTA, never the level; the bias being optimistic is what lets a weak result kill the axis.

    .venv/Scripts/python kaggle/outranked_band_geometry.py --cache <dir of *.npz> --train <dir of *.geff>
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from fragment_selection_gate import OFFICIAL_UM, SPACING, matched_detections

from core.data.tracks import AnnotatedTracks

log = logging.getLogger(__name__)

AFFINITY_COLUMN = 2
DISTANCE_COLUMN = 3
LARGE_DISPLACEMENT_PERCENTILE = 90


def rivals(proposed):
    """Per source detection, the best (affinity, target) proposed out of it."""
    best = defaultdict(list)
    for source, target, affinity in zip(
        proposed[:, 0].astype(int), proposed[:, 1].astype(int), proposed[:, AFFINITY_COLUMN], strict=True
    ):
        best[source].append((affinity, target))
    return best


def collect(cache_dir: Path, train_dir: Path, ruler_um: float):
    displacement = {"all": [], "mutual": [], "outranked": []}
    closer, velocity_true, static_true, margin_gain = [], [], [], []

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

        weight = {
            (int(a), int(b)): (s, d)
            for a, b, s, d in zip(source, target, affinity, proposed[:, DISTANCE_COLUMN], strict=True)
        }
        out_of = rivals(proposed)
        into = defaultdict(list)
        for b, s in zip(target, affinity, strict=True):
            into[b].append(s)

        graph = AnnotatedTracks.from_geff(geff).graph
        matched = matched_detections(graph, coords, ruler_um)
        row = {int(node): i for i, node in enumerate(np.asarray(graph.node_ids))}
        positions = graph.positions().astype(np.float64)
        predecessor = {int(tail): int(head) for head, tail in np.asarray(graph.edges)}

        for head, tail in np.asarray(graph.edges):
            a, b = matched[row[int(head)]], matched[row[int(tail)]]
            if a < 0 or b < 0:
                continue
            found = weight.get((int(a), int(b)))
            if found is None:
                continue
            score, distance = found
            displacement["all"].append(distance)
            rival_score, rival_target = max(out_of[a])
            if score >= rival_score and score >= max(into[b]):
                displacement["mutual"].append(distance)
                continue
            displacement["outranked"].append(distance)
            if rival_target == b:
                continue
            closer.append((distance, weight[(int(a), int(rival_target))][1]))
            before = predecessor.get(int(head))
            if before is None:
                continue
            here = positions[row[int(head)]] * SPACING.as_array()
            predicted = here + (here - positions[row[before]] * SPACING.as_array())
            true_point = coords[int(b), 1:] * SPACING.as_array()
            rival_point = coords[int(rival_target), 1:] * SPACING.as_array()
            moved = (np.linalg.norm(true_point - predicted), np.linalg.norm(rival_point - predicted))
            still = (np.linalg.norm(true_point - here), np.linalg.norm(rival_point - here))
            velocity_true.append(moved[0] < moved[1])
            static_true.append(still[0] < still[1])
            margin_gain.append((still[0] - still[1]) - (moved[0] - moved[1]))

    return displacement, np.array(closer), np.array(velocity_true), np.array(static_true), np.array(margin_gain)


def report(displacement, closer, velocity_true, static_true, margin_gain):
    every = np.array(displacement["all"])
    band = np.array(displacement["outranked"])
    marginal = len(band) / max(len(every), 1)
    log.info("true candidate edges %d   outranked %d (%.4f)", len(every), len(band), marginal)
    for name in ("all", "mutual", "outranked"):
        values = np.array(displacement[name])
        log.info(
            "  %-10s n=%6d  mean %.3f  q50/q90 %s",
            name,
            len(values),
            values.mean(),
            np.percentile(values, [50, 90]).round(3),
        )
    large = np.percentile(every, LARGE_DISPLACEMENT_PERCENTILE)
    log.info(
        "  P(outranked | displacement > q%d=%.2f) = %.4f   marginal %.4f",
        LARGE_DISPLACEMENT_PERCENTILE,
        large,
        (band > large).sum() / max((every > large).sum(), 1),
        marginal,
    )

    decisive = closer[:, 0] != closer[:, 1]
    log.info(
        "  DISTANCE picks true %.4f of non-tied (n=%d, chance 0.5); true farther %.4f",
        (closer[decisive, 0] < closer[decisive, 1]).mean(),
        decisive.sum(),
        (closer[:, 0] > closer[:, 1]).mean(),
    )
    log.info(
        "  ORACLE velocity picks true %.4f vs static %.4f (n=%d, GT-anchored frame -- read the DELTA)",
        velocity_true.mean(),
        static_true.mean(),
        len(velocity_true),
    )
    log.info(
        "  margin gain um: mean %+.3f  median %+.3f  frac>0 %.4f",
        margin_gain.mean(),
        np.median(margin_gain),
        (margin_gain > 0).mean(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of cached candidate graphs (*.npz)")
    parser.add_argument("--train", type=Path, required=True, help="directory of matching train *.geff")
    parser.add_argument("--ruler", type=float, default=OFFICIAL_UM, help="node match radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report(*collect(args.cache, args.train, args.ruler))


if __name__ == "__main__":
    main()
