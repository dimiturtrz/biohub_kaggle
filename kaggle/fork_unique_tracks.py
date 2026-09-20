"""Split the nodes one fork holds and the other does not into MISSED CELLS vs LOCALIZATION BLIPS.

Two submissions each carry thousands of nodes the other has none for, and counting them says
nothing about which kind of difference they are. A node whose whole track is invisible to the other
fork is a cell that fork's detector never found -- a recall difference, and the only part a union
could ever recover. A node sitting inside a track both forks otherwise share is the same cell placed
a couple of microns apart, which the official 7 um matcher does not even notice.

Assign every node to its own fork's track (connected component of that fork's edges), match nodes to
the other fork frame by frame, and report each track's unmatched fraction:

    .venv/Scripts/python kaggle/fork_unique_tracks.py --a runs/kaggle_out/celltrack-public-0947 \
        --b runs/kaggle_out/celltrack-public-v1329f

The two populations separate under the match tolerance: widen it and the blips collapse while the
whole-invisible tracks barely move, because a recall difference is tolerance-insensitive and a
localization difference is not.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

SPACING_UM = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 2.0
WHOLE = 0.999
MAJORITY = 0.5
LONG_TRACK_FRAMES = 5

BANDS = (
    (WHOLE, 1.01, "whole track invisible to the other fork"),
    (MAJORITY, WHOLE, "majority-unmatched"),
    (0.0, MAJORITY, "minority / blips inside a shared track"),
)


def track_of(node_ids, edges):
    """Group node ids into connected components of the edge list (one component = one track)."""
    parent = {n: n for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for source, target in edges:
        if source in parent and target in parent:
            root_s, root_t = find(source), find(target)
            if root_s != root_t:
                parent[root_s] = root_t

    groups = defaultdict(list)
    for n in node_ids:
        groups[find(n)].append(n)
    return groups


def unmatched_mask(a_nodes, b_nodes, match_um):
    """True where a b-node has no a-node within `match_um` in the same frame."""
    out = np.ones(len(b_nodes), dtype=bool)
    frames = b_nodes.t.to_numpy()
    points = b_nodes[["z", "y", "x"]].to_numpy() * SPACING_UM
    for t in np.unique(frames):
        same_frame = a_nodes[a_nodes.t == t]
        if same_frame.empty:
            continue
        sel = frames == t
        tree = cKDTree(same_frame[["z", "y", "x"]].to_numpy() * SPACING_UM)
        distance, _ = tree.query(points[sel], k=1)
        out[np.flatnonzero(sel)] = distance > match_um
    return out


def touched_tracks(a, b, match_um):
    """One row per b-track holding at least one node unmatched in a: (dataset, length, unmatched)."""
    rows = []
    for dataset in sorted(b.dataset.unique()):
        a_nodes = a[(a.dataset == dataset) & (a.row_type == "node")]
        b_nodes = b[(b.dataset == dataset) & (b.row_type == "node")]
        b_edges = b[(b.dataset == dataset) & (b.row_type == "edge")]
        ids = b_nodes.node_id.to_numpy()
        unmatched = dict(zip(ids, unmatched_mask(a_nodes, b_nodes, match_um), strict=True))
        edges = b_edges[["source_id", "target_id"]].to_numpy()
        for members in track_of(ids, edges).values():
            n_unmatched = sum(unmatched[m] for m in members)
            if n_unmatched:
                rows.append((dataset, len(members), n_unmatched))
    return pd.DataFrame(rows, columns=["dataset", "length", "n_unmatched"])


def report(a_path, b_path, match_um, label):
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    tracks = touched_tracks(a, b, match_um)
    tracks["fraction"] = tracks.n_unmatched / tracks.length
    total = int(tracks.n_unmatched.sum())
    log.info("%s: %d unmatched nodes across %d touched tracks", label, total, len(tracks))
    for low, high, name in BANDS:
        band = tracks[(tracks.fraction >= low) & (tracks.fraction < high)]
        nodes = int(band.n_unmatched.sum())
        log.info(
            "  %-42s tracks %5d  nodes %6d  (%.1f%%)  median length %s",
            name,
            len(band),
            nodes,
            100.0 * nodes / max(total, 1),
            band.length.median() if len(band) else 0,
        )
    whole = tracks[tracks.fraction >= WHOLE]
    if len(whole):
        long_enough = whole[whole.length >= LONG_TRACK_FRAMES]
        log.info(
            "  whole-invisible track length: median %.0f, p90 %.0f, max %d | >=%d frames: %d tracks / %d nodes",
            whole.length.median(),
            whole.length.quantile(0.9),
            int(whole.length.max()),
            LONG_TRACK_FRAMES,
            len(long_enough),
            int(long_enough.length.sum()),
        )
    return tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="directory holding a submission.csv")
    parser.add_argument("--b", type=Path, required=True, help="the other submission's directory")
    parser.add_argument("--match-um", type=float, default=MATCH_UM)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a, b = args.a / "submission.csv", args.b / "submission.csv"
    report(a, b, args.match_um, f"{args.b.name}-only (vs {args.a.name})")
    report(b, a, args.match_um, f"{args.a.name}-only (vs {args.b.name})")


if __name__ == "__main__":
    main()
