"""Measure how much two submitted trackers actually DISAGREE, to size an ensemble before paying for one.

An ensemble member only earns its seat if it decides edges the champion gets wrong. Both `submission.csv`
files carry the full node table and edge list on the same 4 test videos, so the disagreement is measurable
offline with no GPU and no submission slot: match nodes between the two by position within a frame, then
compare the edge sets over the matched nodes.

    .venv/Scripts/python kaggle/tracker_agreement.py --a runs/kaggle_out/celltrack-public-0947 \
        --b runs/kaggle_out/celltrack-public-v1329f

A high `shared` fraction means the two trackers are the same tracker wearing two hats and a union/vote buys
nothing ([[ensemble-diversity-has-a-quality-floor]]); the `a_only` / `b_only` counts are the upper bound on
what any combination rule could ever move.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

SPACING_UM = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 2.0


def read_submission(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    nodes = frame[frame["row_type"] == "node"]
    edges = frame[frame["row_type"] == "edge"]
    return nodes, edges


def match_within_video(left: pd.DataFrame, right: pd.DataFrame) -> dict[int, int]:
    """Nearest right-node per left-node within MATCH_UM, frame by frame, keyed by node_id."""
    pairs: dict[int, int] = {}
    for time, left_frame in left.groupby("t"):
        right_frame = right[right["t"] == time]
        if right_frame.empty:
            continue
        left_um = left_frame[["z", "y", "x"]].to_numpy() * SPACING_UM
        right_um = right_frame[["z", "y", "x"]].to_numpy() * SPACING_UM
        distance, nearest = cKDTree(right_um).query(left_um)
        within = distance <= MATCH_UM
        left_ids = left_frame["node_id"].to_numpy()[within]
        right_ids = right_frame["node_id"].to_numpy()[nearest[within]]
        pairs.update(dict(zip(left_ids, right_ids, strict=True)))
    return pairs


def compare_video(
    a_nodes: pd.DataFrame, a_edges: pd.DataFrame, b_nodes: pd.DataFrame, b_edges: pd.DataFrame
) -> dict[str, int]:
    pairs = match_within_video(a_nodes, b_nodes)
    a_in_b = {
        (pairs[source], pairs[target])
        for source, target in zip(a_edges["source_id"], a_edges["target_id"], strict=True)
        if source in pairs and target in pairs
    }
    b_set = set(zip(b_edges["source_id"], b_edges["target_id"], strict=True))
    unmatched = len(a_edges) - len(a_in_b)
    return {
        "a_nodes": len(a_nodes),
        "b_nodes": len(b_nodes),
        "matched_nodes": len(pairs),
        "shared_edges": len(a_in_b & b_set),
        "a_only": len(a_in_b - b_set) + unmatched,
        "b_only": len(b_set - a_in_b),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="a kernel output dir holding submission.csv")
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()

    a_nodes, a_edges = read_submission(args.a / "submission.csv")
    b_nodes, b_edges = read_submission(args.b / "submission.csv")

    totals: dict[str, int] = {}
    for dataset in sorted(set(a_nodes["dataset"]) & set(b_nodes["dataset"])):
        result = compare_video(
            a_nodes[a_nodes["dataset"] == dataset],
            a_edges[a_edges["dataset"] == dataset],
            b_nodes[b_nodes["dataset"] == dataset],
            b_edges[b_edges["dataset"] == dataset],
        )
        log.info("%s %s", dataset, json.dumps(result))
        for key, value in result.items():
            totals[key] = totals.get(key, 0) + value

    union = totals["shared_edges"] + totals["a_only"] + totals["b_only"]
    totals["shared_fraction_of_union"] = round(totals["shared_edges"] / union, 4)
    log.info("TOTAL %s", json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
