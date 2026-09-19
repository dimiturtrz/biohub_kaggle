"""Merge a second tracker's edges into a champion submission, constraint-safe, as an omission repair.

E26/E28 measured our dominant loss as OMISSION (1460 fragmented GT edges vs 74 mislinks), and E32 measured
two public forks 0.008 apart on the board disagreeing about one edge in six. So the mechanism-matched
combination rule is a UNION in the champion's frame: keep every champion edge, and admit a donor edge only
where both endpoints match a champion node and the admission breaks no tracking constraint (a node keeps at
most one parent and at most two children, and an edge stays forward in time by one frame).

    .venv/Scripts/python kaggle/merge_submissions.py --base runs/kaggle_out/celltrack-public-0947 \
        --donor runs/kaggle_out/celltrack-public-v1329f --out runs/merged/0947_union_v1329f.csv
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from tracker_agreement import match_within_video, read_submission

log = logging.getLogger(__name__)

MAX_CHILDREN = 2


def admissible_donor_edges(
    base_nodes: pd.DataFrame, base_edges: pd.DataFrame, donor_nodes: pd.DataFrame, donor_edges: pd.DataFrame
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Donor edges re-expressed on base node ids that the base graph can legally absorb."""
    to_donor = match_within_video(base_nodes, donor_nodes)
    to_base = {donor: base for base, donor in to_donor.items()}
    time_of = dict(zip(base_nodes["node_id"], base_nodes["t"], strict=True))

    parents = dict(zip(base_edges["target_id"], base_edges["source_id"], strict=True))
    children: dict[int, int] = defaultdict(int)
    for source in base_edges["source_id"]:
        children[source] += 1
    existing = set(zip(base_edges["source_id"], base_edges["target_id"], strict=True))

    admitted: list[tuple[int, int]] = []
    rejected = defaultdict(int)
    for donor_source, donor_target in zip(donor_edges["source_id"], donor_edges["target_id"], strict=True):
        if donor_source not in to_base or donor_target not in to_base:
            rejected["unmatched_endpoint"] += 1
            continue
        source, target = to_base[donor_source], to_base[donor_target]
        if (source, target) in existing:
            rejected["already_present"] += 1
        elif target in parents:
            rejected["target_has_parent"] += 1
        elif children[source] >= MAX_CHILDREN:
            rejected["source_at_child_cap"] += 1
        elif time_of[target] - time_of[source] != 1:
            rejected["not_consecutive"] += 1
        else:
            admitted.append((source, target))
            parents[target] = source
            children[source] += 1
    return admitted, dict(rejected)


def merge(base_dir: Path, donor_dir: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    base_nodes, base_edges = read_submission(base_dir / "submission.csv")
    donor_nodes, donor_edges = read_submission(donor_dir / "submission.csv")

    added_rows: list[pd.DataFrame] = []
    report: dict[str, dict] = {}
    for dataset in sorted(set(base_nodes["dataset"]) & set(donor_nodes["dataset"])):
        admitted, rejected = admissible_donor_edges(
            base_nodes[base_nodes["dataset"] == dataset],
            base_edges[base_edges["dataset"] == dataset],
            donor_nodes[donor_nodes["dataset"] == dataset],
            donor_edges[donor_edges["dataset"] == dataset],
        )
        report[dataset] = {"admitted": len(admitted), **rejected}
        log.info("%s %s", dataset, json.dumps(report[dataset]))
        if admitted:
            frame = pd.DataFrame(admitted, columns=["source_id", "target_id"])
            frame["dataset"] = dataset
            frame["row_type"] = "edge"
            for column in ("node_id", "t", "z", "y", "x"):
                frame[column] = -1
            added_rows.append(frame)

    original = pd.read_csv(base_dir / "submission.csv")
    merged = pd.concat([original, *added_rows], ignore_index=True) if added_rows else original
    merged["id"] = range(len(merged))
    return merged[original.columns], report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="champion kernel output dir")
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    merged, report = merge(args.base, args.donor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    total = sum(entry["admitted"] for entry in report.values())
    log.info("wrote %s (%d rows, +%d donor edges)", args.out, len(merged), total)


if __name__ == "__main__":
    main()
