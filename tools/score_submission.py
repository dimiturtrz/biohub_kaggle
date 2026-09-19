"""Score a Kaggle ``submission.csv`` against train GT with the frontier evaluator.

The visible test movies also ship with GT under ``train/``, so a kernel's output can be scored locally:

    python tools/score_submission.py --frontier-repo <repo> --data-dir <train> <submission.csv>...
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def _graph(rows: pl.DataFrame) -> object:
    import tracksdata as td  # noqa: PLC0415

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)
    nodes = rows.filter(pl.col("row_type") == "node")
    ids = {
        nid: graph.add_node({"t": int(t), "z": float(z), "y": float(y), "x": float(x)})
        for nid, t, z, y, x in nodes.select("node_id", "t", "z", "y", "x").iter_rows()
    }
    for source, target in rows.filter(pl.col("row_type") == "edge").select("source_id", "target_id").iter_rows():
        graph.add_edge(ids[source], ids[target], {})
    return graph


def score(submission: Path, data_dir: Path) -> dict:
    from biohub_tracking.io import open_dataset  # noqa: PLC0415
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics, summarise  # noqa: PLC0415
    from evaluate import _read_estimated_n_total  # noqa: PLC0415

    table = pl.read_csv(submission)
    rows = []
    for (dataset,), part in table.group_by("dataset", maintain_order=True):
        ds = open_dataset(data_dir / dataset, require_tracks=True)
        pred = _graph(part)
        er = evaluate(pred, ds.tracks, scale=ds.scale)
        recall = node_recall(pred, ds.tracks)
        row = per_sample_metrics(er, _read_estimated_n_total(data_dir / f"{dataset}.geff"), recall)
        log.info("%s %s", dataset, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})
        rows.append(row)
    return summarise(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("submissions", type=Path, nargs="+")
    args = parser.parse_args()
    sys.path[:0] = [str(args.frontier_repo / "src"), str(args.frontier_repo / "scripts")]
    for submission in args.submissions:
        log.info("%s %s", submission, json.dumps(score(submission, args.data_dir)))


if __name__ == "__main__":
    main()
