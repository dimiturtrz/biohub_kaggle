"""Run the organizers' evaluator on the shared case file; dump expected counts."""

import json
from pathlib import Path

import polars as pl
import tracksdata as td
from tracking_cellmot.metrics import evaluate

ASSETS = Path(__file__).parents[2] / "tests" / "assets"


def build(spec):
    graph = td.graph.InMemoryGraph()
    graph.add_node_attr_key("z", pl.Float64, 0.0)
    graph.add_node_attr_key("y", pl.Float64, 0.0)
    graph.add_node_attr_key("x", pl.Float64, 0.0)
    ids = {}
    for uid, t, z, y, x in spec["nodes"]:
        ids[uid] = graph.add_node({"t": int(t), "z": float(z), "y": float(y), "x": float(x)})
    for s, t in spec["edges"]:
        graph.add_edge(ids[s], ids[t], {})
    return graph, ids


def main():
    payload = json.loads((ASSETS / "metric_cases.json").read_text())
    scale = tuple(payload["spacing"])
    out = {}
    for case in payload["cases"]:
        gt, _ = build(case["gt"])
        pred, _ = build(case["pred"])
        r = evaluate(pred, gt, scale=scale, max_distance=payload["max_distance"])
        out[case["name"]] = dict(
            edge_tp=r.edge_tp,
            edge_fp=r.edge_fp,
            edge_fn=r.edge_fn,
            division_tp=r.division_tp,
            division_fp=r.division_fp,
            division_fn=r.division_fn,
            num_pred_nodes=r.num_pred_nodes,
        )
        print(case["name"], out[case["name"]])
    (ASSETS / "metric_expected.json").write_text(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
