"""Cache frontier ``predict_video`` output once, then sweep linking (ILP) settings on CPU.

``cache`` runs the network (GPU or CPU) with a permissive edge threshold and stores per-movie candidate graphs;
``sweep`` re-filters edges, solves the ILP for every grid point and scores each with the frontier evaluator,
so post-processing is tuned without re-running inference.

    python tools/frontier_ilp_sweep.py cache --frontier-repo R --data-dir D --splits S --weights W --out C
    python tools/frontier_ilp_sweep.py sweep --frontier-repo R --data-dir D --cache C --grid grid.json
"""

import argparse
import contextlib
import itertools
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CACHE_EDGE_THRESHOLD = 0.01


def _import_predict(repo: Path) -> object:
    sys.path[:0] = [str(repo / "src"), str(repo / "scripts")]
    import predict_unet_transformer  # noqa: PLC0415

    return predict_unet_transformer


def cache(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415

    predict = _import_predict(args.frontier_repo)
    names = json.loads(args.splits.read_text())[0]["test"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, window_size, downsample = predict.load_model(args.weights, device)
    cfg = predict.PredictConfig(det_threshold=args.det_threshold, threshold=CACHE_EDGE_THRESHOLD, use_ilp=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in (n for n in names if not (args.out / f"{n}.npz").exists()):
        coords, edges = predict.predict_video(
            model, args.data_dir / name, device, cfg=cfg, window_size=window_size, downsample=downsample
        )
        np.savez(args.out / f"{name}.npz", coords=coords, edges=np.asarray(edges, dtype=np.float64).reshape(-1, 4))
        log.info("cached %s: %d nodes %d edges", name, len(coords), len(edges))


def _solve_point(repo: Path, cache_dir: Path, out_dir: Path, point: dict) -> dict:
    predict = _import_predict(repo)
    import tracksdata as td  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    for npz in sorted(cache_dir.glob("*.npz")):
        payload = np.load(npz)
        edges = payload["edges"][payload["edges"][:, 2] > point["edge_threshold"]]
        graph = predict.build_graph(payload["coords"], [(int(s), int(t), p, d) for s, t, p, d in edges])
        if graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
                appearance_weight=point["appearance"],
                disappearance_weight=point["disappearance"],
                division_weight=point["division"],
            )
            with predict.suppress_output():
                graph = solver.solve(graph)
        predict.save_graph(graph, out_dir / f"{npz.stem}.geff")
    run = {
        "username": "local",
        "method": out_dir.parent.name,
        "split": out_dir.name,
        "dir": out_dir,
        "geffs": sorted(out_dir.glob("*.geff")),
    }
    return {**point, **predict.summarise(predict.evaluate_run(run))}


def sweep(args: argparse.Namespace) -> None:
    os.environ["BIOHUB_DATA_DIR"] = str(args.data_dir)
    grid = json.loads(args.grid.read_text())
    points = [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]
    out_root = args.cache.parent / f"{args.cache.name}_sweep"
    dirs = [out_root / f"p{i:03d}" / "split_0" for i in range(len(points))]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(
            pool.map(_solve_point, [args.frontier_repo] * len(points), [args.cache] * len(points), dirs, points)
        )
    results.sort(key=lambda r: -r["score"])
    (out_root / "results.json").write_text(json.dumps(results, indent=1, default=float))
    for row in results:
        log.info(
            "score=%.4f adjJ=%.4f divJ=%.4f %s",
            row["score"],
            row["adj_edge_jaccard"],
            row["division_jaccard"],
            {k: row[k] for k in grid},
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    cache_parser = sub.add_parser("cache")
    cache_parser.add_argument("--splits", type=Path, required=True)
    cache_parser.add_argument("--weights", type=Path, required=True)
    cache_parser.add_argument("--out", type=Path, required=True)
    cache_parser.add_argument("--det-threshold", type=float, default=0.99)
    cache_parser.add_argument("--cpu", action="store_true")
    sweep_parser = sub.add_parser("sweep")
    sweep_parser.add_argument("--cache", type=Path, required=True)
    sweep_parser.add_argument("--grid", type=Path, required=True)
    sweep_parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with contextlib.chdir(args.frontier_repo):
        {"cache": cache, "sweep": sweep}[args.command](args)


if __name__ == "__main__":
    main()
