#!/usr/bin/env python3
"""Kaggle training entrypoint — XGBoost division-event classifier (plan §5A, Apache 2.0).

Learns a GBDT ranker on GT-only division features (13-col Kaggle subset).
Mirrors the local `scripts/mine_xgboost_context.py` pipeline so the exact
training context and model are reproducible on Kaggle without a local .venv.

The 13 features match `artifacts/tabpfn35/manifest.json` features list:
    daughter_separation_per_frame_um, is_first_frame, is_last_frame,
    local_density, n_frames_after, n_frames_before,
    parent_daughter_dist_absdiff_um, parent_daughter_dist_max_um,
    parent_daughter_dist_sum_um, parent_daughter_dist_um,
    parent_velocity_um, sister_dist_um, sister_symmetry

GT-only mining: positive rows come from GT division forks (parent with
>=2 children in the same frame); hard negatives come from single-continuation
parents paired with a same-frame non-child node. `local_density` and
`parent_velocity_um` are NaN in GT-only mode (no detector output available).

All training runs on Kaggle; local runs require BIOHUB_ALLOW_LOCAL_TRAIN=1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def _phase0_diag() -> None:
    """Write a diagnostic report BEFORE any fragile import.

    Runs first, so even if the kernel later crashes we can pull
    /kaggle/working/diag.json and see exactly what state Kaggle started in:
    which input mounts exist, which packages import, where GEFFs live.
    """
    import importlib
    report: dict = {
        "phase": "diag",
        "kaggle_working": Path("/kaggle/working").exists(),
        "kaggle_input": Path("/kaggle/input").exists(),
    }
    report["input_listing"] = []
    for p in [Path("/kaggle/input"),
              Path("/kaggle/input/biohub-cell-tracking-during-development"),
              Path("/kaggle/input/competitions")]:
        if p.exists():
            try:
                report["input_listing"].append(
                    {"path": str(p), "entries": sorted(os.listdir(p))[:50]})
            except OSError:
                pass
    # Deep-look for competition train dir (may be mounted with/without 'train' subdir)
    for comp in ["/kaggle/input/biohub-cell-tracking-during-development",
                 "/kaggle/input/competitions/biohub-cell-tracking-during-development"]:
        cp = Path(comp)
        if cp.exists():
            report["comp_contents"] = sorted(os.listdir(cp))[:50]
            train_p = cp / "train"
            if train_p.exists():
                entries = sorted(os.listdir(train_p))
                report["comp_train_count"] = len(entries)
                report["comp_train_sample"] = entries[:10]
                # Check a sample GEFF layout
                for e in entries[:3]:
                    eg = train_p / e
                    if eg.is_dir():
                        report.setdefault("geff_layout_samples", []).append(
                            {"name": e, "files": sorted(os.listdir(eg))[:10]})
    report["pkg_imports"] = {}
    for pkg in ["xgboost", "zarr", "polars", "numpy", "scipy", "torch", "tracksdata"]:
        try:
            importlib.import_module(pkg)
            report["pkg_imports"][pkg] = "ok"
        except Exception as e:
            report["pkg_imports"][pkg] = f"FAIL {type(e).__name__}: {e}"
    # Write progressively so a mid-run crash still leaves partial data
    for out in [Path("/kaggle/working/diag.json"), Path("./diag.json")]:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
            print(f"[diag] wrote {out}")
            break
        except OSError:
            continue
    print(json.dumps(report, indent=2))


_phase0_diag()


# --- Bootstrap missing packages (offline wheels preferred, falls back to pip) ---
def _bootstrap():
    """Ensure zarr/xgboost/polars/numpy import.

    Tries the shared bootstrap shipped in biohub-bundle datasets, then falls
    back to a direct pip install. Any exception from the shared bootstrap is
    swallowed so the pip fallback always runs.
    """
    found = False
    for root in [
        Path(__file__).resolve().parents[2],
        Path("/kaggle/working"),
        Path("/kaggle/input/biohub-bundle-v1"),
        Path("/kaggle/input/datasets/noisyislands/biohub-bundle-v1"),
        Path("/kaggle/input/biohub-bundle-v2"),
        Path("/kaggle/input/datasets/noisyislands/biohub-bundle-v2"),
    ]:
        for sub in ["kaggle/train_tabpfn", "kaggle_bundle/kaggle/train_tabpfn"]:
            bp = root / sub / "kaggle_bootstrap.py"
            if bp.exists():
                found = True
                try:
                    sys.path.insert(0, str(root / sub))
                    from kaggle_bootstrap import ensure_deps
                    ensure_deps(["zarr", "polars", "xgboost", "numpy"], strict=False, log=True)
                except Exception as e:
                    print(f"  [bootstrap] shared bootstrap failed ({type(e).__name__}: {e}); "
                          f"falling back to pip")
                break
        if found:
            break
    # Fallback: pip-install anything still missing (needs internet enabled)
    import subprocess
    import importlib
    for pkg in ["xgboost", "zarr", "polars", "numpy"]:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"  [install] pip install {pkg} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False)
    print("  [bootstrap] done")


_bootstrap()
# --- End bootstrap ---


def assert_kaggle_env() -> tuple[bool, bool]:
    is_kaggle = Path("/kaggle/working").exists()
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    # On Kaggle script kernels /kaggle/working always exists; also accept the
    # competition input mount as a valid Kaggle signal.
    if not is_kaggle:
        is_kaggle = Path("/kaggle/input").exists()
    if not is_kaggle and not os.environ.get("BIOHUB_ALLOW_LOCAL_TRAIN"):
        raise RuntimeError(
            "XGBoost division training must run on Kaggle. "
            "Set BIOHUB_ALLOW_LOCAL_TRAIN=1 to run locally for a smoke test."
        )
    return is_kaggle, has_cuda


def discover_geffs() -> list[Path]:
    """Return the list of GEFF files to mine, preferring the Kaggle input mount.

    Note: GT GEFFs are sparse - most contain only `edges/` (no node
    coordinates). Only the small `train_gt` subset (4 movies) and a few
    others carry `nodes/` data. Division feature mining requires node-rich
    GEFFs; edge-only GEFFs contribute 0 rows and are silently skipped by
    `load_geff` (which returns empty nodes).
    """
    # 1. Kaggle input mount (full competition train set, node-rich).
    #    Kaggle mounts competition data at /kaggle/input/<comp-slug>/ with
    #    an optional /competitions/ prefix; the train GEFFs are zarr dirs
    #    named <movie>.geff each containing zarr.json.
    candidates = [
        Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
        Path("/kaggle/input/biohub-cell-tracking-during-development/train/train"),
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train/train"),
        Path(os.environ.get("BIOHUB_DATA_DIR", "")) if os.environ.get("BIOHUB_DATA_DIR") else None,
    ]
    for p in candidates:
        if not (p and p.exists()):
            continue
        geffs: list[Path] = []
        # .geff suffix files
        geffs += sorted(p.glob("*.geff"))
        # zarr directories (zarr.json present)
        geffs += [g for g in sorted(p.iterdir()) if g.is_dir() and (g / "zarr.json").exists()]
        if geffs:
            print(f"Found {len(geffs)} GEFFs under {p}")
            return geffs

    # 2. Local fallback: union of all known GEFF dirs (train_gt first, then
    #    any division node-rich download, then the edge-only full set).
    seen: set[str] = set()
    out: list[Path] = []
    for d in [Path("data/train_gt"), Path("data/train_gt_div"), Path("data/train_gt_all")]:
        if not d.exists():
            continue
        for g in sorted(d.glob("*.geff")):
            key = str(g.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(g)
    if not out:
        raise FileNotFoundError(
            "No .geff files found. On Kaggle the competition train/ dir must be "
            "mounted; locally pass --geff-dirs or set BIOHUB_DATA_DIR."
        )
    return out


# 13-col Kaggle subset (must match artifacts/tabpfn35/manifest.json features)
CONTEXT_FEATURES: tuple[str, ...] = (
    "daughter_separation_per_frame_um",
    "is_first_frame",
    "is_last_frame",
    "local_density",
    "n_frames_after",
    "n_frames_before",
    "parent_daughter_dist_absdiff_um",
    "parent_daughter_dist_max_um",
    "parent_daughter_dist_sum_um",
    "parent_daughter_dist_um",
    "parent_velocity_um",
    "sister_dist_um",
    "sister_symmetry",
)
CTX_IDX = {c: j for j, c in enumerate(CONTEXT_FEATURES)}
VOXEL = (1.625, 0.40625, 0.40625)


def load_geff(geff_path: Path) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, list[int]], int]:
    """Load GT nodes/edges from a GEFF; returns ({}, {}, 0) if node data missing."""
    import numpy as np
    import zarr

    try:
        zroot = zarr.open_group(str(geff_path), mode="r")
        gt_ids = np.asarray(zroot["nodes/ids"]).astype(np.int64)
        gt_t = np.asarray(zroot["nodes/props/t/values"]).astype(np.int64)
        gt_z = np.asarray(zroot["nodes/props/z/values"]).astype(np.int64)
        gt_y = np.asarray(zroot["nodes/props/y/values"]).astype(np.int64)
        gt_x = np.asarray(zroot["nodes/props/x/values"]).astype(np.int64)
    except (KeyError, FileNotFoundError, ValueError, OSError):
        return {}, {}, 0

    nodes = {
        int(i): (int(t_), int(z_), int(y_), int(x_))
        for i, t_, z_, y_, x_ in zip(gt_ids, gt_t, gt_z, gt_y, gt_x)
    }
    out: dict[int, list[int]] = defaultdict(list)
    es = np.asarray(zroot["edges/ids"]).astype(int)
    for s, d in es:
        out[int(s)].append(int(d))
    n_frames = max((v[0] for v in nodes.values()), default=0) + 1
    return nodes, dict(out), n_frames


def dist_um(nodes, a: int, b: int) -> float:
    pa = [nodes[a][1], nodes[a][2], nodes[a][3]]
    pb = [nodes[b][1], nodes[b][2], nodes[b][3]]
    return math.sqrt(sum((pa[i] - pb[i]) ** 2 * VOXEL[i] ** 2 for i in range(3)))


def make_row(parent: int, da: int, db: int, t_split: int, n_frames: int, nodes) -> list[float]:
    """Build a 13-col feature row for a division fork (NaN for unavailable features)."""
    d_pa = dist_um(nodes, parent, da)
    d_pb = dist_um(nodes, parent, db)
    d_sis = dist_um(nodes, da, db)
    sym = 1.0 - abs(d_pa - d_pb) / (d_pa + d_pb + 1e-9)
    t_da = nodes[da][0]
    t_db = nodes[db][0]
    sep_pf = d_sis / max(1, max(t_da, t_db) - t_split) if max(t_da, t_db) > t_split else float("nan")
    row = [float("nan")] * len(CONTEXT_FEATURES)
    row[CTX_IDX["parent_daughter_dist_um"]] = min(d_pa, d_pb)
    row[CTX_IDX["parent_daughter_dist_max_um"]] = max(d_pa, d_pb)
    row[CTX_IDX["parent_daughter_dist_sum_um"]] = d_pa + d_pb
    row[CTX_IDX["parent_daughter_dist_absdiff_um"]] = abs(d_pa - d_pb)
    row[CTX_IDX["sister_dist_um"]] = d_sis
    row[CTX_IDX["sister_symmetry"]] = sym
    row[CTX_IDX["daughter_separation_per_frame_um"]] = sep_pf
    row[CTX_IDX["n_frames_before"]] = float(t_split)
    row[CTX_IDX["n_frames_after"]] = float(max(0, n_frames - t_split - 1))
    row[CTX_IDX["is_first_frame"]] = 1.0 if t_split == 0 else 0.0
    row[CTX_IDX["is_last_frame"]] = 1.0 if t_split == n_frames - 1 else 0.0
    # local_density, parent_velocity_um: NaN in GT-only mode
    return row


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-per-fork", type=int, default=40,
                    help="hard negatives sampled per single-continuation parent")
    ap.add_argument("--max-neg-total", type=int, default=10000,
                    help="cap total negatives in the context")
    ap.add_argument("--out", default=None,
                    help="output dir (default: /kaggle/working/xgboost35_full or ./kaggle_working/xgboost35_full)")
    ap.add_argument("--geff-dirs", type=str, default=None,
                    help="comma-separated extra GEFF dirs to include")
    args = ap.parse_args()

    is_kaggle, has_cuda = assert_kaggle_env()
    print(f"XGBoost division train: kaggle={is_kaggle} cuda={has_cuda} "
          f"fold={args.fold} rounds={args.rounds} depth={args.max_depth} lr={args.lr}")

    # Diagnostic: dump what's in the Kaggle input mount
    if is_kaggle:
        for probe in ["/kaggle/input", "/kaggle/input/biohub-cell-tracking-during-development",
                       "/kaggle/input/competitions", "/kaggle/input/datasets"]:
            probe_p = Path(probe)
            if probe_p.exists():
                entries = sorted(str(e) for e in probe_p.iterdir())
                print(f"  [diag] {probe}: {entries[:20]}")
    geffs = discover_geffs()
    if args.geff_dirs:
        extra = [Path(p) for p in args.geff_dirs.split(",") if p.strip()]
        for d in extra:
            if d.exists():
                more = sorted(d.glob("*.geff")) + [g for g in sorted(d.iterdir()) if g.is_dir() and (g / "zarr.json").exists()]
                geffs += more
    print(f"Mining {len(geffs)} GEFFs (only node-rich GEFFs contribute division rows)")
    # Count node-rich
    node_rich = [g for g in geffs if load_geff(g)[0]]
    print(f"Node-rich: {len(node_rich)} / {len(geffs)} (edge-only skipped)")

    # Collect positives + hard-negative pool
    import numpy as np
    rng = np.random.default_rng(args.seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    n_forks = 0
    n_geffs_with_nodes = 0

    for geff in geffs:
        nodes, out, n_frames = load_geff(geff)
        if not nodes:
            continue
        n_geffs_with_nodes += 1
        by_t: dict[int, list[int]] = defaultdict(list)
        for nid, (t, *_rest) in nodes.items():
            by_t[t].append(nid)

        # Positives: all GT division forks
        for parent, children in out.items():
            if len(children) < 2:
                continue
            c1, c2 = children[0], children[1]
            t_split = nodes[parent][0]
            rows.append(make_row(parent, c1, c2, t_split, n_frames, nodes))
            labels.append(1)
            n_forks += 1

        # Hard negatives: single-continuation parents paired with a same-frame non-child
        for parent, children in out.items():
            if len(children) != 1:
                continue
            t_split = nodes[parent][0]
            c1 = children[0]
            others = [n for n in by_t.get(t_split, []) if n != parent]
            if not others:
                continue
            k = min(args.neg_per_fork, len(others))
            for oc in rng.choice(others, size=k, replace=False):
                rows.append(make_row(parent, c1, int(oc), t_split, n_frames, nodes))
                labels.append(0)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int8)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"Context: {len(y)} rows ({n_pos} pos / {n_neg} neg) from {n_geffs_with_nodes} node-rich GEFFs, "
          f"{n_forks} forks")

    if n_pos == 0:
        raise SystemExit("No positive division forks found in the GEFF input; check --geff-dirs / BIOHUB_DATA_DIR")

    # Cap negatives
    if n_neg > args.max_neg_total:
        neg_idx = np.where(y == 0)[0]
        keep = np.sort(np.concatenate([np.where(y == 1)[0],
                                       rng.choice(neg_idx, size=args.max_neg_total, replace=False)]))
        X, y = X[keep], y[keep]
        n_neg = int((y == 0).sum())
        print(f"Capped neg -> {n_neg}; total rows={len(y)}")

    import xgboost as xgb
    dtrain = xgb.DMatrix(X, label=y)
    pos_w = float(n_neg / max(1, n_pos))
    params = {
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "scale_pos_weight": pos_w,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": args.seed,
        "verbosity": 0,
    }
    t0 = time.time()
    booster = xgb.train(params, dtrain, num_boost_round=args.rounds, verbose_eval=False)
    p1 = booster.predict(xgb.DMatrix(X))
    print(f"Trained {args.rounds} rounds in {time.time()-t0:.1f}s | "
          f"scale_pos_weight={pos_w:.1f} | in-sample P(1) pos={p1[y==1].mean():.4f} neg={p1[y==0].mean():.4f} "
          f"acc={float(((p1>=0.5)==(y==1)).mean()):.4f}")

    # Save artifacts
    out_dir = Path(args.out) if args.out else Path("/kaggle/working" if is_kaggle else "./kaggle_working") / "xgboost35_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "xgboost.json"
    booster.save_model(str(model_path))
    manifest = {
        "model": "xgboost.json",
        "model_sha256": _sha256(model_path),
        "model_license": "Apache-2.0",
        "features": list(CONTEXT_FEATURES),
        "class_order": [0, 1],
        "context_rows": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "seed": args.seed,
        "variant": "division",
        "num_boost_round": args.rounds,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "scale_pos_weight": pos_w,
        "xgboost_params": params,
        "n_forks": n_forks,
        "geffs_with_nodes": n_geffs_with_nodes,
        "geffs_total": len(geffs),
        "fold": args.fold,
        "train_time_s": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "context.json").write_text(json.dumps({"X": X.tolist(), "y": y.tolist()}))
    print(f"Saved model ({manifest['model_sha256'][:16]}…) + manifest + context to {out_dir}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        err_msg = traceback.format_exc()
        print(err_msg)
        # Write the traceback to /kaggle/working so it can be pulled later
        for err_path in [Path("/kaggle/working/error.txt"), Path("./error.txt")]:
            try:
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(err_msg)
                print(f"Wrote traceback to {err_path}")
                break
            except OSError:
                continue
        raise
