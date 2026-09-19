#!/usr/bin/env python3
"""Kaggle training entrypoint — association (transformer) + event solver per plan §7 & §8.

Real pilot: trains a simple MLP on candidate distance to distinguish true vs false edges,
using GT supervision on the fold's train datasets. This is a minimal real training
that runs on Kaggle in minutes, validates the pipeline, and produces a checkpoint
that can be used for inference (harmonic fusion).

All training must run on Kaggle; local runs require BIOHUB_ALLOW_LOCAL_TRAIN=1.
"""

from pathlib import Path
import json
import os
import sys


def _phase0_diag() -> None:
    """Write a diagnostic report BEFORE any fragile import.

    Runs first, so even if the kernel later crashes we can pull
    /kaggle/working/diag.json and see exactly what state Kaggle started in.
    """
    import importlib
    report: dict = {"phase": "diag",
                    "kaggle_working": Path("/kaggle/working").exists(),
                    "kaggle_input": Path("/kaggle/input").exists()}
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
    report["pkg_imports"] = {}
    for pkg in ["torch", "zarr", "polars", "scipy", "numpy"]:
        try:
            importlib.import_module(pkg)
            report["pkg_imports"][pkg] = "ok"
        except Exception as e:
            report["pkg_imports"][pkg] = f"FAIL {type(e).__name__}: {e}"
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
    """Ensure zarr/tracksdata/polars/scipy import.

    Tries the shared bootstrap shipped in biohub-bundle datasets, then falls
    back to a direct pip install. Any exception from the shared bootstrap is
    swallowed so the pip fallback always runs.
    """
    found = False
    for root in [Path(__file__).resolve().parents[2],
                 Path("/kaggle/working"),
                 Path("/kaggle/input/biohub-bundle-v1"),
                 Path("/kaggle/input/datasets/noisyislands/biohub-bundle-v1"),
                 Path("/kaggle/input/biohub-bundle-v2"),
                 Path("/kaggle/input/datasets/noisyislands/biohub-bundle-v2")]:
        for sub in ["kaggle/train_tabpfn", "kaggle_bundle/kaggle/train_tabpfn"]:
            bp = root / sub / "kaggle_bootstrap.py"
            if bp.exists():
                found = True
                try:
                    sys.path.insert(0, str(root / sub))
                    from kaggle_bootstrap import ensure_deps
                    ensure_deps(["zarr", "polars", "scipy"], strict=False, log=True)
                except Exception as e:
                    print(f"  [bootstrap] shared bootstrap failed ({type(e).__name__}: {e}); "
                          f"falling back to pip")
                break
        if found:
            break
    # Fallback: pip-install anything still missing (needs internet enabled)
    import subprocess
    import importlib
    for pkg in ["zarr", "polars", "scipy", "torch", "numpy"]:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"  [install] pip install {pkg} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False)
    print("  [bootstrap] done")

_bootstrap()
# --- End bootstrap ---

def assert_kaggle_env():
    is_kaggle = Path("/kaggle/working").exists()
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass
    if not is_kaggle:
        is_kaggle = Path("/kaggle/input").exists()
    if not is_kaggle and not os.environ.get("BIOHUB_ALLOW_LOCAL_TRAIN"):
        raise RuntimeError("Association training must run on Kaggle.")
    return is_kaggle, has_cuda

def discover_inputs():
    candidates = [
        Path("/kaggle/input/biohub-cell-tracking-during-development"),
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
        Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
        Path(os.environ.get("BIOHUB_DATA_DIR","")) if os.environ.get("BIOHUB_DATA_DIR") else None,
        Path("data/train_gt_all"),
        Path("data/train_gt"),
    ]
    for p in candidates:
        if p and p.exists():
            return p
    print("WARNING: No input mount found, using dummy for pilot")
    return Path("data/train_gt")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    # Candidate pool control
    ap.add_argument("--candidate-mode", choices=["gt_clean", "production_sim", "mixed"],
                   default="production_sim",
                   help="gt_clean: only clean GT-node candidates (old behaviour). "
                        "production_sim: inject synthetic duplicate/noise nodes to "
                        "mimic production detector output (hard negatives). "
                        "mixed: both pools, weighted. Default production_sim.")
    ap.add_argument("--synthetic-dup-rate", type=float, default=0.15,
                   help="Fraction of GT nodes duplicated as fake duplicates "
                        "(offset ≤3 µm, matching production detector duplicate behaviour).")
    ap.add_argument("--synthetic-noise-rate", type=float, default=0.10,
                   help="Fraction of frames that get synthetic noise nodes "
                        "(random position, no GT identity).")
    ap.add_argument("--neg-pos-ratio", type=float, default=1.5,
                   help="Negative-to-positive sample ratio for class balancing. "
                        "Lower = more negative-heavy (production-like). Default 1.5.")
    args = ap.parse_args()
    is_kaggle, has_cuda = assert_kaggle_env()
    print(f"Linker train kaggle={is_kaggle} cuda={has_cuda} fold={args.fold} window={args.window} epochs={args.epochs} lr={args.lr}")

    # Diagnostic: dump what's in the Kaggle input mount
    if is_kaggle:
        for probe in ["/kaggle/input", "/kaggle/input/biohub-cell-tracking-during-development",
                       "/kaggle/input/competitions", "/kaggle/input/datasets"]:
            probe_p = Path(probe)
            if probe_p.exists():
                entries = sorted(str(e) for e in probe_p.iterdir())
                print(f"  [diag] {probe}: {entries[:20]}")

    # Discover data
    d = discover_inputs()
    print(f"Data {d} exists={d.exists() if d else False}")
    # Load folds
    import json
    folds_path = Path("configs/folds.json")
    if not folds_path.exists():
        # Try alternative path when running from /kaggle/working
        for cand in [Path("/kaggle/working/configs/folds.json"), Path("kaggle_bundle/configs/folds.json"), Path("/kaggle/input/biohub-bundle-v1/kaggle_bundle/configs/folds.json")]:
            if cand.exists():
                folds_path = cand
                break
    print(f"Folds path {folds_path} exists={folds_path.exists()}")
    if folds_path.exists():
        folds = json.loads(folds_path.read_text())
        # Use 5-fold random, fallback to LOGO if needed
        try:
            train_datasets = folds["folds_5_random_stratified"][args.fold]["train"]
        except Exception:
            train_datasets = folds["folds_5_random_stratified"][0]["train"]
        print(f"Fold {args.fold} train {len(train_datasets)} datasets, e.g. {train_datasets[:3]}")
        # For pilot, limit to first 4 datasets to keep runtime low
        train_datasets = train_datasets[:4]
        print(f"Pilot limited to {len(train_datasets)} datasets for speed")
    else:
        train_datasets = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
        print(f"No folds, using default {train_datasets}")

    # Now build training data: for each dataset, generate candidates and labels
    import numpy as np
    import torch
    import torch.nn as nn
    from pathlib import Path as P

    VOXEL = np.array([1.625, 0.40625, 0.40625])
    MAX_DIST = 10.0
    TOPK = 3

    def load_gt_nodes_edges(gt_path: P):
        """Load GT nodes and edges directly via zarr (edge-only GEFFs have no nodes group)."""
        import zarr
        nodes = {}
        edges = set()
        try:
            zroot = zarr.open_group(str(gt_path), mode="r")
            if "nodes" in zroot:
                ids = np.asarray(zroot["nodes/ids"]).astype(int)
                tv = np.asarray(zroot["nodes/props/t/values"]).astype(int)
                zv = np.asarray(zroot["nodes/props/z/values"]).astype(int)
                yv = np.asarray(zroot["nodes/props/y/values"]).astype(int)
                xv = np.asarray(zroot["nodes/props/x/values"]).astype(int)
                nodes = {int(i): (int(t_), int(z_), int(y_), int(x_))
                         for i, t_, z_, y_, x_ in zip(ids, tv, zv, yv, xv)}
            if "edges" in zroot:
                es = np.asarray(zroot["edges/ids"]).astype(int)
                edges = {(int(s), int(d)) for s, d in es}
        except Exception as e:
            print(f"  [load_gt] {gt_path.name} failed: {e}")
        return nodes, edges, None

    # Find GT files (handle Kaggle competition train/ subdir)
    gt_files = []
    for ds in train_datasets:
        candidates = [
            d / f"{ds}.geff",
            d / "train" / f"{ds}.geff",
            d / f"{ds}.geff" / "zarr.json",
            Path(f"data/train_gt/{ds}.geff"),
            Path(f"data/train_gt_all/{ds}.geff"),
            Path(f"/kaggle/input/biohub-cell-tracking-during-development/train/{ds}.geff"),
            Path(f"/kaggle/input/competitions/biohub-cell-tracking-during-development/train/{ds}.geff"),
        ]
        found = False
        for cand in candidates:
            if cand and cand.exists():
                # If it's a directory with zarr.json, use cand
                if cand.is_dir() and (cand / "zarr.json").exists():
                    gt_files.append((ds, cand))
                    found = True
                    break
                elif cand.suffix == ".geff" or (cand / "zarr.json").exists():
                    # If cand is the zarr.json path, use its parent
                    actual = cand if cand.suffix == ".geff" else cand.parent
                    if actual.exists():
                        gt_files.append((ds, actual))
                        found = True
                        break
        if found:
            continue
        # Try glob in d and d/train
        for base in [d, d / "train"]:
            if not base.exists():
                continue
            for p in base.glob(f"{ds}.geff"):
                gt_files.append((ds, p))
                found = True
                break
            if found:
                break

    print(f"Found {len(gt_files)} GT files for pilot")
    # Filter to node-rich GEFFs (edge-only GEFFs yield 0 candidate rows)
    node_rich = [(ds, p) for ds, p in gt_files if load_gt_nodes_edges(p)[0]]
    skipped = [ds for ds, p in gt_files if not load_gt_nodes_edges(p)[0]]
    if skipped:
        print(f"Skipping {len(skipped)} edge-only GEFFs: {skipped[:5]}")
    gt_files = node_rich
    if not gt_files:
        print(f"WARNING: No node-rich GT files; using default 4-movie eval split")
        gt_files = [(ds, Path(f"data/train_gt/{ds}.geff")) for ds in
                    ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]]
    print(f"Using {len(gt_files)} node-rich GT files for candidate generation")
    if not gt_files:
        print(f"WARNING: No GT files found for {train_datasets[:2]}, listing {d} contents")
        print(list(d.iterdir())[:10] if d.exists() else "no dir")

    # --- Production-simulation synthetic node injection ---
    # The forensics show the production detector creates duplicates (~3 µm offset)
    # and noise nodes that create impostor candidates far closer than true pairs.
    # We inject synthetic nodes that mimic this behaviour so the MLP sees
    # realistic hard negatives during training.
    rng_global = np.random.default_rng(args.fold * 7919 + 42)

    def inject_synthetic_nodes(nodes: dict, all_frames, by_t, synthetic_dup_rate, synthetic_noise_rate):
        """Return extended node dict + extended GT edges.

        For each GT node, with probability synthetic_dup_rate add a duplicate
        at up to 3 µm offset (in voxel units). For a fraction of frames, add
        synthetic noise nodes at random positions within the bounding box.
        These synthetic nodes have node ids above the max GT node id.
        """
        extended = dict(nodes)
        extended_gt_edges = set()
        base_gt_edges = set()  # caller fills this in

        max_gt_id = max(nodes.keys()) if nodes else 0
        next_id = max_gt_id + 1
        rng = np.random.default_rng(next_id + args.fold)

        # Compute bounding box for noise node placement
        all_nodes_list = list(nodes.values())
        if all_nodes_list:
            z_min = min(p[1] for p in all_nodes_list)
            z_max = max(p[1] for p in all_nodes_list)
            y_min = min(p[2] for p in all_nodes_list)
            y_max = max(p[2] for p in all_nodes_list)
            x_min = min(p[3] for p in all_nodes_list)
            x_max = max(p[3] for p in all_nodes_list)
        else:
            z_min, z_max, y_min, y_max, x_min, x_max = 0, 0, 0, 0, 0, 0

        n_dups = 0
        n_noise = 0
        for nid, (t, z, y, x) in nodes.items():
            # Duplicate injection
            if rng.random() < synthetic_dup_rate:
                # offset in voxel units, up to 3 µm total
                max_vox_off = int(np.ceil(3.0 / 0.40625))  # ~7 voxels in x/y
                dz = int(rng.integers(-2, 3))
                dy = int(rng.integers(-max_vox_off, max_vox_off + 1))
                dx = int(rng.integers(-max_vox_off, max_vox_off + 1))
                dup_id = next_id
                next_id += 1
                extended[dup_id] = (t, z + dz, y + dy, x + dx)
                n_dups += 1
                # Duplicates share the same GT edges as the original node
                for (s, d) in base_gt_edges:
                    if s == nid:
                        extended_gt_edges.add((dup_id, d))
                    if d == nid:
                        extended_gt_edges.add((s, dup_id))

        # Noise nodes: add a few per frame
        for t in all_frames:
            if rng.random() < synthetic_noise_rate:
                z = int(rng.integers(max(0, z_min - 5), z_max + 5))
                y = int(rng.integers(max(0, y_min - 5), y_max + 5))
                x = int(rng.integers(max(0, x_min - 5), x_max + 5))
                nid_new = next_id
                next_id += 1
                extended[nid_new] = (t, z, y, x)
                n_noise += 1

        if n_dups or n_noise:
            print(f"    [synthetic] dups={n_dups} noise={n_noise} (dup_rate={synthetic_dup_rate}, noise_rate={synthetic_noise_rate})")

        return extended, extended_gt_edges

    # --- Link table building (plan §7: real association scoring, not placeholder) ---
    # Candidate pools:
    #   gt_clean: clean GT-node candidates only (original behaviour)
    #   production_sim: inject synthetic dup + noise nodes before candidate
    #     generation so the MLP sees realistic impostor hard negatives
    #   mixed: train on both pools with a weighted mixture
    candidate_mode = args.candidate_mode
    print(f"  Candidate mode: {candidate_mode}")

    class EdgeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(6, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    device = torch.device("cuda" if has_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device {device}")
    model = EdgeMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    # Training loop
    ckpt_dir = Path("/kaggle/working" if is_kaggle else "./kaggle_working") / f"linker_fold{args.fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_n = 0
        for ds, gt_path in gt_files:
            try:
                base_nodes, base_gt_edges, g = load_gt_nodes_edges(gt_path)
                from collections import defaultdict
                import scipy.spatial
                base_all_frames = sorted({p[0] for p in base_nodes.values()})
                base_by_t = defaultdict(dict)
                for nid, (t, z, y, x) in base_nodes.items():
                    base_by_t[t][nid] = (t, z, y, x)

                # Candidate pool construction
                if candidate_mode == "gt_clean":
                    nodes = base_nodes
                    gt_edges = base_gt_edges
                    by_t = base_by_t
                elif candidate_mode == "production_sim":
                    ext_nodes, ext_gt = inject_synthetic_nodes(
                        base_nodes, base_all_frames, base_by_t,
                        args.synthetic_dup_rate, args.synthetic_noise_rate)
                    nodes = ext_nodes
                    gt_edges = base_gt_edges | ext_gt
                    by_t = defaultdict(dict)
                    for nid, (t, z, y, x) in nodes.items():
                        by_t[t][nid] = (t, z, y, x)
                else:  # mixed
                    ext_nodes, ext_gt = inject_synthetic_nodes(
                        base_nodes, base_all_frames, base_by_t,
                        args.synthetic_dup_rate, args.synthetic_noise_rate)
                    nodes = base_nodes | ext_nodes
                    gt_edges = base_gt_edges | ext_gt
                    by_t = defaultdict(dict)
                    for nid, (t, z, y, x) in nodes.items():
                        by_t[t][nid] = (t, z, y, x)

                # Generate candidates (10 um radius, top-3 per target)
                cands = set()
                target_to_cands = defaultdict(list)  # tgt -> [(src, dist)]
                for t in sorted(by_t.keys()):
                    if t+1 not in by_t:
                        continue
                    t0 = by_t[t]
                    t1 = by_t[t+1]
                    ids0 = list(t0.keys())
                    ids1 = list(t1.keys())
                    coords0 = np.array([[t0[i][1], t0[i][2], t0[i][3]] for i in ids0], dtype=float) * VOXEL
                    coords1 = np.array([[t1[i][1], t1[i][2], t1[i][3]] for i in ids1], dtype=float) * VOXEL
                    tree = scipy.spatial.cKDTree(coords1)
                    pairs = tree.query_ball_point(coords0, r=MAX_DIST)
                    for i0, neigh in enumerate(pairs):
                        for i1 in neigh:
                            dist = float(np.linalg.norm(coords0[i0]-coords1[i1]))
                            s, tt = ids0[i0], ids1[i1]
                            cands.add((s,tt,dist))
                            target_to_cands[tt].append((s, dist))
                # Keep top-3 per target (matches production candidate recall)
                for tt, lst in target_to_cands.items():
                    lst.sort(key=lambda x: x[1])
                # Final candidate set: top-3 per target
                final_cands = set()
                for tt, lst in target_to_cands.items():
                    for s, dist in lst[:TOPK]:
                        final_cands.add((s, tt, dist))
                if not final_cands:
                    continue
                # Build training samples: for each candidate, label 1 if gt edge, else 0
                # Features: displacement_um/10, temporal_delta, is_first, is_last,
                #          motion_residual (dist - 4.0 um typical displacement),
                #          competing_link_count (how many other sources target the same node)
                node_t = {nid: by_t[nid][0] for nid in nodes if nid in by_t}
                all_frames = sorted(by_t.keys())
                t_min, t_max = all_frames[0], all_frames[-1] if all_frames else (0, 0)
                # Competing link count per target (from full candidate set, not just top-3)
                comp_count = {tt: len(lst) for tt, lst in target_to_cands.items()}

                # Hard-negative mining: for each target, sample non-GT sources within
                # MAX_DIST as impostor candidates (the forensics pattern — smooth
                # impostors closer than the true pair). This gives the MLP a
                # realistic positive/negative ratio instead of ~98% positive.
                # For each target, build the full candidate pool (all sources within
                # MAX_DIST, not just top-3) and label each as pos/neg.
                target_to_all = defaultdict(list)  # tgt -> [(src, dist)]
                for (s, tt, dist) in cands:
                    target_to_all[tt].append((s, dist))

                feats = []
                labels = []
                rng = np.random.default_rng(42)
                for tt, lst in target_to_all.items():
                    if not lst:
                        continue
                    lst.sort(key=lambda x: x[1])
                    n_pos_this = 0
                    for s, d in lst:
                        is_gt = 1 if (s, tt) in gt_edges else 0
                        ts_ = node_t.get(s, t_min)
                        tt_ = node_t.get(tt, t_max)
                        is_first = 1.0 if ts_ == t_min else 0.0
                        is_last = 1.0 if tt_ == t_max else 0.0
                        temporal_delta = float(tt_ - ts_)
                        motion_resid = max(0.0, d - 4.0)
                        n_comp = max(0, len(lst) - 1)  # other sources targeting same node
                        feats.append([d/10.0, temporal_delta, is_first, is_last, motion_resid/10.0, float(n_comp)])
                        labels.append(float(is_gt))
                        if is_gt:
                            n_pos_this += 1
                if not feats:
                    continue
                feats_arr = np.array(feats, dtype=np.float32)
                labels_arr = np.array(labels, dtype=np.float32)
                n_pos_all = int(labels_arr.sum())
                n_neg_all = int((1 - labels_arr).sum())
                if n_pos_all == 0:
                    continue
                # Balance: keep all positives, sample up to neg_pos_ratio * n_pos negatives randomly
                pos_idx = np.where(labels_arr == 1.0)[0]
                neg_idx = np.where(labels_arr == 0.0)[0]
                n_keep_neg = min(int(args.neg_pos_ratio * n_pos_all), len(neg_idx))
                if n_keep_neg < len(neg_idx):
                    keep_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
                else:
                    keep_neg = neg_idx
                sel = np.sort(np.concatenate([pos_idx, keep_neg]))
                feats_arr, labels_arr = feats_arr[sel], labels_arr[sel]

                feats_t = torch.tensor(feats_arr, dtype=torch.float32, device=device)
                labels_t = torch.tensor(labels_arr, dtype=torch.float32, device=device)
                logits = model(feats_t)
                loss = bce(logits, labels_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(feats_t)
                total_n += len(feats_t)
                print(f"  {ds} epoch {epoch+1} feats {len(feats_t)} pos {int(labels_t.sum().item())} neg {int((labels_t==0).sum().item())} loss {loss.item():.4f}")
            except Exception as e:
                print(f"  {ds} failed {e}")
                import traceback; traceback.print_exc()
                continue
        avg_loss = total_loss / max(total_n,1)
        print(f"Epoch {epoch+1}/{args.epochs} avg loss {avg_loss:.4f} total {total_n}")
        # Save checkpoint
        ckpt = {
            "epoch": epoch+1,
            "model_state": {k: v.cpu() for k,v in model.state_dict().items()},
            "optimizer_state": opt.state_dict(),
            "avg_loss": avg_loss,
            "fold": args.fold,
            "window": args.window,
            "train_datasets": train_datasets,
            "features": ["displacement_um/10", "temporal_delta", "is_first", "is_last",
                         "motion_residual/10", "competing_link_count"],
            "candidate": {"max_distance_um": MAX_DIST, "topk_targets": TOPK},
        }
        torch.save(ckpt, ckpt_dir / f"ckpt_epoch{epoch+1}.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  New best {best_loss:.4f} saved to best.pt")
        # Also save last
        torch.save(ckpt, ckpt_dir / "last.pt")

    prov = {
        "model": "EdgeMLP 6->16->16->1 on [dist/10, temporal_delta, is_first, is_last, motion_resid/10, comp_count], class-balanced neg:pos, pilot real training",
        "candidate": {"max_distance_um": MAX_DIST, "topk_targets": TOPK, "aim_recall": 0.995},
        "solver": {"type": "event_solver explicit b/d/c/h"},
        "candidate_mode": candidate_mode,
        "synthetic_dup_rate": args.synthetic_dup_rate,
        "synthetic_noise_rate": args.synthetic_noise_rate,
        "neg_pos_ratio": args.neg_pos_ratio,
        "window": args.window,
        "epochs": args.epochs,
        "lr": args.lr,
        "fold": args.fold,
        "train_datasets": train_datasets,
        "best_loss": best_loss,
        "note": "Pilot per plan §7. When candidate_mode=production_sim, synthetic duplicate + noise nodes are injected to mimic production detector impostor behaviour.",
    }
    (ckpt_dir / "prov.json").write_text(json.dumps(prov, indent=2))
    print(f"Wrote {ckpt_dir / 'prov.json'} and checkpoints to {ckpt_dir} best {best_loss:.4f}")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        err_msg = traceback.format_exc()
        print(err_msg)
        for err_path in [Path("/kaggle/working/error.txt"), Path("./error.txt")]:
            try:
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(err_msg)
                print(f"Wrote traceback to {err_path}")
                break
            except OSError:
                continue
        raise
