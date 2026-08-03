"""Tracked ourstunet training — their joint recipe driven through OUR harness.

Imports the vendored per-epoch functions (`train_epoch`/`evaluate`) + model/data builders from
`train_unet_transformer` (unedited) and drives OUR `TrackedRun` loop: per-epoch MLflow metrics, patience
early-stop, save-best, human log — with bf16 autocast on the forward (no GradScaler; bf16 keeps fp32's
exponent range so there's no fp16 overflow→NaN, and mixed precision keeps fp32 master weights, so accuracy
is unchanged). Faithful to their recipe (defaults from RunConfig: lr 1e-3, det_loss_weight 10, downsample
(1,4,4), window 2, brightness+flip aug), plus early-stop + tracking.

Runs in `.venv-repro` (has torch+cu128, tracksdata, tracking_cellmot, mlflow):

    .venv-repro/Scripts/python scratchpad/tunet_train_tracked.py \
        --data-dir D:/data/volumetric/microscopy/raw/biohub_cell_tracking/train \
        --splits scratchpad/fold0_splits.json --name ourstunet-clean \
        --set optim.epochs=50 optim.patience=8 eval.subset=8

Not a gated module — the reusable core (TrackedRun/EarlyStop/RunConfig/Tracker/Obs) is committed + tested;
this is the thin vendored wiring. bf16 wants num_workers>0 so collate runs in workers, outside the
autocast scope (their train_epoch owns the loader loop, so we can't scope autocast to the forward alone).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "external/kaggle-cell-tracking-competition/scripts"
# running as a script puts scratchpad/ on sys.path[0], not the repo root — add both the project root
# (celltrack/core) and the vendored scripts dir (their train_unet_transformer + tracking_cellmot).
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))

from celltrack.training.tracked_run import TrackedRun  # noqa: E402 — after sys.path insert
from core.hparams import RunConfig  # noqa: E402
from core.obs import Obs  # noqa: E402
from core.tracking import Tracker  # noqa: E402
from train_unet_transformer import (  # noqa: E402 — vendored, resolves only in .venv-repro
    DEFAULT_AUGMENTATIONS,
    FrameWindowDataset,
    TemporalUNet3D,
    UNetNodeTransformer,
    _POS_EMBED_DIM,
    evaluate,
    load_dataset_windows,
    train_epoch,
)

_UNET_OUT_CHANNELS = 32  # their config.json unet_out_channels
_UNET_LAYERS = [32, 64, 128]  # their config.json unet_layers


def _fold_files(data_dir: Path, splits: Path | None, fold: int) -> tuple[list[Path], list[Path]]:
    """The (train, test) video paths for a fold — from a splits file, or a deterministic seed-0 split."""
    if splits is not None:
        entry = next(f for f in json.loads(splits.read_text()) if f["split"] == fold)
        train, test = entry["train"], entry["test"]
    else:
        stems = sorted(
            p.name[:-5] for p in data_dir.glob("*.zarr") if (data_dir / f"{p.name[:-5]}.geff").exists()
        )
        random.Random(0).shuffle(stems)
        n_val = max(1, len(stems) // 10)
        train, test = stems[n_val:], stems[:n_val]

    def resolve(stem: str) -> Path:
        return data_dir / (stem if stem.endswith(".zarr") else f"{stem}.zarr")

    return [resolve(s) for s in train], [resolve(s) for s in test]


def _load_windows(files: list[Path], cfg: RunConfig, desc: str) -> list:
    data = []
    for f in Obs.progress(files, desc, total=len(files)):
        meta, windows = load_dataset_windows(f, window_size=cfg.data.window_size, downsample=cfg.data.downsample)
        data.append((meta, windows))
    return data


def _build_loaders(cfg: RunConfig, args: argparse.Namespace, log) -> tuple[DataLoader, DataLoader]:
    """Load the fold's videos into windowed datasets and wrap them in train/test DataLoaders."""
    train_files, test_files = _fold_files(args.data_dir, args.splits, args.fold)
    if args.max_videos:
        train_files, test_files = train_files[: args.max_videos], test_files[: args.max_videos]
    log.info("fold %d: %d train, %d test videos", args.fold, len(train_files), len(test_files))
    with Obs.timed(log, "load data"):
        train_data = _load_windows(train_files, cfg, "train")
        test_data = _load_windows(test_files, cfg, "test")
    max_nodes = max(max(w.node_counts) for _, ws in train_data + test_data for w in ws)
    augmentations = None if args.no_augment else DEFAULT_AUGMENTATIONS
    train_ds = FrameWindowDataset(train_data, max_nodes=max_nodes, augmentations=augmentations)
    test_ds = FrameWindowDataset(test_data, max_nodes=max_nodes)

    def build(dataset: object, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset, batch_size=cfg.optim.batch, shuffle=shuffle, num_workers=args.num_workers,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0, pin_memory=False,
        )

    return build(train_ds, shuffle=True), build(test_ds, shuffle=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--splits", type=Path, default=None, help="fold splits json; omit for seed-0 auto-split")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--name", default="ourstunet")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-videos", type=int, default=0, help="cap train+test files (0 = all) for a smoke")
    ap.add_argument("--max-iters", type=int, default=None, help="per-epoch train-iter cap (smoke)")
    ap.add_argument("--no-augment", action="store_true", help="disable their brightness+flip augmentations")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="RunConfig dotted overrides")
    args = ap.parse_args()

    cfg = RunConfig(name=args.name).apply_overrides(args.overrides)
    run_dir = args.out_dir or Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(run_dir / "config.json")
    log = Obs.setup(run_dir / "train.log")
    tracker = Tracker(
        "celltrack", cfg.name,
        params=cfg.model_dump(),
        tags={"fold": args.fold, "splits": str(args.splits or "seed0")},
    ).track_run(run_dir)

    train_loader, test_loader = _build_loaders(cfg, args, log)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    unet = TemporalUNet3D(in_channels=1, out_channels=_UNET_OUT_CHANNELS, layers=_UNET_LAYERS)
    model = UNetNodeTransformer(
        unet=unet, unet_out_channels=_UNET_OUT_CHANNELS, pos_feat_dim=4 * _POS_EMBED_DIM
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)
    log.info("device=%s | params=%d | precision=%s", device, sum(p.numel() for p in model.parameters()), cfg.optim.precision)

    save_path = run_dir / "edge_predictor_best.pth"

    def amp():
        if cfg.optim.precision == "bf16":
            return torch.autocast(device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def do_train() -> dict[str, float]:
        with amp():
            edge, det = train_epoch(
                model, train_loader, optimizer, device,
                cfg.optim.det_loss_weight, cfg.optim.neg_weight,
                max_iters=args.max_iters, pool_kernel_um=cfg.data.pool_kernel_um,
            )
        return {"edge": edge, "det": det}

    def do_eval() -> dict[str, float]:
        with amp():
            loss, acc, recall = evaluate(model, test_loader, device, pool_kernel_um=cfg.data.pool_kernel_um)
        return {"test_loss": loss, "acc": acc, "recall": recall, "score": acc * recall}

    def do_save() -> None:
        torch.save({k.replace("unet.module.", "unet.", 1): v for k, v in model.state_dict().items()}, save_path)

    best = TrackedRun(cfg, tracker, log).fit(do_train, do_eval, do_save)
    log.info("done: best score=%.4f -> %s", best, save_path)
    tracker.artifact(run_dir / "config.json")
    tracker.artifact(save_path)
    tracker.end()


if __name__ == "__main__":
    main()
