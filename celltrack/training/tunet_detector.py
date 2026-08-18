"""Train our own weights for the temporal-U-Net detector on every train video but the four held-out movies.

Detector-only supervision: a frame's ground-truth cell centres are positive voxels, everything else a
heavily down-weighted negative, and the network learns to fire a sharp local maximum at each centre. No
edge transformer and no tracksdata are involved — the detection head is the whole win-path (recall), and
the linker that consumes it is ours. The input recipe is pilkwang's (x4 Y/X downsample, per-video
quantile-norm), trained through the same fake-pair forward used at inference so train and test see the
same distribution.

The split is three-way and lives in `celltrack.eval.proxy`: eight movies are held out of training, the
densely-annotated VALIDATION four choose the checkpoint through the *shipped* CellTracker (`ModelEvaluator`),
and the TEST four are never selected on, so they stay a clean leaderboard estimate (`proxy_eval`). Everything
else in `train/` is trained on — a hashed fold split is neither needed nor honest here (the published weights
we warm-start from were fitted on ~93 % of fold-0's validation, and that eval ranked the same weights 0.837
against 0.893 through the real pipeline).

Warm-starting from the published weights (``--warm-start``) continues their detector on our training
videos; from scratch it is the honest own-trained baseline. Either way the best checkpoint by the
LB-aligned proxy score is saved, so the run stops at its own peak rather than a fixed step count. The
trainer takes prepared frames and an evaluator as inputs; ``main`` mounts that proxy evaluator.
"""

import argparse
import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltrack.data.augmentation import Augmentation
from celltrack.data.tunet_dataset import FrameDataset, FrameTarget
from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector
from celltrack.eval.model_evaluator import EvalResult, ModelEvaluator
from celltrack.eval.proxy import TEST_MOVIES, VALIDATION_MOVIES, TestMovieProxy
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.operating_point import TrackerConfig
from celltrack.training.early_stop import EarlyStop
from celltrack.training.run_tracking import RunSetup, TrainingRun, TrainingSplit
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.obs import Obs
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
# The published pilkwang detector, kept under the (gitignored) data root's reference area — not vendored,
# not a machine-specific absolute path. Fetch: copy weights/unet_transformer/split_0 from the support pack.
_PACK_REL = Path("reference") / "pilkwang" / "split_0"
# The second pilkwang seed's edge head — the tracker blends both seeds' affinities (0.8/0.2). Independent of the
# detector under test, so the eval mounts it once; the trained weights supply only the detection nodes.
_PACK2_REL = Path("reference") / "pilkwang" / "seed2" / "weights" / "unet_transformer" / "split_0"
_HEARTBEAT_UPDATES = 100  # log within-window progress this often, so a long window isn't silent
_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class TUNetTrainConfig:
    """Every knob of a detector fine-tune, so a run is one object rather than a long argument list."""

    steps: int = 1500
    lr: float = 1e-4
    neg_weight: float = 1e-2
    # Sigmoid response above which an UNANNOTATED voxel stops being supervised as background (see
    # `BalancedBCE`). Our labels name ~1-2% of a frame's cells, so a zero target teaches the detector to stop
    # firing on the rest; from scratch that is survivable, warm-starting a saturated detector it is not. None
    # keeps the objective these weights were trained under; `main` derives its value from `TrackerConfig`.
    ignore_ambiguous_above: float | None = None
    downsample: tuple[int, int, int] = (1, 4, 4)
    out_channels: int = 32
    layers: tuple[int, ...] = (32, 64, 128)
    device: str = "cuda"
    # Frames are decompressed by a thread pool (blosc frees the GIL), not DataLoader worker processes — those
    # deadlock on this Windows box on repeated spawn. `threads` parallel readers, `prefetch` frames in flight.
    threads: int = 12
    prefetch: int = 24
    # Every training frame is the same downsampled shape, so a step forwards `batch_size` frames in one pass
    # (batch=1 starved the GPU). The loss averages over the batch, so the learning rate carries over unchanged.
    batch_size: int = 8
    grad_clip: float = 1.0
    eval_every: int = 500  # subset eval is a full-frame forward per video, serial with training — keep it sparse
    eval_subset: int = 2
    # The selector's operating point is the SHIPPED one, carried whole. A bare threshold grafted onto
    # TrackerConfig's neutral defaults ranks checkpoints under a linker we do not deploy, and the sign of an
    # affinity-side change depends on its consumer — so a partially-specified operating point can silently
    # discard the better model. `--eval-threshold` overrides this one field for a detector whose calibration
    # differs (our from-scratch detector peaked at 0.999 where the published pack peaks at 0.97).
    eval_tracker: TrackerConfig = field(default_factory=TrackerConfig.shipped)
    # The in-loop eval is a checkpoint SELECTOR, not the final number — 4x flip-TTA buys a faithful score the
    # selector doesn't need, at 4x the eval cost. Off by default; the final full-frame calibration re-enables it.
    eval_tta: bool = False
    # Stop after this many consecutive non-improving evals (the validation curve dips then recovers, so 8 clears
    # the longest observed recovering dip). Save-best means stopping early never costs quality, only compute.
    # <1 disables (runs the full `steps` budget).
    patience: int = 8
    es_min_delta: float = 0.0
    seed: int = 0
    augmentation: Augmentation = field(default_factory=Augmentation)
    cosine_lr: bool = False  # decay lr to zero over `steps` (cosine); off keeps the flat lr of the baseline
    # Detection-only single-frame windows (skip the temporal backbone's duplicate frame) — ~half the convs,
    # validated near-lossless (g50). Safe for training our own weights; inference stays two-frame until an LB A/B.
    single_frame: bool = False
    compile_backbone: bool = False  # torch.compile the U-Net (static 64^3 shape); one-time warmup, then fused kernels
    recipe: DetectorRecipe = field(default_factory=DetectorRecipe)


@dataclass(frozen=True)
class _Optimization:
    """The optimiser and its optional learning-rate schedule, advanced together after each step."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None

    def zero_grad(self) -> None:
        """Clear the accumulated gradients before the next backward pass."""
        self.optimizer.zero_grad()

    def step(self) -> None:
        """Take one optimiser step, then advance the schedule if there is one."""
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def state_dict(self) -> dict[str, object]:
        """The optimiser and schedule state, so a resumed run continues the same momentum and decay."""
        schedule = None if self.scheduler is None else self.scheduler.state_dict()
        return {"optimizer": self.optimizer.state_dict(), "scheduler": schedule}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the optimiser and (if present on both sides) the schedule from a saved state."""
        self.optimizer.load_state_dict(cast(dict[str, object], state["optimizer"]))
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(cast(dict[str, object], state["scheduler"]))


class TUNetDetectorTrainer:
    """Runs the detector-only training loop under one configuration, checkpointing the best proxy-pipeline score."""

    def __init__(self, config: TUNetTrainConfig) -> None:
        self.config = config

    def train(self, targets: list[FrameTarget], evaluator: ModelEvaluator, setup: RunSetup) -> float:
        """Train (or fine-tune) on the prepared frames, saving the best proxy-pipeline score. Returns that best score.

        The loop runs in eval-sized windows; each window's frames are decompressed by a thread pool (see
        `FrameDataset.stream`) so the GPU stays fed without DataLoader worker processes, which deadlock on
        repeated spawn on this box. A full resume snapshot (weights, optimiser, schedule, step, best) is written
        beside the checkpoint every window; ``setup.resume`` picks the latest up where a killed run left off.
        """
        save_to = setup.save_to
        torch.set_float32_matmul_precision("high")  # TF32 on the fp32 matmuls the bf16 autocast leaves alone
        torch.backends.cudnn.benchmark = True  # one static frame shape (64^3) — cudnn picks the fastest algo once
        detector = self._detector(warm_start=setup.warm_start).to(self.config.device)
        if self.config.device == "cuda":
            # channels_last_3d is a lossless layout that measured ~1.2x on the feature convs — always on for CUDA,
            # not a knob. torch stubs omit the memory_format overload of Module.to; the call is runtime-valid.
            detector = detector.to(memory_format=torch.channels_last_3d)  # type: ignore[no-matching-overload]
        if self.config.compile_backbone:
            # torch.compile returns an OptimizedModule (a Module) the stubs type as Any; compile the backbone
            # here because it is called via forward_batch, not the wrapped forward. The temporal-attention SDPA
            # is forced to the MATH backend at forward time (see `_step`) — its efficient/flash backward has a
            # broken compiled meta-kernel, while math's backward is primitives inductor traces cleanly.
            detector.unet = torch.compile(detector.unet, dynamic=False)  # type: ignore[bad-assignment]
        optimizer = torch.optim.AdamW(detector.parameters(), lr=self.config.lr)
        optimization = _Optimization(optimizer, self._scheduler(optimizer))

        run = TrainingRun.open(asdict(self.config), setup)
        resume_path = save_to.with_suffix(".resume.pt")
        done, restored = self._restore(detector, optimization, resume_path, resume=setup.resume)
        if restored is not None:
            best = restored
        else:
            initial = evaluator.evaluate(detector)
            # Selection reads the CLAMPED score, never the faithful one: the competition's node-count factor
            # pays a bonus for undershooting, so the faithful metric can rank a detector that found 30% fewer
            # cells ABOVE an honest one — and this trainer optimises detection, where that bias bites hardest.
            best = initial.selection_score
            logger.info(
                "init proxy %.4f (sel %.4f) | node R %.3f ratio %+.2f",
                initial.score,
                best,
                initial.node_recall,
                initial.node_ratio,
            )
            run.window(done, self._metrics(initial, best))
            detector.save_checkpoint(save_to, self.config.recipe)

        stop = EarlyStop(self.config.patience, self.config.es_min_delta) if self.config.patience >= 1 else None
        if stop is not None:
            stop.update(best)  # seed with the init score so a first worse window doesn't read as an improvement
        run_start = time.perf_counter()
        while done < self.config.steps:
            window = min(self.config.eval_every, self.config.steps - done)
            loss, rate = self._run_window(detector, optimization, targets, window, done)
            done += window
            result = evaluator.evaluate(detector)
            score = result.selection_score
            improved = stop.update(score) if stop is not None else score >= best
            marker = ""
            if improved:
                best, marker = score, " *saved"
                detector.save_checkpoint(save_to, self.config.recipe)
            self._save_resume(resume_path, detector, optimization, done, best)
            window_metrics = {"train_loss": loss, "it_per_s": rate, "frames_per_s": rate * self.config.batch_size}
            run.window(done, {**self._metrics(result, best), **window_metrics})
            eta_hours = (self.config.steps - done) / rate / _SECONDS_PER_HOUR if rate else 0.0
            logger.info(
                "step %5d/%d | loss %.4f | proxy %.4f (sel %.4f) | best %.4f%s | node R %.3f ratio %+.2f | "
                "%.1f it/s (%.0f frames/s) | elapsed %.2fh ETA %.2fh",
                done,
                self.config.steps,
                loss,
                result.score,
                score,
                best,
                marker,
                result.node_recall,
                result.node_ratio,
                rate,
                rate * self.config.batch_size,
                (time.perf_counter() - run_start) / _SECONDS_PER_HOUR,
                eta_hours,
            )
            if stop is not None and stop.should_stop:
                logger.info("early stop: %d evals no gain (step %d, best %.4f)", self.config.patience, done, best)
                break

        logger.info("best proxy-pipeline score = %.4f, saved to %s", best, save_to)
        run.close(best)
        return best

    def _metrics(self, result: EvalResult, best: float) -> dict[str, float]:
        """The eval half of one tracked row — the selected-on score, the best so far, and the node levers."""
        return {
            "proxy_score": result.score,
            "best_score": best,
            "node_recall": result.node_recall,
            "node_ratio": result.node_ratio,
        }

    def _scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
        """A cosine decay to zero over the whole run, or none for the baseline's flat learning rate."""
        if not self.config.cosine_lr:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.steps)

    def _restore(
        self, detector: TemporalUNetDetector, optimization: _Optimization, resume_path: Path, *, resume: bool
    ) -> tuple[int, float | None]:
        """Load a resume snapshot into the detector and optimiser, returning (step, best) — (0, None) if fresh."""
        if not (resume and resume_path.exists()):
            return 0, None
        state = torch.load(resume_path, map_location=self.config.device, weights_only=False)
        detector.load_state_dict(state["detector"])
        optimization.load_state_dict(state["optimization"])
        logger.info("resumed from %s at step %d (best %.4f)", resume_path, state["done"], state["best"])
        return int(state["done"]), float(state["best"])

    def _save_resume(
        self, resume_path: Path, detector: TemporalUNetDetector, optimization: _Optimization, done: int, best: float
    ) -> None:
        """Write the full run state — weights, optimiser, schedule, step, best — so a killed run can continue."""
        torch.save(
            {
                "detector": detector.state_dict(),
                "optimization": optimization.state_dict(),
                "done": done,
                "best": best,
            },
            resume_path,
        )

    def _run_window(
        self,
        detector: TemporalUNetDetector,
        optimization: _Optimization,
        targets: list[FrameTarget],
        steps: int,
        seed_offset: int,
    ) -> tuple[float, float]:
        """Train `steps` optimiser steps of `batch_size` frames each, decompressed by a thread pool.

        Returns (mean loss, steps/s). One step forwards a whole batch, so the window draws `steps * batch_size`
        frames from the pool; the last step of a window may run a short batch if that product isn't exact.
        """
        detector.train()
        batch = self.config.batch_size
        dataset = FrameDataset(
            targets, steps * batch, self.config.downsample, self.config.seed + seed_offset, self.config.augmentation
        )
        running, done, t0 = 0.0, 0, time.perf_counter()
        for frames, centres in dataset.batches(self.config.threads, self.config.prefetch, batch):
            running += self._step(detector, optimization, frames, centres)
            done += 1
            if done % _HEARTBEAT_UPDATES == 0:
                rate = done / (time.perf_counter() - t0)
                logger.info(
                    "  ..%d/%d in window | loss %.4f | %.1f it/s (%.0f frames/s)",
                    done,
                    steps,
                    running / done,
                    rate,
                    rate * batch,
                )
        return running / max(done, 1), done / (time.perf_counter() - t0)

    def _detector(self, *, warm_start: bool) -> TemporalUNetDetector:
        """A detector to train — warm-started from the published pack, or fresh at the configured size."""
        if warm_start:
            pack = DataRoot.from_config(_CONFIG).processed(_DATASET) / _PACK_REL
            detector, _ = TemporalUNetDetector.from_pack(pack, map_location=self.config.device)
            logger.info("warm-started from pilkwang pack at %s", pack)
            return detector
        return TemporalUNetDetector(self.config.out_channels, self.config.layers)

    def _step(
        self,
        detector: TemporalUNetDetector,
        optimization: _Optimization,
        frames: Float[Tensor, "b z y x"],
        centres: list[Int[Tensor, "n 3"]],
    ) -> float:
        """One optimisation step over a batch of frames and their GT voxel centres; returns the scalar loss."""
        frames = frames.to(self.config.device, non_blocking=True)
        # Under compile, pin SDPA to the math backend so inductor traces its (clean) composite backward, not the
        # efficient kernel whose compiled backward asserts on the temporal-attention strides. No-op when eager.
        attention = sdpa_kernel(SDPBackend.MATH) if self.config.compile_backbone else contextlib.nullcontext()
        with attention, torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.device == "cuda"):
            logits = detector.forward_batch(frames, single_frame=self.config.single_frame)
            loss = BalancedBCE.of(logits, centres, self.config.neg_weight, self.config.ignore_ambiguous_above)
        optimization.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(detector.parameters(), self.config.grad_clip)
        optimization.step()
        return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the temporal-U-Net detector on the non-held-out train videos.")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--neg-weight", type=float, default=1e-2)
    # DERIVED, not swept: given bare, the flag takes the pipeline's own inference operating point — the
    # response at which the shipped tracker already calls the voxel a cell, so the voxels it spares are
    # exactly the detections our 1-2% annotation cannot adjudicate.
    parser.add_argument(
        "--ignore-ambiguous-above",
        type=float,
        nargs="?",
        const=TrackerConfig.shipped().threshold,
        default=None,
        help="leave unannotated voxels above this sigmoid response unsupervised (bare = the tracker threshold)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=12, help="parallel frame-decompress threads")
    parser.add_argument("--batch-size", type=int, default=8, help="frames forwarded per optimiser step (one shape)")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-subset", type=int, default=2)
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=TrackerConfig.shipped().threshold,
        help="detection threshold of the selector eval (defaults to the SHIPPED operating point's)",
    )
    parser.add_argument("--eval-tta", action="store_true", help="flip-TTA in the eval (4x cost; off=selector)")
    parser.add_argument("--patience", type=int, default=8, help="stop after N non-improving evals (<1 disables)")
    parser.add_argument("--warm-start", action="store_true", help="initialise from the published pilkwang weights")
    parser.add_argument(
        "--aug-brightness",
        type=float,
        default=0.1,
        help="multiplicative intensity jitter half-range (default on: ±10%%, the frontier's brightness_augment)",
    )
    parser.add_argument("--aug-offset", type=float, default=0.0, help="additive intensity jitter half-range")
    parser.add_argument(
        "--aug-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="random flips along y and x (the in-plane axes; on by default, --no-aug-flip to disable)",
    )
    parser.add_argument("--cosine-lr", action="store_true", help="cosine-decay the learning rate to zero over the run")
    parser.add_argument("--single-frame", action="store_true", help="detection-only T=1 windows (~2x fewer convs, g50)")
    parser.add_argument("--compile-backbone", action="store_true", help="torch.compile the U-Net (static shape)")
    parser.add_argument("--resume", action="store_true", help="continue from the .resume.pt snapshot")
    parser.add_argument("--weights", type=str, default="detector_tunet_ours.pt")
    args = parser.parse_args()

    config = TUNetTrainConfig(
        steps=args.steps,
        lr=args.lr,
        neg_weight=args.neg_weight,
        ignore_ambiguous_above=args.ignore_ambiguous_above,
        device=args.device,
        threads=args.threads,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_subset=args.eval_subset,
        eval_tracker=replace(TrackerConfig.shipped(), threshold=args.eval_threshold),
        eval_tta=args.eval_tta,
        patience=args.patience,
        augmentation=Augmentation(
            brightness=args.aug_brightness,
            offset=args.aug_offset,
            flip_axes=(1, 2) if args.aug_flip else (),
        ),
        cosine_lr=args.cosine_lr,
        single_frame=args.single_frame,
        compile_backbone=args.compile_backbone,
    )
    root = DataRoot.from_config(_CONFIG)
    dz, dy, dx = config.downsample

    proc = root.processed(_DATASET)
    save_to = proc / args.weights
    log = Obs.setup(save_to.with_suffix(".log"), truncate=not args.resume)  # tail-able while the run goes
    tracker_config = config.eval_tracker
    recipe = replace(config.recipe, tta=config.eval_tta)  # selector eval skips TTA by default (4x cheaper)
    with Obs.timed(log, "mounting the proxy evaluator"):
        validation = TestMovieProxy.load(root, VALIDATION_MOVIES)
        evaluator = ModelEvaluator.mount(
            validation, (proc / _PACK_REL, proc / _PACK2_REL), recipe, config.device, tracker_config
        )

    train_paths = TestMovieProxy.training_videos(root.videos("train"))
    split = TrainingSplit(len(train_paths), len(VALIDATION_MOVIES), len(TEST_MOVIES))
    logger.info(
        "split: %d train / %d validation (selected on) / %d test (estimate only)",
        split.train,
        split.validation,
        split.test,
    )
    targets: list[FrameTarget] = []
    with Obs.timed(log, f"enumerating GT frames over {len(train_paths)} train videos"):
        for path in Obs.progress(train_paths, "videos", len(train_paths)):
            quantiles = cast(ImageStatistics, zarr.open_group(path, mode="r").attrs["image_statistics"])["quantiles"]
            q_low, q_high = float(quantiles["0.001"]), float(quantiles["0.999"])
            coordinates = AnnotatedTracks.from_geff(root.track_store(path)).graph.coordinates
            for timepoint in np.unique(coordinates[:, 0]):
                frame_coords = coordinates[coordinates[:, 0] == timepoint][:, 1:] // np.array([dz, dy, dx])
                targets.append(FrameTarget(path, int(timepoint), q_low, q_high, frame_coords.astype(np.int64)))
    logger.info("%d annotated frames", len(targets))

    logger.info("selecting through the shipped tracker on the %d validation movies", len(evaluator.proxy.paths))

    setup = RunSetup(save_to, warm_start=args.warm_start, resume=args.resume, split=split)
    TUNetDetectorTrainer(config).train(targets, evaluator, setup)


if __name__ == "__main__":
    main()
