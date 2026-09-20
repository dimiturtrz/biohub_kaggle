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

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltrack.data.augmentation import Augmentation
from celltrack.data.tunet_dataset import FrameDataset, FrameTarget
from celltrack.detectors.detection import CELL_SCALE_UM
from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector
from celltrack.eval.kernel_reference import VOXEL_SCALE_UM
from celltrack.eval.model_evaluator import EvalResult, ModelEvaluator
from celltrack.eval.proxy import TEST_MOVIES, VALIDATION_MOVIES, TestMovieProxy
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.losses.center_offset import CenterOffsetLoss
from celltrack.operating_point import TrackerConfig
from celltrack.training.early_stop import EarlyStop
from celltrack.training.run_tracking import RunSetup, TrainingRun, TrainingSplit
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
_BYTES_PER_GB = 1024**3


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
    # Backbone norm: "batch" (published BN, byte-identical) or "group" (GroupNorm, domain-robust, from-scratch
    # only — BN leaks the train movies' running stats onto val frames across the microscope gap; a warm pack's
    # BN buffers cannot load a bufferless GroupNorm, so the swap is rejected at build under warm_start).
    norm: str = "batch"
    # Weight on the center-offset term (`CenterOffsetLoss`), and the head that produces it. 0.0 keeps the
    # shipped detection-only objective and builds no head at all, so the default run is byte-identical to
    # every arm before this one. Above 0 the detector learns, per voxel, the micron vector to the centre it
    # belongs to — the cue a peak readout structurally lacks when two cells' responses merge into one blob.
    offset_weight: float = 0.0
    device: str = "cuda"
    # Fraction of total VRAM the allocator may hold. On Windows/WDDM a GPU over-commit does NOT OOM — it pages
    # over PCIe and CRAWLS while util still reads 100% (power draw is the honest tell). Capping the allocator
    # turns overshoot into an immediate OOM you can react to instead of a silent half-day. Off by default (the
    # published recipe fits 32GB); set it for the (1,2,2) re-pool whose per-frame working set is 4x larger.
    vram_cap: float | None = None
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

    def voxel_um(self) -> tuple[float, float, float]:
        """The micron size of one TRAINING voxel — the imaging scale stretched by this run's downsample.

        Under the shipped (1, 4, 4) the grid is isotropic at 1.625 um, but the offset target must not assume
        that: a re-pooled (1, 2, 2) run has a different y/x and the same regressed number would mean a
        different displacement.
        """
        return cast(
            tuple[float, float, float],
            tuple(scale * step for scale, step in zip(VOXEL_SCALE_UM, self.downsample, strict=True)),
        )

    def offset_radius(self) -> tuple[int, int, int]:
        """How far from a centre the offset term is supervised, in voxels — one cell radius, per axis.

        DERIVED from the physical cell scale the detector's own Gaussian target uses, not chosen: beyond a cell
        radius a voxel is not part of that cell, and with ~6% of cells annotated its true owner is probably not
        in the annotation at all, so supervising further would teach a confidently wrong vector.
        """
        return cast(tuple[int, int, int], tuple(max(1, round(CELL_SCALE_UM / size)) for size in self.voxel_um()))

    @staticmethod
    def arg_parser() -> argparse.ArgumentParser:
        """The command-line surface of this config — one place so `main` stays a wiring shim, not a wall of flags."""
        parser = argparse.ArgumentParser(description="Train the temporal-U-Net detector on the train split.")
        parser.add_argument("--steps", type=int, default=1500)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--neg-weight", type=float, default=1e-2)
        parser.add_argument(
            "--offset-weight",
            type=float,
            default=0.0,
            help="weight on the center-offset term; 0 disables the head entirely (shipped objective)",
        )
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
        parser.add_argument(
            "--vram-cap",
            type=float,
            default=None,
            metavar="FRACTION",
            help="cap the allocator at this fraction of total VRAM so overshoot OOMs instead of PCIe-crawling (0-1)",
        )
        parser.add_argument(
            "--downsample",
            type=int,
            nargs=3,
            default=(1, 4, 4),
            metavar=("DZ", "DY", "DX"),
            help="input strided-read factor (z,y,x); (1,2,2) re-pools y/x to 0.8125um to resolve the 1-voxel confusor",
        )
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
        parser.add_argument("--cosine-lr", action="store_true", help="cosine-decay the learning rate to zero")
        parser.add_argument("--single-frame", action="store_true", help="detection-only T=1 windows (~2x fewer convs)")
        parser.add_argument(
            "--norm",
            choices=("batch", "group"),
            default="batch",
            help="backbone norm: 'batch' (published BN) or 'group' (domain-robust GroupNorm, from-scratch only)",
        )
        parser.add_argument("--compile-backbone", action="store_true", help="torch.compile the U-Net (static shape)")
        parser.add_argument("--resume", action="store_true", help="continue from the .resume.pt snapshot")
        parser.add_argument("--weights", type=str, default="detector_tunet_ours.pt")
        return parser


@dataclass(frozen=True)
class StepLoss:
    """What one optimisation step cost, split by term so a log line can show whether each one is moving."""

    total: float
    offset: float  # microns of mean center-offset error, UNWEIGHTED — 0.0 when no offset head is installed


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
            if self.config.vram_cap is not None:
                torch.cuda.set_per_process_memory_fraction(self.config.vram_cap, torch.cuda.current_device())
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
            self._log_peak_vram(done, window)
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

    def _log_peak_vram(self, done: int, window: int) -> None:
        """Log the steady-state VRAM reserve for a fit; reset once after the first window so the compile and
        construction spikes don't pollute the reading the fit's slope is taken from."""
        if self.config.device != "cuda":
            return
        if done == window:
            torch.cuda.reset_peak_memory_stats()
        logger.info("peak VRAM reserved %.2f GB", torch.cuda.max_memory_reserved() / _BYTES_PER_GB)

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
        running, offsets, done, t0 = 0.0, 0.0, 0, time.perf_counter()
        for frames, centres in dataset.batches(self.config.threads, self.config.prefetch, batch):
            step = self._step(detector, optimization, frames, centres)
            running += step.total
            offsets += step.offset
            done += 1
            if done % _HEARTBEAT_UPDATES == 0:
                rate = done / (time.perf_counter() - t0)
                logger.info(
                    "  ..%d/%d in window | loss %.4f (offset %.3f um) | %.1f it/s (%.0f frames/s)",
                    done,
                    steps,
                    running / done,
                    offsets / done,
                    rate,
                    rate * batch,
                )
        return running / max(done, 1), done / (time.perf_counter() - t0)

    def _detector(self, *, warm_start: bool) -> TemporalUNetDetector:
        """A detector to train — warm-started from the published pack, or fresh at the configured size."""
        if warm_start:
            if self.config.norm != "batch":
                raise ValueError("norm='group' is from-scratch only: a pilkwang BN pack cannot warm-start GroupNorm")
            pack = DataRoot.from_config(_CONFIG).processed(_DATASET) / _PACK_REL
            detector, _ = TemporalUNetDetector.from_pack(
                pack, map_location=self.config.device, offset_head=self.config.offset_weight > 0.0
            )
            logger.info("warm-started from pilkwang pack at %s", pack)
            return detector
        return TemporalUNetDetector(
            self.config.out_channels,
            self.config.layers,
            norm=self.config.norm,
            offset_head=self.config.offset_weight > 0.0,
        )

    def _step(
        self,
        detector: TemporalUNetDetector,
        optimization: _Optimization,
        frames: Float[Tensor, "b z y x"],
        centres: list[Int[Tensor, "n 3"]],
    ) -> "StepLoss":
        """One optimisation step over a batch of frames and their GT voxel centres; returns the loss terms."""
        frames = frames.to(self.config.device, non_blocking=True)
        # Under compile, pin SDPA to the math backend so inductor traces its (clean) composite backward, not the
        # efficient kernel whose compiled backward asserts on the temporal-attention strides. No-op when eager.
        attention = sdpa_kernel(SDPBackend.MATH) if self.config.compile_backbone else contextlib.nullcontext()
        with attention, torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.device == "cuda"):
            heads = detector.forward_heads(frames, single_frame=self.config.single_frame)
            detection = BalancedBCE.of(
                heads.logits, centres, self.config.neg_weight, self.config.ignore_ambiguous_above
            )
            offset = self._offset_loss(heads.offsets, centres)
            loss = detection if offset is None else detection + self.config.offset_weight * offset
        optimization.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(detector.parameters(), self.config.grad_clip)
        optimization.step()
        return StepLoss(float(loss.detach()), float(offset.detach()) if offset is not None else 0.0)

    def _offset_loss(
        self, offsets: Float[Tensor, "b three z y x"] | None, centres: list[Int[Tensor, "n 3"]]
    ) -> Float[Tensor, ""] | None:
        """The center-offset term in microns, or None when no offset head is installed.

        Reported in microns unweighted, so the log says how far the predicted centre actually is — a number
        comparable to the cell radius rather than to whatever `offset_weight` happens to be.
        """
        if offsets is None:
            return None
        return CenterOffsetLoss.of(offsets, centres, self.config.offset_radius(), self.config.voxel_um())


def main() -> None:
    args = TUNetTrainConfig.arg_parser().parse_args()

    config = TUNetTrainConfig(
        steps=args.steps,
        lr=args.lr,
        neg_weight=args.neg_weight,
        offset_weight=args.offset_weight,
        ignore_ambiguous_above=args.ignore_ambiguous_above,
        device=args.device,
        vram_cap=args.vram_cap,
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
        norm=args.norm,
        downsample=tuple(args.downsample),
        recipe=replace(DetectorRecipe(), downsample=tuple(args.downsample)),
    )
    root = DataRoot.from_config(_CONFIG)

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
    with Obs.timed(log, f"enumerating GT frames over {len(train_paths)} train videos"):
        targets = FrameTarget.index(root, Obs.progress(train_paths, "videos", len(train_paths)), config.downsample)
    logger.info("%d annotated frames", len(targets))

    logger.info("selecting through the shipped tracker on the %d validation movies", len(evaluator.proxy.paths))

    setup = RunSetup(save_to, warm_start=args.warm_start, resume=args.resume, split=split)
    TUNetDetectorTrainer(config).train(targets, evaluator, setup)


if __name__ == "__main__":
    main()
