"""Train our own weights for the temporal-U-Net detector on our fold split.

Detector-only supervision: a frame's ground-truth cell centres are positive voxels, everything else a
heavily down-weighted negative, and the network learns to fire a sharp local maximum at each centre. No
edge transformer and no tracksdata are involved — the detection head is the whole win-path (recall), and
the linker that consumes it is ours. The input recipe is pilkwang's (x4 Y/X downsample, per-video
quantile-norm), trained through the same fake-pair forward used at inference so train and test see the
same distribution.

Warm-starting from the published weights (``--warm-start``) continues their detector on our fold-train
videos; from scratch it is the honest own-trained baseline. Either way the best checkpoint by fold-0
subset score is saved, so the run stops at its own peak rather than a fixed step count. The trainer takes
prepared frames and an evaluator as inputs; ``main`` does the fold loading.
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from jaxtyping import Float, Int
from torch import Tensor, nn

from celltrack.bracket import ValidationFold
from celltrack.linefit_smoother import LinefitSmoother
from celltrack.linkers import LinkerConfig
from celltrack.short_track_filter import ShortTrackFilter
from celltrack.training.tunet_dataset import Augmentation, FrameDataset, FrameTarget
from celltrack.tunet import DetectorRecipe, TemporalUNetDetector
from core.data.split import AcquisitionFolds
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.geometry import Spacing
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
# The published pilkwang detector, kept under the (gitignored) data root's reference area — not vendored,
# not a machine-specific absolute path. Fetch: copy weights/unet_transformer/split_0 from the support pack.
_PACK_REL = Path("reference") / "pilkwang" / "split_0"


@dataclass(frozen=True)
class TUNetTrainConfig:
    """Every knob of a detector fine-tune, so a run is one object rather than a long argument list."""

    steps: int = 1500
    lr: float = 1e-4
    neg_weight: float = 1e-2
    downsample: tuple[int, int, int] = (1, 4, 4)
    out_channels: int = 32
    layers: tuple[int, ...] = (32, 64, 128)
    device: str = "cuda"
    # Frames are decompressed by a thread pool (blosc frees the GIL), not DataLoader worker processes — those
    # deadlock on this Windows box on repeated spawn. `threads` parallel readers, `prefetch` frames in flight.
    threads: int = 12
    prefetch: int = 24
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_subset: int = 2
    eval_threshold: float = 0.5
    seed: int = 0
    augmentation: Augmentation = field(default_factory=Augmentation)
    cosine_lr: bool = False  # decay lr to zero over `steps` (cosine); off keeps the flat lr of the baseline
    recipe: DetectorRecipe = field(default_factory=DetectorRecipe)


@dataclass(frozen=True)
class _FoldEval:
    """The fold-0 subset scorer's fixed context — videos, matcher, and the post-proc chain, built once."""

    videos: list[tuple[Path, AnnotatedTracks]]
    spacing: Spacing
    matcher: DistanceMatcher
    short: ShortTrackFilter
    smooth: LinefitSmoother
    linker_config: LinkerConfig


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


class TUNetDetectorTrainer:
    """Runs the detector-only training loop under one configuration, checkpointing the best fold-0 score."""

    def __init__(self, config: TUNetTrainConfig) -> None:
        self.config = config

    def train(self, targets: list[FrameTarget], evaluator: _FoldEval, *, warm_start: bool, save_to: Path) -> float:
        """Train (or fine-tune) on the prepared frames, saving the best fold-0 score. Returns that best score.

        The loop runs in eval-sized windows; each window's frames are decompressed by a thread pool (see
        `FrameDataset.stream`) so the GPU stays fed without DataLoader worker processes, which deadlock on
        repeated spawn on this box.
        """
        detector = self._detector(warm_start=warm_start).to(self.config.device)
        optimizer = torch.optim.AdamW(detector.parameters(), lr=self.config.lr)
        optimization = _Optimization(optimizer, self._scheduler(optimizer))

        best = self._score_fold(evaluator, detector, self.config.eval_threshold)
        logger.info("init fold-0 subset score = %.4f", best)
        detector.save_checkpoint(save_to, self.config.recipe)

        done = 0
        while done < self.config.steps:
            window = min(self.config.eval_every, self.config.steps - done)
            loss, rate = self._run_window(detector, optimization, targets, window, done)
            done += window
            score = self._score_fold(evaluator, detector, self.config.eval_threshold)
            marker = ""
            if score >= best:
                best, marker = score, " *saved"
                detector.save_checkpoint(save_to, self.config.recipe)
            logger.info(
                "step %5d/%d | loss %.4f | fold-0 subset %.4f | best %.4f%s | %.1f it/s",
                done,
                self.config.steps,
                loss,
                score,
                best,
                marker,
                rate,
            )

        logger.info("best fold-0 subset score = %.4f, saved to %s", best, save_to)
        return best

    def _scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
        """A cosine decay to zero over the whole run, or none for the baseline's flat learning rate."""
        if not self.config.cosine_lr:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.steps)

    def _run_window(
        self,
        detector: TemporalUNetDetector,
        optimization: _Optimization,
        targets: list[FrameTarget],
        steps: int,
        seed_offset: int,
    ) -> tuple[float, float]:
        """Train `steps` steps, frames decompressed by a thread pool. Returns (mean loss, it/s)."""
        detector.train()
        dataset = FrameDataset(
            targets, steps, self.config.downsample, self.config.seed + seed_offset, self.config.augmentation
        )
        running, t0 = 0.0, time.perf_counter()
        for frame, coords in dataset.stream(self.config.threads, self.config.prefetch):
            running += self._step(detector, optimization, frame, coords)
        return running / steps, steps / (time.perf_counter() - t0)

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
        frame: Float[Tensor, "z y x"],
        coords: Int[Tensor, "n 3"],
    ) -> float:
        """One optimisation step over a single frame and its GT voxel centres; returns the scalar loss."""
        frame = frame.to(self.config.device, non_blocking=True)
        coords = coords.to(self.config.device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=self.config.device == "cuda"):
            logits = detector(frame)
            loss = self._detection_loss(logits, coords, self.config.neg_weight)
        optimization.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(detector.parameters(), self.config.grad_clip)
        optimization.step()
        return float(loss.detach())

    def _score_fold(self, evaluator: _FoldEval, detector: TemporalUNetDetector, threshold: float) -> float:
        """The pooled fold-0 subset score — the same edge-Jaccard the leaderboard reports — at one threshold."""
        buckets: dict[str, list[VideoMetrics]] = {}
        linker = evaluator.linker_config.build(evaluator.spacing)
        for path, truth in evaluator.videos:
            detections = detector.detections(path, threshold, self.config.recipe, self.config.device)
            graph = evaluator.smooth.transform(evaluator.short.transform(linker.link(detections)))
            buckets.setdefault(AcquisitionFolds.prefix_of(path), []).append(
                VideoMetrics.of(graph, truth, evaluator.matcher)
            )
        return SplitScore.of([m for ms in buckets.values() for m in ms]).score

    @staticmethod
    def _detection_loss(
        logits: Float[Tensor, "z y x"], coords: Int[Tensor, "n 3"], neg_weight: float
    ) -> Float[Tensor, ""]:
        """Count-normalised BCE: GT voxels weigh 1/n_pos, others neg_weight/n_neg — pilkwang's detector loss."""
        target = torch.zeros_like(logits)
        bounds = torch.tensor(logits.shape, device=coords.device)
        inside = (coords >= 0).all(dim=1) & (coords < bounds).all(dim=1)
        kept = coords[inside]
        target[kept[:, 0], kept[:, 1], kept[:, 2]] = 1.0
        n_pos = target.sum().clamp(min=1)
        n_neg = (target.numel() - n_pos).clamp(min=1)
        weight = torch.where(target == 1.0, 1.0 / n_pos, neg_weight / n_neg)
        return F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Train the temporal-U-Net detector on our fold split.")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--neg-weight", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=12, help="parallel frame-decompress threads")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-subset", type=int, default=2)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--warm-start", action="store_true", help="initialise from the published pilkwang weights")
    parser.add_argument("--aug-brightness", type=float, default=0.0, help="multiplicative intensity jitter half-range")
    parser.add_argument("--aug-offset", type=float, default=0.0, help="additive intensity jitter half-range")
    parser.add_argument("--aug-flip", action="store_true", help="random flips along y and x (the in-plane axes)")
    parser.add_argument("--cosine-lr", action="store_true", help="cosine-decay the learning rate to zero over the run")
    parser.add_argument("--weights", type=str, default="detector_tunet_ours.pt")
    args = parser.parse_args()

    config = TUNetTrainConfig(
        steps=args.steps,
        lr=args.lr,
        neg_weight=args.neg_weight,
        device=args.device,
        threads=args.threads,
        eval_every=args.eval_every,
        eval_subset=args.eval_subset,
        eval_threshold=args.eval_threshold,
        augmentation=Augmentation(
            brightness=args.aug_brightness,
            offset=args.aug_offset,
            flip_axes=(1, 2) if args.aug_flip else (),
        ),
        cosine_lr=args.cosine_lr,
    )
    root = DataRoot.from_config(_CONFIG)
    dz, dy, dx = config.downsample

    train_paths = AcquisitionFolds().split(root.videos("train"), 0).train
    logger.info("enumerating GT frames over %d train videos...", len(train_paths))
    targets: list[FrameTarget] = []
    for path in train_paths:
        quantiles = cast(ImageStatistics, zarr.open_group(path, mode="r").attrs["image_statistics"])["quantiles"]
        q_low, q_high = float(quantiles["0.001"]), float(quantiles["0.999"])
        coordinates = AnnotatedTracks.from_geff(root.track_store(path)).graph.coordinates
        for timepoint in np.unique(coordinates[:, 0]):
            frame_coords = coordinates[coordinates[:, 0] == timepoint][:, 1:] // np.array([dz, dy, dx])
            targets.append(FrameTarget(path, int(timepoint), q_low, q_high, frame_coords.astype(np.int64)))
    logger.info("%d annotated frames", len(targets))

    fold = ValidationFold.load(root, 0)
    fold = fold.stratified_subset(config.eval_subset) if config.eval_subset else fold
    evaluator = _FoldEval(
        videos=[(path, truth) for path, (_video, truth) in zip(fold.videos, fold.images(), strict=True)],
        spacing=fold.spacing,
        matcher=DistanceMatcher(spacing=fold.spacing),
        short=ShortTrackFilter(min_length=3),
        smooth=LinefitSmoother(strength=0.8),
        linker_config=LinkerConfig(name="motion"),
    )
    logger.info("eval on %d fold-0 subset videos", len(evaluator.videos))

    save_to = root.processed(_DATASET) / args.weights
    TUNetDetectorTrainer(config).train(targets, evaluator, warm_start=args.warm_start, save_to=save_to)


if __name__ == "__main__":
    main()
