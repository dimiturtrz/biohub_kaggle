"""Train the joint detector+edge model on our fold split — one backbone serving detection and association.

Detection-only training (`tunet_detector`) freezes association; the joint loop runs the temporal backbone
once on the real consecutive pair `(t, t+1)` and reads all three heads off the shared features — the
detection head on each frame and the edge transformer on the node features (see `joint_model`). The three
losses are kept SEPARATE (a detection BalancedBCE per frame, a SoftmaxFocalBCE over the links) so a window
logs each term on its own: that decoupling is the whole point, since the edge and detection curves move on
different scales and a summed number hides which one is learning.

The loop mirrors the detector trainer — eval-sized windows, EarlyStop + save-best + resume, the bf16 /
channels_last / optional torch.compile speed setup — but the eval is the mean validation loss (lower is
better, fed negated into the max-based EarlyStop) rather than a fold score, because the joint objective is
the thing being tracked. Warm-starting (`--warm-start`) continues pilkwang's published detector+transformer;
from scratch it is the honest own-trained baseline. Either way the best checkpoint by held-out loss is saved.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import zarr
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltrack.data.joint_dataset import PairDataset, PairTarget
from celltrack.detectors.tunet import TemporalUNetDetector
from celltrack.edges.edge_scoring import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.eval.bracket import ValidationFold
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.losses.softmax_focal_bce import SoftmaxFocalBCE
from celltrack.training.early_stop import EarlyStop
from celltrack.training.joint_model import JointModel
from celltrack.training.tunet_detector import _Optimization
from core.data.split import AcquisitionFolds
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
# The published pilkwang pack, under the (gitignored) data root's reference area — warm-start source.
_PACK_REL = Path("reference") / "pilkwang" / "split_0"
_HEARTBEAT_UPDATES = 100  # log within-window progress this often, so a long window isn't silent
_SECONDS_PER_HOUR = 3600.0
# The fresh transformer's shape, matching the pilkwang head the warm-start path loads.
_HIDDEN_DIM, _N_HEADS, _N_BLOCKS = 128, 4, 4
_POS_FEATURE_DIM = 4 * _POS_EMBED_DIM  # sinusoidal embed of (t, z, y, x)

Pair = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class JointTrainConfig:
    """Every knob of a joint fine-tune, so a run is one object rather than a long argument list."""

    steps: int = 1500
    lr: float = 1e-4
    neg_weight: float = 1e-2
    det_weight: float = 1.0  # weight on the (two-frame) detection term relative to the edge term
    downsample: tuple[int, int, int] = (1, 4, 4)
    out_channels: int = 32
    layers: tuple[int, ...] = (32, 64, 128)
    device: str = "cuda"
    grad_clip: float = 1.0
    eval_every: int = 500  # the validation pass is serial with training — keep the window coarse
    # Stop after this many consecutive non-improving evals; save-best means stopping early never costs
    # quality, only compute. <1 disables (runs the full `steps` budget).
    patience: int = 5
    es_min_delta: float = 0.0
    seed: int = 0
    compile_backbone: bool = False  # torch.compile the U-Net (static shape); one-time warmup, then fused kernels


@dataclass(frozen=True)
class _EvalResult:
    """One validation pass's mean losses — the total the run selects on, plus its edge and detection terms."""

    loss: float
    edge: float
    det: float


class JointTrainer:
    """Runs the joint detector+edge training loop under one configuration, checkpointing the best val loss."""

    def __init__(self, config: JointTrainConfig) -> None:
        self.config = config

    def train(
        self,
        train_targets: list[PairTarget],
        val_targets: list[PairTarget],
        *,
        warm_start: bool,
        save_to: Path,
        resume: bool = False,
    ) -> float:
        """Train (or fine-tune) on the GT pairs, saving the best held-out loss. Returns that best (lowest) loss.

        The loop runs in eval-sized windows; each window streams `eval_every` GT pairs (one at a time — node
        counts are ragged, so there is no batching across pairs). A full resume snapshot (both heads, the
        optimiser, step, best) is written beside the checkpoint every window; ``resume=True`` continues it.
        """
        torch.set_float32_matmul_precision("high")  # TF32 on the fp32 matmuls the bf16 autocast leaves alone
        torch.backends.cudnn.benchmark = True  # one static frame shape — cudnn picks the fastest algo once
        model = self._model(warm_start=warm_start).to(self.config.device)
        if self.config.device == "cuda":
            # channels_last_3d is a lossless layout that measured faster on the feature convs. torch stubs
            # omit the memory_format overload of Module.to; the call is runtime-valid.
            model = model.to(memory_format=torch.channels_last_3d)  # type: ignore[no-matching-overload]
        if self.config.compile_backbone:
            # Compile the backbone (static shape); the temporal-attention SDPA is forced to MATH at forward
            # time (see `_step`) — its efficient backward has a broken compiled meta-kernel.
            model.detector.unet = torch.compile(model.detector.unet, dynamic=False)  # type: ignore[bad-assignment]
        optimization = _Optimization(torch.optim.AdamW(model.parameters(), lr=self.config.lr), None)

        resume_path = save_to.with_suffix(".resume.pt")
        done, restored = self._restore(model, optimization, resume_path, resume=resume)
        if restored is not None:
            best = restored
        else:
            initial = self._evaluate(model, val_targets)
            best = initial.loss
            logger.info("init val loss %.4f | edge %.4f det %.4f", best, initial.edge, initial.det)
            self._save_checkpoint(save_to, model)

        stop = EarlyStop(self.config.patience, self.config.es_min_delta) if self.config.patience >= 1 else None
        if stop is not None:
            stop.update(-best)  # seed with the init loss (negated — EarlyStop is max-based, lower loss is better)
        run_start = time.perf_counter()
        while done < self.config.steps:
            window = min(self.config.eval_every, self.config.steps - done)
            loss, edge, det, rate = self._run_window(model, optimization, train_targets, window, done)
            done += window
            result = self._evaluate(model, val_targets)
            improved = stop.update(-result.loss) if stop is not None else result.loss <= best
            marker = ""
            if improved:
                best, marker = result.loss, " *saved"
                self._save_checkpoint(save_to, model)
            self._save_resume(resume_path, model, optimization, done, best)
            eta_hours = (self.config.steps - done) / rate / _SECONDS_PER_HOUR if rate else 0.0
            logger.info(
                "step %5d/%d | train loss %.4f (edge %.4f det %.4f) | val %.4f | best %.4f%s | "
                "%.1f it/s | elapsed %.2fh ETA %.2fh",
                done,
                self.config.steps,
                loss,
                edge,
                det,
                result.loss,
                best,
                marker,
                rate,
                (time.perf_counter() - run_start) / _SECONDS_PER_HOUR,
                eta_hours,
            )
            if stop is not None and stop.should_stop:
                logger.info("early stop: %d evals no gain (step %d, best %.4f)", self.config.patience, done, best)
                break

        logger.info("best fold-0 val loss = %.4f, saved to %s", best, save_to)
        return best

    def _model(self, *, warm_start: bool) -> JointModel:
        """A joint model to train — warm-started from the published pack, or fresh at the configured size."""
        if warm_start:
            pack = DataRoot.from_config(_CONFIG).processed(_DATASET) / _PACK_REL
            scorer = EdgeTransformerScorer.from_pack(pack, self.config.device)
            logger.info("warm-started from pilkwang pack at %s", pack)
            return JointModel(scorer.detector, scorer.transformer, self.config.downsample)
        detector = TemporalUNetDetector(self.config.out_channels, self.config.layers)
        transformer = EdgeTransformerScorer._transformer_cls()(  # noqa: SLF001 — the pilkwang head class, mounted
            feat_dim=self.config.out_channels + _POS_FEATURE_DIM,
            hidden_dim=_HIDDEN_DIM,
            n_heads=_N_HEADS,
            n_blocks=_N_BLOCKS,
        )
        return JointModel(detector, transformer, self.config.downsample)

    def _run_window(
        self,
        model: JointModel,
        optimization: _Optimization,
        targets: list[PairTarget],
        steps: int,
        seed_offset: int,
    ) -> tuple[float, float, float, float]:
        """Train `steps` optimiser steps of one GT pair each; returns the mean (loss, edge, det, steps/s)."""
        model.train()
        dataset = PairDataset(targets, steps, self.config.downsample, self.config.seed + seed_offset)
        loss_sum, edge_sum, det_sum, done, t0 = 0.0, 0.0, 0.0, 0, time.perf_counter()
        for pair in dataset.stream():
            loss, edge, det = self._step(model, optimization, pair)
            loss_sum, edge_sum, det_sum, done = loss_sum + loss, edge_sum + edge, det_sum + det, done + 1
            if done % _HEARTBEAT_UPDATES == 0:
                rate = done / (time.perf_counter() - t0)
                logger.info("  ..%d/%d in window | loss %.4f | %.1f it/s", done, steps, loss_sum / done, rate)
        divisor = max(done, 1)
        return loss_sum / divisor, edge_sum / divisor, det_sum / divisor, done / (time.perf_counter() - t0)

    def _step(self, model: JointModel, optimization: _Optimization, pair: Pair) -> tuple[float, float, float]:
        """One optimisation step over a single GT pair; returns (total, edge, detection) as separate floats."""
        loss, edge, det = self._losses(model, pair)
        optimization.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimization.step()
        return float(loss.detach()), float(edge.detach()), float(det.detach())

    def _losses(self, model: JointModel, pair: Pair) -> tuple[Tensor, Tensor, Tensor]:
        """The three loss tensors for one pair — shared by the training step and the no-grad eval.

        Positions reach the model at FULL resolution (it divides by the downsample for the feature grid,
        mirroring the inference scorer); the detection heads emit on the downsampled grid, so their GT
        centres are divided here to match.
        """
        device = self.config.device
        frame_t, frame_t1, source_centres, target_centres, edge_matrix = (item.to(device) for item in pair)
        downsample = torch.tensor(self.config.downsample, dtype=torch.float32, device=device)
        centres_t_grid = (source_centres.to(torch.float32) / downsample).round().long()
        centres_t1_grid = (target_centres.to(torch.float32) / downsample).round().long()
        # Under compile, pin SDPA to the math backend so inductor traces its clean composite backward; no-op eager.
        attention = sdpa_kernel(SDPBackend.MATH) if self.config.compile_backbone else contextlib.nullcontext()
        with attention, torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            detection_t, detection_t1, edge_logits = model.forward(frame_t, frame_t1, source_centres, target_centres)
            edge = SoftmaxFocalBCE.of(edge_logits, edge_matrix)
            det = BalancedBCE.of(detection_t.unsqueeze(0), [centres_t_grid], self.config.neg_weight) + BalancedBCE.of(
                detection_t1.unsqueeze(0), [centres_t1_grid], self.config.neg_weight
            )
            loss = edge + self.config.det_weight * det
        return loss, edge, det

    @torch.no_grad()
    def _evaluate(self, model: JointModel, val_targets: list[PairTarget]) -> _EvalResult:
        """Mean total loss over the held-out pairs (each read once), with its edge and detection terms."""
        model.eval()
        loss_sum, edge_sum, det_sum = 0.0, 0.0, 0.0
        for target in val_targets:
            pair = PairDataset([target], 1, self.config.downsample, self.config.seed)[0]
            loss, edge, det = self._losses(model, pair)
            loss_sum, edge_sum, det_sum = loss_sum + float(loss), edge_sum + float(edge), det_sum + float(det)
        divisor = max(len(val_targets), 1)
        return _EvalResult(loss_sum / divisor, edge_sum / divisor, det_sum / divisor)

    def _restore(
        self, model: JointModel, optimization: _Optimization, resume_path: Path, *, resume: bool
    ) -> tuple[int, float | None]:
        """Load a resume snapshot into both heads and the optimiser, returning (step, best) — (0, None) if fresh."""
        if not (resume and resume_path.exists()):
            return 0, None
        state = torch.load(resume_path, map_location=self.config.device, weights_only=False)
        model.detector.load_state_dict(state["detector"])
        model.transformer.load_state_dict(state["transformer"])
        optimization.load_state_dict(state["optimization"])
        logger.info("resumed from %s at step %d (best %.4f)", resume_path, state["done"], state["best"])
        return int(state["done"]), float(state["best"])

    def _save_resume(
        self, resume_path: Path, model: JointModel, optimization: _Optimization, done: int, best: float
    ) -> None:
        """Write the full run state — both heads, optimiser, step, best — so a killed run can continue."""
        torch.save(
            {
                "detector": model.detector.state_dict(),
                "transformer": model.transformer.state_dict(),
                "optimization": optimization.state_dict(),
                "done": done,
                "best": best,
            },
            resume_path,
        )

    def _save_checkpoint(self, save_to: Path, model: JointModel) -> None:
        """Persist both trained heads plus the structure needed to reconstruct the joint model."""
        torch.save(
            {
                "detector_state": model.detector.state_dict(),
                "transformer_state": model.transformer.state_dict(),
                "config": {
                    "out_channels": self.config.out_channels,
                    "layers": list(self.config.layers),
                    "downsample": list(self.config.downsample),
                },
            },
            save_to,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Train the joint detector+edge model on our fold split.")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--warm-start", action="store_true", help="initialise from the published pilkwang weights")
    parser.add_argument("--det-weight", type=float, default=1.0, help="weight on the detection term vs the edge term")
    parser.add_argument("--compile-backbone", action="store_true", help="torch.compile the U-Net (static shape)")
    parser.add_argument("--patience", type=int, default=5, help="stop after N non-improving evals (<1 disables)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-videos", type=int, default=2, help="held-out validation videos to score the loss on")
    parser.add_argument("--resume", action="store_true", help="continue from the .resume.pt snapshot")
    parser.add_argument("--weights", type=str, default="joint_tunet_ours.pt")
    args = parser.parse_args()

    config = JointTrainConfig(
        steps=args.steps,
        det_weight=args.det_weight,
        device=args.device,
        patience=args.patience,
        compile_backbone=args.compile_backbone,
    )
    root = DataRoot.from_config(_CONFIG)

    def _enumerate(video: Path, graph_source: AnnotatedTracks) -> list[PairTarget]:
        """Every consecutive-frame GT pair of one video, with its per-video intensity quantiles read from OME."""
        quantiles = cast(ImageStatistics, zarr.open_group(video, mode="r").attrs["image_statistics"])["quantiles"]
        q_low, q_high = float(quantiles["0.001"]), float(quantiles["0.999"])
        return PairTarget.enumerate(graph_source.graph, video, q_low, q_high)

    train_paths = AcquisitionFolds().split(root.videos("train"), 0).train
    logger.info("enumerating GT pairs over %d train videos...", len(train_paths))
    train_targets: list[PairTarget] = []
    for path in train_paths:
        train_targets.extend(_enumerate(path, AnnotatedTracks.from_geff(root.track_store(path))))
    logger.info("%d train pairs", len(train_targets))

    fold = ValidationFold.load(root, 0)
    held = list(zip(fold.videos, fold.annotations, strict=True))[: args.val_videos]
    val_targets: list[PairTarget] = []
    for video, truth in held:
        val_targets.extend(_enumerate(video, truth))
    logger.info("%d val pairs over %d held-out videos", len(val_targets), len(held))

    save_to = root.processed(_DATASET) / args.weights
    JointTrainer(config).train(
        train_targets, val_targets, warm_start=args.warm_start, save_to=save_to, resume=args.resume
    )


if __name__ == "__main__":
    main()
