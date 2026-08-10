"""Train the joint detector+edge model — one backbone serving detection and association.

Detection-only training (`tunet_detector`) freezes association; the joint loop runs the temporal backbone
once on the real consecutive pair `(t, t+1)` and reads all three heads off the shared features — the
detection head on each frame and the edge transformer on the node features (see `joint_model`). The three
losses are kept SEPARATE (a detection BalancedBCE per frame, a SoftmaxFocalBCE over the links) so a window
logs each term on its own: that decoupling is the whole point, since the edge and detection curves move on
different scales and a summed number hides which one is learning.

The loop mirrors the detector trainer — eval-sized windows, EarlyStop + save-best + resume, the bf16 /
channels_last / optional torch.compile speed setup — and selects on the same LB-aligned number: the
`ModelEvaluator` score of BOTH trained heads through the shipped `CellTracker` on the densely-annotated
VALIDATION movies (`celltrack.eval.proxy`), with the TEST four held out of training and out of selection so
they remain a clean leaderboard estimate. Held-out loss does not transfer, and it hides which half moved; the
pipeline score is the metric the leaderboard reports, with the edge PRC-AUC over the same validation movies'
pairs logged beside it as the association half's own diagnostic.
Warm-starting (`--warm-start`) continues pilkwang's published detector+transformer; from scratch it is the honest
own-trained baseline. Either way the best checkpoint by proxy score is saved.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import zarr
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltrack.data.joint_dataset import PairDataset, PairTarget
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TEST_MOVIES, VALIDATION_MOVIES, TestMovieProxy
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.losses.softmax_focal_bce import SoftmaxFocalBCE
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from celltrack.tracker import TrackerConfig
from celltrack.training.early_stop import EarlyStop
from celltrack.training.run_tracking import RunSetup, TrainingRun, TrainingSplit
from celltrack.training.tunet_detector import _Optimization
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.metrics.edge_auc import EdgeAUC
from core.obs import Obs
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
# The published pilkwang pack, under the (gitignored) data root's reference area — warm-start source.
_PACK_REL = Path("reference") / "pilkwang" / "split_0"
# The second pilkwang seed — the proxy eval's tracker is the shipped one, whose recipe blends two seeds' packs.
# Only the PROXY needs them: a joint run's own heads supply both the nodes and the affinity being selected on.
_PACK2_REL = Path("reference") / "pilkwang" / "seed2" / "weights" / "unet_transformer" / "split_0"
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
    # The eval is a full `CellTracker` pass over the four proxy movies, serial with training — keep it coarse.
    eval_every: int = 500
    eval_threshold: float = 0.5  # the detection threshold the selector's tracker runs at
    # The selector eval skips flip-TTA: it buys a faithful score the checkpoint choice doesn't need, at 4x cost.
    eval_tta: bool = False
    # Stop after this many consecutive non-improving evals; save-best means stopping early never costs
    # quality, only compute. <1 disables (runs the full `steps` budget).
    patience: int = 5
    es_min_delta: float = 0.0
    seed: int = 0
    compile_backbone: bool = False  # torch.compile the U-Net (static shape); one-time warmup, then fused kernels


@dataclass(frozen=True)
class PairSplit:
    """The two pair sets a joint run reads — what it optimises on, and what its edge AUC is measured over."""

    train: list[PairTarget]
    val: list[PairTarget]


@dataclass(frozen=True)
class _EvalResult:
    """One eval window's numbers — the LB-aligned proxy score selected on, its node levers, and the edge AUC."""

    score: float
    node_recall: float
    node_ratio: float
    edge_auc: float


class JointTrainer:
    """Runs the joint detector+edge training loop under one configuration, checkpointing the best proxy score."""

    def __init__(self, config: JointTrainConfig) -> None:
        self.config = config

    def train(self, pairs: PairSplit, evaluator: ModelEvaluator, setup: RunSetup) -> float:
        """Train (or fine-tune) on the GT pairs, saving the best proxy-pipeline score. Returns that best score.

        The loop runs in eval-sized windows; each window streams `eval_every` GT pairs (one at a time — node
        counts are ragged, so there is no batching across pairs). A full resume snapshot (both heads, the
        optimiser, step, best) is written beside the checkpoint every window; ``setup.resume`` continues it.
        """
        save_to = setup.save_to
        torch.set_float32_matmul_precision("high")  # TF32 on the fp32 matmuls the bf16 autocast leaves alone
        torch.backends.cudnn.benchmark = True  # one static frame shape — cudnn picks the fastest algo once
        model = self._model(warm_start=setup.warm_start).to(self.config.device)
        if self.config.device == "cuda":
            # channels_last_3d is a lossless layout that measured faster on the feature convs. torch stubs
            # omit the memory_format overload of Module.to; the call is runtime-valid.
            model = model.to(memory_format=torch.channels_last_3d)  # type: ignore[no-matching-overload]
        if self.config.compile_backbone:
            # Compile the backbone (static shape); the temporal-attention SDPA is forced to MATH at forward
            # time (see `_step`) — its efficient backward has a broken compiled meta-kernel.
            model.detector.unet = torch.compile(model.detector.unet, dynamic=False)  # type: ignore[bad-assignment]
        optimization = _Optimization(torch.optim.AdamW(model.parameters(), lr=self.config.lr), None)

        run = TrainingRun.open(asdict(self.config), setup)
        resume_path = save_to.with_suffix(".resume.pt")
        done, restored = self._restore(model, optimization, resume_path, resume=setup.resume)
        if restored is not None:
            best = restored
        else:
            initial = self._evaluate(model, evaluator, pairs.val)
            best = initial.score
            logger.info(
                "init proxy %.4f | node R %.3f ratio %+.2f | edge AUC %.4f",
                best,
                initial.node_recall,
                initial.node_ratio,
                initial.edge_auc,
            )
            run.window(done, self._metrics(initial, best))
            self._save_checkpoint(save_to, model)

        stop = EarlyStop(self.config.patience, self.config.es_min_delta) if self.config.patience >= 1 else None
        if stop is not None:
            stop.update(best)  # seed with the init score so a first worse window doesn't read as an improvement
        run_start = time.perf_counter()
        while done < self.config.steps:
            window = min(self.config.eval_every, self.config.steps - done)
            loss, edge, det, rate = self._run_window(model, optimization, pairs.train, window, done)
            done += window
            result = self._evaluate(model, evaluator, pairs.val)
            improved = stop.update(result.score) if stop is not None else result.score >= best
            marker = ""
            if improved:
                best, marker = result.score, " *saved"
                self._save_checkpoint(save_to, model)
            self._save_resume(resume_path, model, optimization, done, best)
            losses = {"train_loss": loss, "edge_loss": edge, "det_loss": det, "it_per_s": rate}
            run.window(done, {**self._metrics(result, best), **losses})
            eta_hours = (self.config.steps - done) / rate / _SECONDS_PER_HOUR if rate else 0.0
            logger.info(
                "step %5d/%d | train loss %.4f (edge %.4f det %.4f) | proxy %.4f | best %.4f%s | "
                "node R %.3f ratio %+.2f | edge AUC %.4f | %.1f it/s | elapsed %.2fh ETA %.2fh",
                done,
                self.config.steps,
                loss,
                edge,
                det,
                result.score,
                best,
                marker,
                result.node_recall,
                result.node_ratio,
                result.edge_auc,
                rate,
                (time.perf_counter() - run_start) / _SECONDS_PER_HOUR,
                eta_hours,
            )
            if stop is not None and stop.should_stop:
                logger.info("early stop: %d evals no gain (step %d, best %.4f)", self.config.patience, done, best)
                break

        logger.info("best proxy score = %.4f, saved to %s", best, save_to)
        run.close(best)
        return best

    def _metrics(self, result: _EvalResult, best: float) -> dict[str, float]:
        """The eval half of one tracked row — the selected-on score, the best so far, and both halves' levers."""
        return {
            "proxy_score": result.score,
            "best_score": best,
            "node_recall": result.node_recall,
            "node_ratio": result.node_ratio,
            "edge_auc": result.edge_auc,
        }

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

    def _evaluate(self, model: JointModel, evaluator: ModelEvaluator, val_targets: list[PairTarget]) -> _EvalResult:
        """Both trained heads through the shipped tracker on the proxy, plus their edge ranking on the val pairs."""
        model.eval()
        result = evaluator.evaluate_joint(model)
        return _EvalResult(result.score, result.node_recall, result.node_ratio, self._edge_auc(model, val_targets))

    @torch.no_grad()
    def _edge_auc(self, model: JointModel, val_targets: list[PairTarget]) -> float:
        """Mean edge PRC-AUC over the held-out pairs — the association half's ranking quality, one number.

        Scores are the pipeline's own: the logits soft-maxed over the sources of each target. Pairs whose
        annotation carries no link leave the average precision undefined and are skipped, not counted as zero.
        """
        model.eval()
        device = self.config.device
        scored: list[float] = []
        for target in val_targets:
            pair = PairDataset([target], 1, self.config.downsample, self.config.seed)[0]
            frame_t, frame_t1, sources, sinks, _ = (item.to(device) for item in pair)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, _, logits = model.forward(frame_t, frame_t1, sources, sinks)
            auc = EdgeAUC.of(torch.softmax(logits.float(), dim=0).cpu().numpy(), target.edge_matrix)
            if not math.isnan(auc):
                scored.append(auc)
        return sum(scored) / len(scored) if scored else float("nan")

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
    parser = argparse.ArgumentParser(description="Train the joint detector+edge model on the non-held-out videos.")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--warm-start", action="store_true", help="initialise from the published pilkwang weights")
    parser.add_argument("--det-weight", type=float, default=1.0, help="weight on the detection term vs the edge term")
    parser.add_argument("--compile-backbone", action="store_true", help="torch.compile the U-Net (static shape)")
    parser.add_argument("--patience", type=int, default=5, help="stop after N non-improving evals (<1 disables)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-videos", type=int, default=2, help="validation movies the edge AUC is measured over")
    parser.add_argument("--eval-threshold", type=float, default=0.5, help="detection threshold of the selector eval")
    parser.add_argument("--resume", action="store_true", help="continue from the .resume.pt snapshot")
    parser.add_argument("--weights", type=str, default="joint_tunet_ours.pt")
    args = parser.parse_args()

    config = JointTrainConfig(
        steps=args.steps,
        det_weight=args.det_weight,
        device=args.device,
        patience=args.patience,
        eval_threshold=args.eval_threshold,
        compile_backbone=args.compile_backbone,
    )
    root = DataRoot.from_config(_CONFIG)

    def _enumerate(video: Path, graph_source: AnnotatedTracks) -> list[PairTarget]:
        """Every consecutive-frame GT pair of one video, with its per-video intensity quantiles read from OME."""
        quantiles = cast(ImageStatistics, zarr.open_group(video, mode="r").attrs["image_statistics"])["quantiles"]
        q_low, q_high = float(quantiles["0.001"]), float(quantiles["0.999"])
        return PairTarget.enumerate(graph_source.graph, video, q_low, q_high)

    proc = root.processed(_DATASET)
    save_to = proc / args.weights
    log = Obs.setup(save_to.with_suffix(".log"), truncate=not args.resume)  # tail-able while the run goes
    recipe = DetectorRecipe(downsample=config.downsample, tta=config.eval_tta)
    tracker_config = TrackerConfig(threshold=config.eval_threshold)
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
    train_targets: list[PairTarget] = []
    with Obs.timed(log, f"enumerating GT pairs over {len(train_paths)} train videos"):
        for path in Obs.progress(train_paths, "videos", len(train_paths)):
            train_targets.extend(_enumerate(path, AnnotatedTracks.from_geff(root.track_store(path))))
    logger.info("%d train pairs", len(train_targets))

    # The edge AUC reads the SAME held-out movies the checkpoint is selected on — one validation set, and a
    # pair-level forward per annotated frame, so `--val-videos` caps how many of them contribute.
    held = list(zip(evaluator.proxy.paths, evaluator.proxy.truths, strict=True))[: args.val_videos]
    val_targets: list[PairTarget] = []
    for video, truth in held:
        val_targets.extend(_enumerate(video, truth))
    logger.info("%d val pairs over %d validation movies", len(val_targets), len(held))

    setup = RunSetup(save_to, warm_start=args.warm_start, resume=args.resume, split=split)
    JointTrainer(config).train(PairSplit(train_targets, val_targets), evaluator, setup)


if __name__ == "__main__":
    main()
