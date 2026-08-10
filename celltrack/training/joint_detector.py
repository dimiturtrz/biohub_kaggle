"""Train the joint detector+edge model — one backbone serving detection and association.

Detection-only training (`tunet_detector`) freezes association; the joint loop runs the temporal backbone
once on the real consecutive pair `(t, t+1)` and reads all three heads off the shared features — the
detection head on each frame and the edge transformer on the node features (see `joint_model`). The losses are
kept SEPARATE (a detection BalancedBCE per frame, a SoftmaxFocalBCE over the links, and an optional InfoNCE
over the node features themselves) so a window logs each term on its own: that decoupling is the whole point,
since the curves move on different scales and a summed number hides which one is learning. The contrastive
term is the only one that asks the BACKBONE for discrimination — the edge loss only asks the head to rank
pairs given whatever features it is handed — and it is off (weight 0) unless a run asks for it.

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
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import zarr
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltrack.data.difficulty_sampler import DifficultySampler
from celltrack.data.joint_dataset import PairDataset, PairTarget
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TEST_MOVIES, VALIDATION_MOVIES, TestMovieProxy
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.losses.info_nce import InfoNCE
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
    # Weight on the InfoNCE term over the NODE FEATURES — the only part of the objective that asks the
    # backbone for discrimination rather than invariance. 0.0 by default: the term is still computed and
    # logged as a diagnostic, but contributes exactly nothing, so an unasked run is the run of yesterday.
    contrastive_weight: float = 0.0
    temperature: float = 0.07  # the InfoNCE softmax temperature over candidate targets
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
    # Decay the rate to zero over the run (cosine). Warm-starting a CONVERGED model is exactly the case a
    # schedule is for: a flat rate keeps taking full-size steps away from an optimum the weights already
    # sit in, while an annealed one explores early and settles. `_Optimization` already advances and
    # persists a scheduler — the joint loop simply never built one.
    cosine_lr: bool = False
    # WHICH published pack to continue. Warm-starting from seed1 produces that seed's descendant, and a
    # descendant adds nothing to an ensemble containing its parent (measured: +0.0075 alone, but 0.9315
    # against the pair's 0.9334). Training the same recipe from a DIFFERENT parent is how the result
    # earns a place — ensembles pay for disagreement, not for quality alone.
    warm_pack: Path = _PACK_REL
    compile_backbone: bool = False  # torch.compile the U-Net (static shape); one-time warmup, then fused kernels
    # Draw each step's pair in proportion to its measured top-1 defect instead of uniformly with replacement
    # (see `DifficultySampler`). Off by default: with it off the loop is the uniform stream of yesterday.
    difficulty_sampling: bool = False


@dataclass(frozen=True)
class PairSplit:
    """The two pair sets a joint run reads — what it optimises on, and what its edge AUC is measured over."""

    train: list[PairTarget]
    val: list[PairTarget]


@dataclass(frozen=True)
class _PairOutcome:
    """One pair's forward — the loss total that is stepped, each term on its own scale, and the edge tensors.

    The edge logits and the annotated matrix ride along because the difficulty estimate the sampler feeds on is
    read off exactly them, and they are free here: the training step already computed both.
    """

    total: Tensor
    edge: Tensor
    detection: Tensor
    contrastive: Tensor
    edge_logits: Tensor
    edge_matrix: Tensor

    def logged(self) -> dict[str, float]:
        """The detached per-term floats under the names the run tracks — a summed number hides which half moved."""
        return {
            "train_loss": float(self.total.detach()),
            "edge_loss": float(self.edge.detach()),
            "det_loss": float(self.detection.detach()),
            "contrastive_loss": float(self.contrastive.detach()),
        }

    def difficulty(self) -> float | None:
        """Fraction of ANNOTATED sources whose true successor the logits do not rank top-1; `None` if there are none.

        Not the loss: with ~1-2% dense annotation a pair's loss tracks how many edges it happens to carry, so
        weighting by it would chase label count. This is bounded [0, 1], independent of edge count, and is the
        defect itself — a mislink is precisely a source whose argmax target is not its annotated successor.
        """
        annotated = self.edge_matrix.sum(dim=1) > 0
        if not bool(annotated.any()):
            return None
        rows = self.edge_matrix[annotated]
        predicted = self.edge_logits[annotated].argmax(dim=1, keepdim=True)
        return 1.0 - float(rows.gather(1, predicted).mean())


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
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        optimization = _Optimization(optimizer, self._schedule(optimizer))

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
        sampler = self._sampler(pairs.train)
        while done < self.config.steps:
            window = min(self.config.eval_every, self.config.steps - done)
            dataset = PairDataset(pairs.train, window, self.config.downsample, self.config.seed + done)
            losses = self._run_window(model, optimization, dataset, sampler)
            done += window
            result = self._evaluate(model, evaluator, pairs.val)
            improved = stop.update(result.score) if stop is not None else result.score >= best
            marker = ""
            if improved:
                best, marker = result.score, " *saved"
                self._save_checkpoint(save_to, model)
            self._save_resume(resume_path, model, optimization, done, best)
            run.window(done, {**self._metrics(result, best), **losses})
            rate = losses["it_per_s"]
            eta_hours = (self.config.steps - done) / rate / _SECONDS_PER_HOUR if rate else 0.0
            logger.info(
                "step %5d/%d | train loss %.4f (edge %.4f det %.4f nce %.4f) | proxy %.4f | best %.4f%s | "
                "node R %.3f ratio %+.2f | edge AUC %.4f | %.1f it/s | elapsed %.2fh ETA %.2fh",
                done,
                self.config.steps,
                losses["train_loss"],
                losses["edge_loss"],
                losses["det_loss"],
                losses["contrastive_loss"],
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
            if sampler is not None:
                logger.info("  sampler | coverage %.3f | entropy %.3f", losses["coverage"], losses["sampling_entropy"])
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

    def _schedule(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
        """A cosine decay to zero over the whole run, or none for a flat rate."""
        if not self.config.cosine_lr:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.steps)

    def _model(self, *, warm_start: bool) -> JointModel:
        """A joint model to train — warm-started from the published pack, or fresh at the configured size."""
        if warm_start:
            pack = DataRoot.from_config(_CONFIG).processed(_DATASET) / self.config.warm_pack
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

    def _sampler(self, targets: list[PairTarget]) -> DifficultySampler | None:
        """One sampler for the WHOLE run (difficulty has to survive across windows), or None for the uniform stream."""
        if not self.config.difficulty_sampling:
            return None
        corpus = PairDataset(targets, self.config.steps, self.config.downsample, self.config.seed)
        return DifficultySampler(corpus.target_count, self.config.seed)

    def _run_window(
        self,
        model: JointModel,
        optimization: _Optimization,
        dataset: PairDataset,
        sampler: DifficultySampler | None = None,
    ) -> dict[str, float]:
        """Train one optimiser step per pair of `dataset`; returns each term's window mean plus `it_per_s`."""
        model.train()
        steps = len(dataset)
        sums: dict[str, float] = {}
        done, t0 = 0, time.perf_counter()
        for target_index, pair in self._pairs(dataset, sampler):
            outcome = self._step(model, optimization, pair)
            for name, value in outcome.logged().items():
                sums[name] = sums.get(name, 0.0) + value
            self._observe(sampler, target_index, outcome)
            done += 1
            if done % _HEARTBEAT_UPDATES == 0:
                rate = done / (time.perf_counter() - t0)
                logger.info("  ..%d/%d in window | loss %.4f | %.1f it/s", done, steps, sums["train_loss"] / done, rate)
        divisor = max(done, 1)
        means = {name: total / divisor for name, total in sums.items()}
        return {**means, "it_per_s": done / (time.perf_counter() - t0), **self._sampling(sampler)}

    def _pairs(self, dataset: PairDataset, sampler: DifficultySampler | None) -> Iterator[tuple[int | None, Pair]]:
        """The window's pairs, each with the corpus index that produced it — `None` on the untouched uniform stream."""
        if sampler is None:
            yield from ((None, pair) for pair in dataset.stream())
            return
        for _ in range(len(dataset)):
            index = sampler.draw()
            yield index, dataset.pair(index)

    def _observe(self, sampler: DifficultySampler | None, target_index: int | None, outcome: _PairOutcome) -> None:
        """Close the feedback loop — the step's own top-1 defect reweights that pair. A no-op on the uniform path."""
        if sampler is None or target_index is None:
            return
        sampler.observe(target_index, outcome.difficulty())

    def _sampling(self, sampler: DifficultySampler | None) -> dict[str, float]:
        """The sampler's two health numbers for the window row; empty when uniform, so an unasked run logs as before."""
        if sampler is None:
            return {}
        return {"coverage": sampler.coverage(), "sampling_entropy": sampler.entropy()}

    def _step(self, model: JointModel, optimization: _Optimization, pair: Pair) -> _PairOutcome:
        """One optimisation step over a single GT pair; returns that pair's loss terms and edge tensors."""
        outcome = self._losses(model, pair)
        optimization.zero_grad()
        outcome.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimization.step()
        return outcome

    def _losses(self, model: JointModel, pair: Pair) -> _PairOutcome:
        """The loss tensors for one pair — shared by the training step and the no-grad eval.

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
            out = model.forward(frame_t, frame_t1, source_centres, target_centres)
            edge = SoftmaxFocalBCE.of(out.edge_logits, edge_matrix)
            det = BalancedBCE.of(
                out.detection_t.unsqueeze(0), [centres_t_grid], self.config.neg_weight
            ) + BalancedBCE.of(out.detection_t1.unsqueeze(0), [centres_t1_grid], self.config.neg_weight)
            contrastive = InfoNCE.of(
                out.source_features, out.target_features, edge_matrix, self.config.temperature, self.config.out_channels
            )
            loss = edge + self.config.det_weight * det + self.config.contrastive_weight * contrastive
        return _PairOutcome(loss, edge, det, contrastive, out.edge_logits, edge_matrix)

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
            pair = PairDataset([target], 1, self.config.downsample, self.config.seed).pair(0)
            frame_t, frame_t1, sources, sinks, _ = (item.to(device) for item in pair)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model.forward(frame_t, frame_t1, sources, sinks).edge_logits
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
    # Eval cadence was a config field with NO flag, so every run silently used 500 — which also fixes what
    # `--patience` means, since patience counts EVAL WINDOWS. At 18024 pairs (one per step) an epoch is
    # ~18000 steps, so patience 5 at eval_every 500 is 14% of an epoch, and at 9000 it is five epochs.
    parser.add_argument("--eval-every", type=int, default=500, help="steps per eval window")
    parser.add_argument("--warm-start", action="store_true", help="initialise from the published pilkwang weights")
    parser.add_argument("--det-weight", type=float, default=1.0, help="weight on the detection term vs the edge term")
    parser.add_argument(
        "--contrastive-weight", type=float, default=0.0, help="weight on the InfoNCE term over the node features"
    )
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE softmax temperature over targets")
    # Fine-tuning a CONVERGED model at its original training rate is the classic way to walk off its optimum,
    # which is what 3000 warm-start steps at 1e-4 did (never beat the init). Exposed so the rate is a
    # variable of the experiment rather than an inherited constant.
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate; lower it when warm-starting")
    parser.add_argument("--cosine-lr", action="store_true", help="decay the rate to zero over the run")
    parser.add_argument(
        "--warm-pack", choices=("seed1", "seed2"), default="seed1", help="which published pack to continue"
    )
    parser.add_argument("--compile-backbone", action="store_true", help="torch.compile the U-Net (static shape)")
    parser.add_argument(
        "--difficulty-sampling",
        action="store_true",
        help="draw pairs in proportion to their measured top-1 defect instead of uniformly with replacement",
    )
    parser.add_argument("--patience", type=int, default=5, help="stop after N non-improving evals (<1 disables)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-videos", type=int, default=2, help="validation movies the edge AUC is measured over")
    parser.add_argument("--eval-threshold", type=float, default=0.5, help="detection threshold of the selector eval")
    parser.add_argument("--resume", action="store_true", help="continue from the .resume.pt snapshot")
    parser.add_argument("--weights", type=str, default="joint_tunet_ours.pt")
    args = parser.parse_args()

    config = JointTrainConfig(
        steps=args.steps,
        eval_every=args.eval_every,
        det_weight=args.det_weight,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        lr=args.lr,
        cosine_lr=args.cosine_lr,
        warm_pack=_PACK_REL if args.warm_pack == "seed1" else _PACK2_REL,
        device=args.device,
        patience=args.patience,
        eval_threshold=args.eval_threshold,
        compile_backbone=args.compile_backbone,
        difficulty_sampling=args.difficulty_sampling,
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
