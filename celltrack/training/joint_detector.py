"""Train the joint detector+edge model — one backbone serving detection and association.

Detection-only training (`tunet_detector`) freezes association; the joint loop runs the temporal backbone
once on the real consecutive pair `(t, t+1)` and reads all three heads off the shared features — the
detection head on each frame and the edge transformer on the node features (see `joint_model`). The losses are
kept SEPARATE (a detection BalancedBCE per frame, a SoftmaxFocalBCE over the links, an optional InfoNCE
over the node features themselves, and an optional hard-negative ranking term over the mined near-decoys of
each annotated source) so a window logs each term on its own: that decoupling is the whole point,
since the curves move on different scales and a summed number hides which one is learning. The contrastive
term is the only one that asks for discrimination at all — the edge loss only asks the head to rank pairs
given whatever features it is handed — and it is off (weight 0) unless a run asks for it. WHERE it acts is
its own decision (`ContrastiveSite`): asked of the shared features it is measured to trade detection away
for discrimination, so the term can instead be given a projection head of its own to shape. The hard-negative
term is the other one that is off by default: it mines the same near-decoys the refuted `edge_finetune` probe
did, but scores them by MARGIN (`HardNegativeMargin`) and — the difference that makes it a new experiment —
with the backbone unfrozen, since every association retrain that failed here trained against frozen features.

The loop mirrors the detector trainer — eval-sized windows, EarlyStop + save-best + resume, the bf16 /
channels_last / optional torch.compile speed setup — and selects on the same LB-aligned number: the
`ModelEvaluator` score of BOTH trained heads through the shipped `CellTracker` on the densely-annotated
VALIDATION movies (`celltrack.eval.proxy`), with the TEST four held out of training and out of selection so
they remain a clean leaderboard estimate. Held-out loss does not transfer, and it hides which half moved; the
pipeline score is the metric the leaderboard reports, with the edge PRC-AUC over the same validation movies'
pairs logged beside it as the association half's own diagnostic. Selection reads the CLAMPED form of that score
(`SplitScore.selection_score`) — the faithful metric pays a bonus for under-detection which our sparsely
annotated proxy does not charge for and the dense hidden evaluation does — while the faithful number is logged
beside it unchanged, as `proxy X (sel Y)`.
Warm-starting (`--warm-start`) continues pilkwang's published detector+transformer; from scratch it is the honest
own-trained baseline. Either way the best checkpoint by selection score is saved.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltrack.data.difficulty_sampler import DifficultySampler
from celltrack.data.gpu_scene import GpuScenes
from celltrack.data.joint_dataset import PairDataset, PairSample, PairTarget
from celltrack.data.pair_curriculum import FixedMixture, PacedMixture, PairCurriculum, PairDraw
from celltrack.data.synthetic_scene import SceneConfig
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.eval.model_evaluator import EvalResult, ModelEvaluator
from celltrack.eval.proxy import VALIDATION_MOVIES, TestMovieProxy
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.losses.hard_negative_margin import HardNegativeMargin
from celltrack.losses.sample_weight import SampleWeight
from celltrack.losses.softmax_focal_bce import SoftmaxFocalBCE
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.models.joint_model import JointForward, JointModel
from celltrack.models.lora import LoRA
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from celltrack.operating_point import TrackerConfig
from celltrack.training.contrastive_term import ContrastiveTerm
from celltrack.training.early_stop import EarlyStop
from celltrack.training.joint_checkpoint import RESUME_SUFFIX, JointCheckpoint, RunProgress
from celltrack.training.joint_cli import JointCli
from celltrack.training.joint_config import (
    WARM_PACKS,
    DataCfg,
    EvalCfg,
    JointTrainConfig,
    LoraCfg,
    LossCfg,
    ModelCfg,
    OptimCfg,
    RuntimeCfg,
    ScheduleCfg,
)
from celltrack.training.pair_split import PairSplit
from celltrack.training.prior_velocity_source import PriorVelocitySource
from celltrack.training.run_tracking import RunSetup, TrainingRun
from celltrack.training.tunet_detector import _Optimization
from core.metrics.edge_auc import EdgeAUC
from core.obs import Obs
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
# The proxy eval's tracker is the shipped one, whose recipe blends BOTH published seeds' packs (`WARM_PACKS`).
# Only the PROXY needs them: a joint run's own heads supply both the nodes and the affinity being selected on.
_HEARTBEAT_UPDATES = 100  # log within-window progress this often, so a long window isn't silent
_SECONDS_PER_HOUR = 3600.0
# The fresh transformer's shape, matching the pilkwang head the warm-start path loads.
_HIDDEN_DIM, _N_HEADS, _N_BLOCKS = 128, 4, 4
_POS_FEATURE_DIM = 4 * _POS_EMBED_DIM  # sinusoidal embed of (t, z, y, x)
# The uniform stream draws no index at all; -1 says so without a `int | None` that every reader must unwrap.
_UNTRACKED_INDEX = -1
# The (1,4,4) downsample makes the heads' grid isotropic at the z-spacing, so one downsampled voxel is this many
# micrometres on every axis — the unit the reliability weight's crowd gate is measured in (see downsample memory).
_ISOTROPIC_UM = 1.625
_GATE_UM = TrackerConfig.shipped().linker.gate_um  # the linker's admission gate — a source's competing rivals


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
    hard_negative: Tensor
    edge_logits: Tensor
    edge_matrix: Tensor
    ignored_fraction: float | None = None
    reliability: Tensor | None = None  # the pair's raw SNR/(1+crowd) weight, batch-normalised by the step

    def logged(self) -> dict[str, float]:
        """The detached per-term floats under the names the run tracks — a summed number hides which half moved.

        The masked fraction rides along only when the mask is on, so an unasked run's rows are the rows of
        yesterday. It is worth a column: it falling toward zero as the detector sharpens IS the self-balancing
        property — a model whose confident responses are all annotated has nothing left to leave unsupervised.
        """
        ambiguous = {} if self.ignored_fraction is None else {"ignored_fraction": self.ignored_fraction}
        return {
            "train_loss": float(self.total.detach()),
            "edge_loss": float(self.edge.detach()),
            "det_loss": float(self.detection.detach()),
            "contrastive_loss": float(self.contrastive.detach()),
            "hard_negative_loss": float(self.hard_negative.detach()),
            **ambiguous,
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
    """One eval window's read-out: the whole shipped-pipeline eval, plus the edge head's collapse check.

    `pipeline` carries the leaderboard-comparable score, the clamped `selection_score` a window is selected on
    (see `VideoMetrics.clamped_edge_jaccard`), the node levers, and the mislink numbers that measure association
    where it is HARD. `edge_auc` is NOT one of those: it ranks the affinity over GROUND-TRUTH node pairs, the
    easy case, and reads ~1.0 trained or not — a collapse alarm, logged as one (see `JointTrainer._edge_auc`).
    """

    pipeline: EvalResult
    edge_auc: float


class JointTrainer:
    """Runs the joint detector+edge training loop under one configuration, checkpointing the best proxy score."""

    def __init__(self, config: JointTrainConfig) -> None:
        self.config = config
        # The contrastive term's placement is fixed for the whole run, and the projecting sites carry weights
        # the optimiser must own — so it is built once, here, beside the config that chose it.
        self.contrastive = ContrastiveTerm(config.loss, config.model.out_channels)

    def train(self, pairs: PairSplit, evaluator: ModelEvaluator, setup: RunSetup) -> float:
        """Train (or fine-tune) on the GT pairs, saving the best SELECTION score. Returns that best score.

        The loop runs in eval-sized windows; each window streams `eval_every` GT pairs (one at a time — node
        counts are ragged, so there is no batching across pairs). A full resume snapshot (both heads, the
        optimiser, step, best) is written beside the checkpoint every window; ``setup.resume`` continues it.

        One pair per step makes the corpus itself the epoch, so the schedule is resolved here — the first
        moment the pair count is known — and every later reader (the scheduler's horizon included) sees steps.
        """
        steps_per_epoch = max(1, len(pairs.train))
        self.config = self.config.resolved(steps_per_epoch)
        schedule = self.config.schedule
        save_to = setup.save_to
        torch.set_float32_matmul_precision("high")  # TF32 on the fp32 matmuls the bf16 autocast leaves alone
        torch.backends.cudnn.benchmark = True  # one static frame shape — cudnn picks the fastest algo once
        model, optimization = self._prepare(warm_start=setup.warm_start)

        run = TrainingRun.open(self.config.model_dump(), setup)
        resume_path = save_to.with_suffix(RESUME_SUFFIX)
        done, restored = self._restore(model, optimization, resume_path, resume=setup.resume)
        best = restored if restored is not None else self._initial_best(model, evaluator, pairs, run, save_to)

        stop = EarlyStop(schedule.patience, schedule.es_min_delta) if schedule.patience >= 1 else None
        if stop is not None:
            stop.update(best)  # seed with the init score so a first worse window doesn't read as an improvement
        run_start = time.perf_counter()
        curriculum = self._curriculum(pairs)
        corpus = pairs.corpus()
        while done < schedule.steps:
            window = min(schedule.eval_every, schedule.steps - done)
            dataset = PairDataset(
                corpus,
                window,
                self.config.data.downsample,
                self.config.runtime.seed + done,
                with_previous=self.config.data.prior_velocity,
            )
            losses = self._run_window(model, optimization, dataset, curriculum)
            done += window
            result = self._evaluate(model, evaluator, pairs.val)
            pipeline = result.pipeline
            # Weighted by the difficulty the window was TRAINED at, then the controller is told the raw score:
            # a policy that moves the diet must not be judged by a metric that forgot the diet moved.
            selected = pipeline.selection_score * (curriculum.selection_weight() if curriculum else 1.0)
            if curriculum is not None:
                curriculum.update(pipeline.selection_score)
            improved = stop.update(selected) if stop is not None else selected >= best
            marker = ""
            if improved:
                best, marker = selected, " *saved"
                self._checkpoint().save_best(save_to, model)
            progress = RunProgress(done=done, best=best)
            self._checkpoint().save_snapshot(resume_path, model, self.contrastive, optimization, progress)
            run.window(done, {**self._metrics(result, best), **losses})
            rate = losses["it_per_s"]
            eta_hours = (schedule.steps - done) / rate / _SECONDS_PER_HOUR if rate else 0.0
            logger.info(
                "epoch %g/%g | step %5d/%d | train loss %.4f (edge %.4f det %.4f nce %.4f hn %.4f) | "
                "proxy %.4f (sel %.4f) | "
                "best %.4f%s | node R %.3f ratio %+.2f | mislinks %d inv %.2f P true %.3f vs chosen %.3f | "
                "sanity AUC %.4f | %.1f it/s | elapsed %.2fh ETA %.2fh",
                round(done / steps_per_epoch, 2),
                round(schedule.steps / steps_per_epoch, 2),
                done,
                schedule.steps,
                losses["train_loss"],
                losses["edge_loss"],
                losses["det_loss"],
                losses["contrastive_loss"],
                losses["hard_negative_loss"],
                pipeline.score,
                selected,
                best,
                marker,
                pipeline.node_recall,
                pipeline.node_ratio,
                pipeline.mislinks,
                pipeline.inverted_fraction,
                pipeline.mean_p_true,
                pipeline.mean_p_chosen,
                result.edge_auc,
                rate,
                (time.perf_counter() - run_start) / _SECONDS_PER_HOUR,
                eta_hours,
            )
            if curriculum is not None:
                health = curriculum.health()
                logger.info("  curriculum | %s", " | ".join(f"{name} {value:.3f}" for name, value in health.items()))
            if stop is not None and stop.should_stop:
                logger.info("early stop: %d evals no gain (step %d, best %.4f)", schedule.patience, done, best)
                break

        logger.info("best selection score = %.4f, saved to %s", best, save_to)
        run.close(best)
        return best

    def _initial_best(
        self, model: JointModel, evaluator: ModelEvaluator, pairs: PairSplit, run: TrainingRun, save_to: Path
    ) -> float:
        """The score before a single step — the bar a warm start must beat, tracked and checkpointed as window 0."""
        initial = self._evaluate(model, evaluator, pairs.val)
        logger.info(
            "init proxy %.4f (sel %.4f) | node R %.3f ratio %+.2f | mislinks %d inv %.2f P true %.3f vs chosen %.3f"
            " | sanity AUC %.4f",
            initial.pipeline.score,
            initial.pipeline.selection_score,
            initial.pipeline.node_recall,
            initial.pipeline.node_ratio,
            initial.pipeline.mislinks,
            initial.pipeline.inverted_fraction,
            initial.pipeline.mean_p_true,
            initial.pipeline.mean_p_chosen,
            initial.edge_auc,
        )
        run.window(0, self._metrics(initial, initial.pipeline.selection_score))
        self._checkpoint().save_best(save_to, model)
        return initial.pipeline.selection_score

    def _metrics(self, result: _EvalResult, best: float) -> dict[str, float]:
        """The eval half of one tracked row — the honest score, the selected-on one, the best so far, and the levers."""
        return {
            "proxy_score": result.pipeline.score,
            "selection_score": result.pipeline.selection_score,
            "best_score": best,
            "node_recall": result.pipeline.node_recall,
            "node_ratio": result.pipeline.node_ratio,
            "edge_auc": result.edge_auc,
            "inverted_fraction": result.pipeline.inverted_fraction,
            "mean_p_true": result.pipeline.mean_p_true,
            "mean_p_chosen": result.pipeline.mean_p_chosen,
            "mislinks": result.pipeline.mislinks,
        }

    def _schedule(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
        """A cosine decay to zero over the whole run, or none for a flat rate."""
        if not self.config.optim.cosine_lr:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.schedule.steps)

    def _prepare(self, *, warm_start: bool) -> tuple[JointModel, _Optimization]:
        """The run's trained objects, built in the one order that keeps BOTH a warm start and a widened head.

        The order is load-bearing. `widen_linear_inputs` REPLACES the head's input projection with a new
        parameter object, so the sequence must be: build at the published width → load the pretrained weights
        into it → widen → and only then hand `model.parameters()` to the optimiser. An optimiser constructed
        before the widening would hold the discarded projection and train nothing through it, in silence.
        """
        model = self._model(warm_start=warm_start).to(self.config.runtime.device)
        if self.config.model.temporal_position:  # motion sight; the position table lands on the model's device
            model.detector.install_temporal_position()
        self._velocity.widen(model)
        if self._lora.enabled:  # freeze the base, adapt the association locus; detection is anchored
            adapted = LoRA.inject(model, self._lora.to_config())
            if adapted == 0:
                raise ValueError(f"LoRA targets {self._lora.targets} matched no layers")
            logger.info("LoRA: froze base, adapted %d layers (rank %d)", adapted, self._lora.rank)
            model = model.to(self.config.runtime.device)  # the new adapter params default to CPU — move them
        if self.config.runtime.device == "cuda":
            # channels_last_3d is a lossless layout that measured faster on the feature convs. torch stubs
            # omit the memory_format overload of Module.to; the call is runtime-valid.
            model = model.to(memory_format=torch.channels_last_3d)  # type: ignore[no-matching-overload]
        if self.config.runtime.compile_backbone:
            # Compile the backbone (static shape); the temporal-attention SDPA is forced to MATH at forward
            # time (see `_step`) — its efficient backward has a broken compiled meta-kernel.
            model.detector.unet = torch.compile(model.detector.unet, dynamic=False)  # type: ignore[bad-assignment]
            if self.config.data.gpu_scene_fraction:  # fuse the uniform-N generated-scene edge head (measured 4x)
                model.compile_scene_head(SceneConfig().n_cells)
        self.contrastive.to(self.config.runtime.device)
        optimizer = torch.optim.AdamW(self._trained(model), lr=self.config.optim.lr)
        return model, _Optimization(optimizer, self._schedule(optimizer))

    def _trained(self, model: JointModel) -> list[nn.Parameter]:
        """Every parameter the optimiser owns and the clip covers — both heads, plus the contrastive head.

        At the `FEATURES` site the contrastive term holds none, so the list IS `model.parameters()` in the
        same order: an unasked run's optimiser and gradient clip are the ones of yesterday, to the byte.

        Under LoRA the base is frozen, so the optimiser is handed ONLY the adapter deltas — the frozen weights
        would contribute zero gradient but still be tracked, and the point of the adapters is that they are all
        that moves.
        """
        if self._lora.enabled:
            return [*LoRA.adapter_parameters(model), *self.contrastive.parameters()]
        return [*model.parameters(), *self.contrastive.parameters()]

    @property
    def _velocity(self) -> PriorVelocitySource:
        """The run's history feed, reading the flag live — `train` re-resolves the config before the model exists."""
        return PriorVelocitySource(enabled=self.config.data.prior_velocity)

    @property
    def _lora(self) -> LoraCfg:
        """The run's adapter config, read live — keeps callers off a four-deep reach through `config.model.lora`."""
        return self.config.model.lora

    def _model(self, *, warm_start: bool) -> JointModel:
        """A joint model to train — warm-started from the published pack, or fresh at the configured size."""
        architecture, downsample = self.config.model, self.config.data.downsample
        if warm_start:
            pack = DataRoot.from_config(_CONFIG).processed(_DATASET) / architecture.warm_pack
            scorer = EdgeTransformerScorer.from_pack(pack, self.config.runtime.device)
            logger.info("warm-started from pilkwang pack at %s", pack)
            return JointModel(scorer.detector, scorer.transformer, downsample)
        detector = TemporalUNetDetector(architecture.out_channels, architecture.layers)
        transformer = EdgeTransformerScorer._transformer_cls()(  # noqa: SLF001 — the pilkwang head class, mounted
            feat_dim=architecture.out_channels + _POS_FEATURE_DIM,
            hidden_dim=_HIDDEN_DIM,
            n_heads=_N_HEADS,
            n_blocks=_N_BLOCKS,
        )
        return JointModel(detector, transformer, downsample)

    def _curriculum(self, pairs: PairSplit) -> PairCurriculum | None:
        """The run's ONE policy over (which pair, how much it counts), or None for the uniform stream.

        Built once for the whole run because every policy here carries state across windows — a difficulty
        estimate, an EMA, a realised share. The three are mutually exclusive by construction: a paced mix
        needs a synthetic population, a fixed mix needs a share of one, and the per-pair difficulty sampler
        needs neither.
        """
        data, seed = self.config.data, self.config.runtime.seed
        counts = (len(pairs.train), len(pairs.synthetic))
        if pairs.synthetic and data.paced_curriculum:
            windows = math.ceil(self.config.schedule.steps / self.config.schedule.eval_every)
            return PacedMixture(*counts, windows, seed)
        if pairs.synthetic and data.synthetic_fraction:
            return FixedMixture(*counts, data.synthetic_fraction, seed)
        if data.difficulty_sampling:
            return DifficultySampler(len(pairs.train), seed)
        return None

    def _run_window(
        self,
        model: JointModel,
        optimization: _Optimization,
        dataset: PairDataset,
        curriculum: PairCurriculum | None = None,
    ) -> dict[str, float]:
        """Train one optimiser step per pair of `dataset`; returns each term's window mean plus `it_per_s`."""
        model.train()
        steps = len(dataset)
        batch_size = self.config.optim.batch_size
        gpu_count = round(batch_size * self.config.data.gpu_scene_fraction)
        real_count = max(1, batch_size - gpu_count)
        sums: dict[str, float] = {}
        done, t0 = 0, time.perf_counter()
        batch: list[tuple[PairDraw, PairSample]] = []
        for draw, pair in self._pairs(dataset, curriculum):
            batch.append((draw, pair))
            if len(batch) < real_count and done + len(batch) < steps:
                continue
            # Each batch is `real_count` real pairs plus `gpu_count` FRESH GPU-generated hard scenes, forwarded
            # together. Only the real pairs are logged and fed back — the generated ones shape the gradient but
            # are not the run's measured population — so the loss row stays comparable to a no-generation run.
            outcomes = self._step_batch(model, optimization, batch + self._gpu_batch(gpu_count), len(batch))
            for (draw, _), outcome in zip(batch, outcomes[: len(batch)], strict=True):
                for name, value in outcome.logged().items():
                    sums[name] = sums.get(name, 0.0) + value
                self._observe(curriculum, draw.index, outcome)
                done += 1
            batch = []
            if done % _HEARTBEAT_UPDATES < batch_size:
                rate = done / (time.perf_counter() - t0)
                logger.info("  ..%d/%d in window | loss %.4f | %.1f it/s", done, steps, sums["train_loss"] / done, rate)
        divisor = max(done, 1)
        means = {name: total / divisor for name, total in sums.items()}
        health = curriculum.health() if curriculum is not None else {}
        return {**means, "it_per_s": done / (time.perf_counter() - t0), **health}

    def _pairs(self, dataset: PairDataset, curriculum: PairCurriculum | None) -> Iterator[tuple[PairDraw, PairSample]]:
        """The window's pairs, each with the draw that produced it — index `None` on the untouched uniform stream."""
        if curriculum is None:
            runtime = self.config.runtime
            stream = dataset.stream(runtime.loader_threads, runtime.loader_prefetch)
            yield from ((PairDraw(index=_UNTRACKED_INDEX), pair) for pair in stream)
            return
        for _ in range(len(dataset)):
            draw = curriculum.step()
            yield draw, dataset.pair(draw.index)

    def _observe(self, curriculum: PairCurriculum | None, index: int, outcome: _PairOutcome) -> None:
        """Close the per-pair feedback loop — the step's own top-1 defect. A no-op on the uniform path."""
        if curriculum is None or index == _UNTRACKED_INDEX:
            return
        curriculum.observe(index, outcome.difficulty())

    def _step_batch(
        self, model: JointModel, optimization: _Optimization, batch: list[tuple[PairDraw, PairSample]], real_count: int
    ) -> list[_PairOutcome]:
        """One step over a batch — backbone forwarded ONCE over stacked windows, gradient the batch MEAN.

        The first `real_count` pairs are real and train BOTH heads. Generated scenes beyond train association
        always, and detection too when `gpu_scene_detection` is set — their 100% labels are the honest detection
        signal the real corpus (~98% of cells supervised as background) lacks, judged on the LB not sparse recall.
        """
        device = self.config.runtime.device
        scene_detection = self.config.data.gpu_scene_detection
        samples = [pair.to(device) for _, pair in batch]
        velocities = [self._velocity.of(model, sample, self._autocast()) for sample in samples]
        windows = torch.stack([torch.stack([s.frame_t, s.frame_t1], dim=0) for s in samples], dim=0)  # (B, 2, Z, Y, X)
        with self._attention(), self._autocast():
            forwards = model.forward_batch(
                windows, [s.source_centres for s in samples], [s.target_centres for s in samples], velocities
            )
            outcomes = [
                self._losses_from(out, sample, detection=index < real_count or scene_detection)
                for index, (out, sample) in enumerate(zip(forwards, samples, strict=True))
            ]
        weights = self._reliability_weights(outcomes)
        terms = [
            draw.weight * weight * out.total for (draw, _), out, weight in zip(batch, outcomes, weights, strict=True)
        ]
        (torch.stack(terms).sum() / len(batch)).backward()
        nn.utils.clip_grad_norm_(self._trained(model), self.config.optim.grad_clip)
        optimization.step()
        optimization.zero_grad()
        return outcomes

    def _reliability_weights(self, outcomes: list[_PairOutcome]) -> list[Tensor]:
        """Per-pair reliability multipliers normalised to batch-mean 1 — redistributes emphasis, no rescale.

        All ones when the weighting is off (any pair carries no reliability), so an unasked run steps exactly
        yesterday's gradient. Mean-1 normalisation keeps the total loss scale — and therefore the effective
        learning rate — unchanged, so only the RELATIVE weight of clean vs ambiguous pairs moves.
        """
        raw = [out.reliability for out in outcomes]
        device = outcomes[0].total.device
        if any(weight is None for weight in raw):
            return [torch.ones((), device=device) for _ in outcomes]
        stacked = torch.stack([weight for weight in raw if weight is not None])
        return list(stacked / stacked.mean().clamp(min=1e-6))

    def _gpu_batch(self, count: int) -> list[tuple[PairDraw, PairSample]]:
        """`count` FRESH GPU-generated hard scene-pairs — untracked (they shape the gradient, not the feedback).

        The generator persists across windows (lazily created, seeded once) so every batch draws a NEW crowded
        population rather than replaying a pool — the whole point of generating on the GPU rather than loading.
        """
        if count <= 0:
            return []
        if not hasattr(self, "_scene_source"):
            device = self.config.runtime.device
            object.__setattr__(self, "_scene_source", GpuScenes(SceneConfig(), device))
            object.__setattr__(self, "_scene_rng", torch.Generator(device).manual_seed(self.config.runtime.seed))
        pairs = self._scene_source.batch(count, self._scene_rng)  # type: ignore[attr-defined]
        return [(PairDraw(index=_UNTRACKED_INDEX), pair) for pair in pairs]

    def _losses(self, model: JointModel, pair: PairSample) -> _PairOutcome:
        """One pair's loss tensors — the per-pair forward then `_losses_from`; used by the no-grad eval."""
        sample = pair.to(self.config.runtime.device)
        velocity = self._velocity.of(model, sample, self._autocast())
        with self._attention(), self._autocast():
            out = model.forward(sample.frame_t, sample.frame_t1, sample.source_centres, sample.target_centres, velocity)
            return self._losses_from(out, sample)

    def _attention(self):
        """Pin SDPA to the math backend under compile so inductor traces its clean composite backward; no-op eager."""
        return sdpa_kernel(SDPBackend.MATH) if self.config.runtime.compile_backbone else contextlib.nullcontext()

    def _losses_from(self, out: JointForward, sample: PairSample, *, detection: bool = True) -> _PairOutcome:
        """Every loss term for one pair from its already-computed forward (GT centres divided to the heads' grid).

        `detection=False` (generated hard scenes) zeroes the detection term: their blob appearance would corrupt
        a detector saturated on real data, while the association terms they carry teach the appearance-agnostic
        motion rule. The backbone still sees them through the edge gradient, which is ~30x smaller than det here.
        """
        loss_config = self.config.loss
        source_centres, target_centres, edge_matrix = sample.source_centres, sample.target_centres, sample.edge_matrix
        downsample = torch.tensor(self.config.data.downsample, dtype=torch.float32, device=str(out.edge_logits.device))
        centres_t_grid = (source_centres.to(torch.float32) / downsample).round().long()
        centres_t1_grid = (target_centres.to(torch.float32) / downsample).round().long()
        edge = SoftmaxFocalBCE.of(
            out.edge_logits, edge_matrix, loss_config.link_axes, balanced=loss_config.balanced_links
        )
        ignore = loss_config.ignore_ambiguous_above
        det = (
            BalancedBCE.of(out.detection_t.unsqueeze(0), [centres_t_grid], loss_config.neg_weight, ignore)
            + BalancedBCE.of(out.detection_t1.unsqueeze(0), [centres_t1_grid], loss_config.neg_weight, ignore)
            if detection
            else torch.zeros((), device=str(out.edge_logits.device))
        )
        contrastive = self.contrastive.forward(out.source_features, out.target_features, edge_matrix)
        decoys = HardNegativeMargin.mine(source_centres, target_centres, edge_matrix, loss_config.hard_negatives)
        hard_negative = HardNegativeMargin.of(out.edge_logits, edge_matrix, decoys)
        loss = (
            edge
            + loss_config.det_weight * det
            + loss_config.contrastive_weight * contrastive
            + loss_config.hard_negative_weight * hard_negative
        )
        ignored = self._ignored((out.detection_t, centres_t_grid), (out.detection_t1, centres_t1_grid))
        reliability = (
            SampleWeight.of(
                sample.frame_t,
                centres_t_grid,
                centres_t1_grid.to(torch.float32) * _ISOTROPIC_UM,  # isotropic µm on the heads' own grid
                centres_t_grid.to(torch.float32) * _ISOTROPIC_UM,
                _GATE_UM,
            )
            if loss_config.reliability_weighting
            else None
        )
        return _PairOutcome(
            loss, edge, det, contrastive, hard_negative, out.edge_logits, edge_matrix, ignored, reliability
        )

    def _autocast(self) -> torch.autocast:
        """The run's precision context — bf16 on CUDA, a no-op elsewhere; one definition for every forward here."""
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.runtime.device == "cuda")

    def _ignored(self, *frames: tuple[Tensor, Tensor]) -> float | None:
        """The pair's mean unsupervised fraction, or `None` when nothing is masked — a sigmoid, no backward."""
        above = self.config.loss.ignore_ambiguous_above
        if above is None:
            return None
        shares = [BalancedBCE.ignored_fraction(logits.unsqueeze(0), [centres], above) for logits, centres in frames]
        return sum(shares) / len(shares)

    def _evaluate(self, model: JointModel, evaluator: ModelEvaluator, val_targets: list[PairTarget]) -> _EvalResult:
        """Both trained heads through the shipped tracker on the proxy, plus their edge ranking on the val pairs."""
        model.eval()
        return _EvalResult(evaluator.evaluate_joint(model), self._edge_auc(model, val_targets))

    @torch.no_grad()
    def _edge_auc(self, model: JointModel, val_targets: list[PairTarget]) -> float:
        """Mean edge PRC-AUC over the held-out pairs — a COLLAPSE sanity check, not association quality.

        Scores are the pipeline's own: the logits soft-maxed over the sources of each target. Pairs whose
        annotation carries no link leave the average precision undefined and are skipped, not counted as zero.

        Read it as a floor alarm only. It ranks candidates among GROUND-TRUTH node pairs — where the shipped
        linker already sits at 0.9947 — so it is ~1.0 trained or not (a from-scratch run measured 0.6823 at
        random init and 0.9987 half an epoch later, then flat); only a head that stopped ranking moves it. The
        association number that can FAIL is the mislink inversion `ModelEvaluator` returns from the eval pass.
        """
        model.eval()
        device, data = self.config.runtime.device, self.config.data
        scored: list[float] = []
        for target in val_targets:
            corpus = PairDataset(
                [target], 1, data.downsample, self.config.runtime.seed, with_previous=data.prior_velocity
            )
            sample = corpus.pair(0).to(device)
            velocity = self._velocity.of(model, sample, self._autocast())
            with self._autocast():
                logits = model.forward(
                    sample.frame_t, sample.frame_t1, sample.source_centres, sample.target_centres, velocity
                ).edge_logits
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
        progress = self._checkpoint().restore(resume_path, model, self.contrastive, optimization)
        logger.info("resumed from %s at step %d (best %.4f)", resume_path, progress.done, progress.best)
        return progress.done, progress.best

    def _checkpoint(self) -> JointCheckpoint:
        """The run's persistence, carrying the shape a saved model is rebuilt from."""
        return JointCheckpoint(
            out_channels=self.config.model.out_channels,
            layers=tuple(self.config.model.layers),
            downsample=self.config.data.downsample,
            device=self.config.runtime.device,
        )


def main() -> None:
    args = JointCli.build_parser().parse_args()

    # The flat flags fan out into the concern each one belongs to — the CLI is the interface, this is the model.
    config = JointTrainConfig(
        model=ModelCfg(
            warm_pack=WARM_PACKS[args.warm_pack],
            temporal_position=args.temporal_position,
            lora=LoraCfg(
                enabled=args.lora, rank=args.lora_rank, alpha=args.lora_alpha, targets=tuple(args.lora_targets)
            ),
        ),
        data=DataCfg(
            detected_videos=args.detected_videos,
            difficulty_sampling=args.difficulty_sampling,
            prior_velocity=args.prior_velocity,
            synthetic_fraction=args.synthetic_fraction,
            synthetic_sequences=args.synthetic_sequences,
            synthetic_scenes=args.synthetic_scenes,
            gpu_scene_fraction=args.gpu_scene_fraction,
            gpu_scene_detection=args.gpu_scene_detection,
            paced_curriculum=args.paced_curriculum,
        ),
        optim=OptimCfg(lr=args.lr, cosine_lr=args.cosine_lr, batch_size=args.batch_size),
        loss=LossCfg(
            det_weight=args.det_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_site=args.contrastive_site,
            temperature=args.temperature,
            hard_negative_weight=args.hard_negative_weight,
            hard_negatives=args.hard_negatives,
            ignore_ambiguous_above=args.ignore_ambiguous_above,
            symmetric_links=args.symmetric_links,
            balanced_links=args.balanced_links,
            reliability_weighting=args.reliability_weighting,
        ),
        eval=EvalCfg(tracker=replace(TrackerConfig.shipped(), threshold=args.eval_threshold)),
        schedule=ScheduleCfg(
            steps=args.steps,
            eval_every=args.eval_every,
            patience=args.patience,
            epochs=args.epochs,
            evals_per_epoch=args.evals_per_epoch,
            patience_epochs=args.patience_epochs,
        ),
        runtime=RuntimeCfg(
            device=args.device,
            compile_backbone=args.compile_backbone,
            loader_threads=args.loader_threads,
            loader_prefetch=args.loader_prefetch,
        ),
    )
    root = DataRoot.from_config(_CONFIG)
    proc = root.processed(_DATASET)
    save_to = proc / args.weights
    log = Obs.setup(save_to.with_suffix(".log"), truncate=not args.resume)  # tail-able while the run goes
    recipe = DetectorRecipe(downsample=config.data.downsample, tta=config.eval.tta)
    tracker_config = config.eval.tracker
    with Obs.timed(log, "mounting the proxy evaluator"):
        validation = TestMovieProxy.load(root, VALIDATION_MOVIES)
        packs = (proc / WARM_PACKS["seed1"], proc / WARM_PACKS["seed2"])
        evaluator = ModelEvaluator.mount(validation, packs, recipe, config.runtime.device, tracker_config)

    pairs, split = PairSplit.assemble(root, config, evaluator.proxy, log, args.val_videos)

    setup = RunSetup(save_to, warm_start=args.warm_start, resume=args.resume, split=split)
    JointTrainer(config).train(pairs, evaluator, setup)


if __name__ == "__main__":
    main()
