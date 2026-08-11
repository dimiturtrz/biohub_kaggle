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

import argparse
import contextlib
import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import zarr
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltrack.data.difficulty_sampler import DifficultySampler
from celltrack.data.joint_dataset import PairDataset, PairSample, PairTarget
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.eval.model_evaluator import EvalResult, ModelEvaluator
from celltrack.eval.proxy import TEST_MOVIES, VALIDATION_MOVIES, TestMovieProxy
from celltrack.losses.balanced_bce import BalancedBCE
from celltrack.losses.hard_negative_margin import HardNegativeMargin
from celltrack.losses.softmax_focal_bce import SoftmaxFocalBCE
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from celltrack.operating_point import TrackerConfig
from celltrack.training.contrastive_term import ContrastiveTerm
from celltrack.training.early_stop import EarlyStop
from celltrack.training.joint_config import (
    WARM_PACKS,
    ContrastiveSite,
    DataCfg,
    EvalCfg,
    JointTrainConfig,
    LossCfg,
    ModelCfg,
    OptimCfg,
    RuntimeCfg,
    ScheduleCfg,
)
from celltrack.training.prior_velocity_source import PriorVelocitySource
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
# The proxy eval's tracker is the shipped one, whose recipe blends BOTH published seeds' packs (`WARM_PACKS`).
# Only the PROXY needs them: a joint run's own heads supply both the nodes and the affinity being selected on.
_HEARTBEAT_UPDATES = 100  # log within-window progress this often, so a long window isn't silent
_SECONDS_PER_HOUR = 3600.0
# The fresh transformer's shape, matching the pilkwang head the warm-start path loads.
_HIDDEN_DIM, _N_HEADS, _N_BLOCKS = 128, 4, 4
_POS_FEATURE_DIM = 4 * _POS_EMBED_DIM  # sinusoidal embed of (t, z, y, x)


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
    hard_negative: Tensor
    edge_logits: Tensor
    edge_matrix: Tensor
    ignored_fraction: float | None = None

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
        resume_path = save_to.with_suffix(".resume.pt")
        done, restored = self._restore(model, optimization, resume_path, resume=setup.resume)
        best = restored if restored is not None else self._initial_best(model, evaluator, pairs, run, save_to)

        stop = EarlyStop(schedule.patience, schedule.es_min_delta) if schedule.patience >= 1 else None
        if stop is not None:
            stop.update(best)  # seed with the init score so a first worse window doesn't read as an improvement
        run_start = time.perf_counter()
        sampler = self._sampler(pairs.train)
        while done < schedule.steps:
            window = min(schedule.eval_every, schedule.steps - done)
            dataset = PairDataset(
                pairs.train,
                window,
                self.config.data.downsample,
                self.config.runtime.seed + done,
                with_previous=self.config.data.prior_velocity,
            )
            losses = self._run_window(model, optimization, dataset, sampler)
            done += window
            result = self._evaluate(model, evaluator, pairs.val)
            pipeline = result.pipeline
            selected = pipeline.selection_score
            improved = stop.update(selected) if stop is not None else selected >= best
            marker = ""
            if improved:
                best, marker = selected, " *saved"
                self._save_checkpoint(save_to, model)
            self._save_resume(resume_path, model, optimization, done, best)
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
            if sampler is not None:
                logger.info("  sampler | coverage %.3f | entropy %.3f", losses["coverage"], losses["sampling_entropy"])
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
        self._save_checkpoint(save_to, model)
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
        self._velocity.widen(model)
        if self.config.runtime.device == "cuda":
            # channels_last_3d is a lossless layout that measured faster on the feature convs. torch stubs
            # omit the memory_format overload of Module.to; the call is runtime-valid.
            model = model.to(memory_format=torch.channels_last_3d)  # type: ignore[no-matching-overload]
        if self.config.runtime.compile_backbone:
            # Compile the backbone (static shape); the temporal-attention SDPA is forced to MATH at forward
            # time (see `_step`) — its efficient backward has a broken compiled meta-kernel.
            model.detector.unet = torch.compile(model.detector.unet, dynamic=False)  # type: ignore[bad-assignment]
        self.contrastive.to(self.config.runtime.device)
        optimizer = torch.optim.AdamW(self._trained(model), lr=self.config.optim.lr)
        return model, _Optimization(optimizer, self._schedule(optimizer))

    def _trained(self, model: JointModel) -> list[nn.Parameter]:
        """Every parameter the optimiser owns and the clip covers — both heads, plus the contrastive head.

        At the `FEATURES` site the contrastive term holds none, so the list IS `model.parameters()` in the
        same order: an unasked run's optimiser and gradient clip are the ones of yesterday, to the byte.
        """
        return [*model.parameters(), *self.contrastive.parameters()]

    @property
    def _velocity(self) -> PriorVelocitySource:
        """The run's history feed, reading the flag live — `train` re-resolves the config before the model exists."""
        return PriorVelocitySource(enabled=self.config.data.prior_velocity)

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

    def _sampler(self, targets: list[PairTarget]) -> DifficultySampler | None:
        """One sampler for the WHOLE run (difficulty has to survive across windows), or None for the uniform stream."""
        if not self.config.data.difficulty_sampling:
            return None
        seed = self.config.runtime.seed
        corpus = PairDataset(targets, self.config.schedule.steps, self.config.data.downsample, seed)
        return DifficultySampler(corpus.target_count, seed)

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

    def _pairs(
        self, dataset: PairDataset, sampler: DifficultySampler | None
    ) -> Iterator[tuple[int | None, PairSample]]:
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

    def _step(self, model: JointModel, optimization: _Optimization, pair: PairSample) -> _PairOutcome:
        """One optimisation step over a single GT pair; returns that pair's loss terms and edge tensors."""
        outcome = self._losses(model, pair)
        optimization.zero_grad()
        outcome.total.backward()
        nn.utils.clip_grad_norm_(self._trained(model), self.config.optim.grad_clip)
        optimization.step()
        return outcome

    def _losses(self, model: JointModel, pair: PairSample) -> _PairOutcome:
        """The loss tensors for one pair — shared by the training step and the no-grad eval.

        Positions reach the model at FULL resolution (it divides by the downsample for the feature grid,
        mirroring the inference scorer); the detection heads emit on the downsampled grid, so their GT
        centres are divided here to match.
        """
        device = self.config.runtime.device
        loss_config = self.config.loss
        sample = pair.to(device)
        velocity = self._velocity.of(model, sample, self._autocast())
        source_centres, target_centres, edge_matrix = sample.source_centres, sample.target_centres, sample.edge_matrix
        downsample = torch.tensor(self.config.data.downsample, dtype=torch.float32, device=device)
        centres_t_grid = (source_centres.to(torch.float32) / downsample).round().long()
        centres_t1_grid = (target_centres.to(torch.float32) / downsample).round().long()
        # Under compile, pin SDPA to the math backend so inductor traces its clean composite backward; no-op eager.
        compiled = self.config.runtime.compile_backbone
        attention = sdpa_kernel(SDPBackend.MATH) if compiled else contextlib.nullcontext()
        with attention, self._autocast():
            out = model.forward(sample.frame_t, sample.frame_t1, source_centres, target_centres, velocity)
            edge = SoftmaxFocalBCE.of(out.edge_logits, edge_matrix)
            ignore = loss_config.ignore_ambiguous_above
            det = BalancedBCE.of(
                out.detection_t.unsqueeze(0), [centres_t_grid], loss_config.neg_weight, ignore
            ) + BalancedBCE.of(out.detection_t1.unsqueeze(0), [centres_t1_grid], loss_config.neg_weight, ignore)
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
        return _PairOutcome(loss, edge, det, contrastive, hard_negative, out.edge_logits, edge_matrix, ignored)

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
        state = torch.load(resume_path, map_location=self.config.runtime.device, weights_only=False)
        model.detector.load_state_dict(state["detector"])
        model.transformer.load_state_dict(state["transformer"])
        # A snapshot written before the contrastive head existed carries no entry; its optimiser state has
        # no slot for one either, so a run resuming such a file is by construction one that projects nothing.
        projection = state.get("contrastive")
        if projection is not None:
            self.contrastive.load_state_dict(projection)
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
                "contrastive": self.contrastive.state_dict(),
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
                    "out_channels": self.config.model.out_channels,
                    "layers": list(self.config.model.layers),
                    "downsample": list(self.config.data.downsample),
                },
            },
            save_to,
        )


def _parser() -> argparse.ArgumentParser:
    """Every flag a joint run takes — kept apart from `main` so the run's WIRING reads as its own short function."""
    parser = argparse.ArgumentParser(description="Train the joint detector+edge model on the non-held-out videos.")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=500, help="steps per eval window")
    # The step scale above is a probe-run unit: one pair per step means the ~18000-pair corpus makes an epoch,
    # so `--steps 3000` is 17% of ONE pass against the published pack's 50 epochs, and `--patience` counts
    # EVAL WINDOWS — five of them is 14% of an epoch at eval-every 500 and five whole epochs at 9000. The
    # flags below say the same things in passes over the corpus, and win over their raw twin when given.
    parser.add_argument("--epochs", type=float, default=None, help="passes over the GT pairs; overrides --steps")
    parser.add_argument(
        "--evals-per-epoch", type=int, default=None, help="eval windows per epoch; overrides --eval-every"
    )
    parser.add_argument(
        "--patience-epochs",
        type=float,
        default=None,
        help="epochs without a gain before stopping; overrides --patience",
    )
    parser.add_argument("--warm-start", action="store_true", help="initialise from the published pilkwang weights")
    parser.add_argument("--det-weight", type=float, default=1.0, help="weight on the detection term vs the edge term")
    parser.add_argument(
        "--contrastive-weight", type=float, default=0.0, help="weight on the InfoNCE term over the node features"
    )
    # WHERE that weight acts. On the features it is measured to trade detection away for discrimination
    # (proxy 0.8414 -> 0.7292, node recall 0.966 -> 0.867); the projecting sites give the term its own
    # embedding to shape, `detached_projection` keeping the backbone out of its reach entirely.
    parser.add_argument(
        "--contrastive-site",
        choices=tuple(ContrastiveSite),
        default=ContrastiveSite.FEATURES,
        type=ContrastiveSite,
        help="where the InfoNCE term acts: on the shared features, or through its own projection head",
    )
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE softmax temperature over targets")
    # The zni probe's mining under a MARGIN objective and an UNFROZEN backbone — the one combination never run.
    # Its absolute form (decoy probability -> 0, UNet frozen) flattened the head; a weight here is matched to the
    # term's own logged magnitude against the edge term, so the arm measures the mechanism and not a rescaling.
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=0.0,
        help="weight on the ranking term holding each source's true logit above its nearest wrong targets'",
    )
    parser.add_argument(
        "--hard-negatives", type=int, default=4, help="nearest wrong targets mined per annotated source"
    )
    # Fine-tuning a CONVERGED model at its original training rate is the classic way to walk off its optimum,
    # which is what 3000 warm-start steps at 1e-4 did (never beat the init). Exposed so the rate is a
    # variable of the experiment rather than an inherited constant.
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate; lower it when warm-starting")
    parser.add_argument("--cosine-lr", action="store_true", help="decay the rate to zero over the run")
    parser.add_argument(
        "--warm-pack", choices=tuple(WARM_PACKS), default="seed1", help="which published pack to continue"
    )
    parser.add_argument("--compile-backbone", action="store_true", help="torch.compile the U-Net (static shape)")
    parser.add_argument(
        "--difficulty-sampling",
        action="store_true",
        help="draw pairs in proportion to their measured top-1 defect instead of uniformly with replacement",
    )
    parser.add_argument(
        "--prior-velocity",
        action="store_true",
        help="feed each source its t-1 -> t displacement as head input (~50%% slower per step)",
    )
    # The threshold is DERIVED, not swept: given bare, the flag takes the pipeline's own inference operating
    # point — the response at which the shipped tracker would already have called the voxel a cell, so every
    # voxel the mask spares is one the deployed model detects and our sparse annotation cannot adjudicate.
    parser.add_argument(
        "--ignore-ambiguous-above",
        type=float,
        nargs="?",
        const=TrackerConfig().threshold,
        default=None,
        help="leave unannotated voxels above this sigmoid response unsupervised (bare = the tracker threshold)",
    )
    parser.add_argument("--patience", type=int, default=5, help="stop after N non-improving evals (<1 disables)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-videos", type=int, default=2, help="validation movies the edge AUC is measured over")
    parser.add_argument("--eval-threshold", type=float, default=0.5, help="detection threshold of the selector eval")
    parser.add_argument("--resume", action="store_true", help="continue from the .resume.pt snapshot")
    parser.add_argument("--weights", type=str, default="joint_tunet_ours.pt")
    return parser


def main() -> None:
    args = _parser().parse_args()

    # The flat flags fan out into the concern each one belongs to — the CLI is the interface, this is the model.
    config = JointTrainConfig(
        model=ModelCfg(warm_pack=WARM_PACKS[args.warm_pack]),
        data=DataCfg(difficulty_sampling=args.difficulty_sampling, prior_velocity=args.prior_velocity),
        optim=OptimCfg(lr=args.lr, cosine_lr=args.cosine_lr),
        loss=LossCfg(
            det_weight=args.det_weight,
            contrastive_weight=args.contrastive_weight,
            contrastive_site=args.contrastive_site,
            temperature=args.temperature,
            hard_negative_weight=args.hard_negative_weight,
            hard_negatives=args.hard_negatives,
            ignore_ambiguous_above=args.ignore_ambiguous_above,
        ),
        eval=EvalCfg(threshold=args.eval_threshold),
        schedule=ScheduleCfg(
            steps=args.steps,
            eval_every=args.eval_every,
            patience=args.patience,
            epochs=args.epochs,
            evals_per_epoch=args.evals_per_epoch,
            patience_epochs=args.patience_epochs,
        ),
        runtime=RuntimeCfg(device=args.device, compile_backbone=args.compile_backbone),
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
    recipe = DetectorRecipe(downsample=config.data.downsample, tta=config.eval.tta)
    tracker_config = TrackerConfig(threshold=config.eval.threshold)
    with Obs.timed(log, "mounting the proxy evaluator"):
        validation = TestMovieProxy.load(root, VALIDATION_MOVIES)
        packs = (proc / WARM_PACKS["seed1"], proc / WARM_PACKS["seed2"])
        evaluator = ModelEvaluator.mount(validation, packs, recipe, config.runtime.device, tracker_config)

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
