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

The loop mirrors the detector trainer — eval-sized windows, EarlyStop + save-best + resume, bf16 /
channels_last / optional torch.compile — and selects on the same LB-aligned number: the `ModelEvaluator` score
of BOTH heads through the shipped `CellTracker` on the densely-annotated VALIDATION movies
(`celltrack.eval.proxy`), TEST four held out of training and selection as a clean leaderboard estimate.
Selection reads the CLAMPED form (`SplitScore.selection_score`) — the faithful metric pays an under-detection
bonus our sparse proxy does not but the dense hidden eval does — logged as `proxy X (sel Y)`. Warm-starting
(`--warm-start`) continues pilkwang's heads; from scratch is the honest own-trained baseline.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltrack.data.difficulty_sampler import DifficultySampler
from celltrack.data.gpu_scene import GpuScenes
from celltrack.data.joint_dataset import PairDataset, PairOptions, PairSample, PairTarget
from celltrack.data.pair_curriculum import FixedMixture, PacedMixture, PairCurriculum, PairDraw
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.eval.model_evaluator import EvalResult, ModelEvaluator
from celltrack.eval.proxy import VALIDATION_MOVIES, TestMovieProxy
from celltrack.losses.association_objective import AssociationObjective
from celltrack.losses.tracking_loss import (
    BatchedPrediction,
    PairGroundTruth,
    PairPrediction,
    TrackingLoss,
    _PairOutcome,
)
from celltrack.models.joint_model import BatchedForward, JointForward, JointModel
from celltrack.training.base_calibration import BaseCalibration
from celltrack.training.contrastive_term import ContrastiveTerm, ContrastiveTermConfig
from celltrack.training.early_stop import EarlyStop
from celltrack.training.joint_assembly import JointAssembly
from celltrack.training.joint_checkpoint import RESUME_SUFFIX, JointCheckpoint, RunProgress
from celltrack.training.joint_cli import JointCli
from celltrack.training.joint_config import (
    WARM_PACKS,
    JointTrainConfig,
)
from celltrack.training.pair_split import PairSplit
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
# The uniform stream draws no index at all; -1 says so without a `int | None` that every reader must unwrap.
_UNTRACKED_INDEX = -1


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
        self.contrastive = ContrastiveTerm(ContrastiveTermConfig.from_loss(config.loss, config.model.out_channels))
        # The link objective's form is fixed for the run and a slack row carries a learned scalar the optimiser
        # must own, so — like the contrastive term — it is built once here from the config that chose it.
        self._objective = AssociationObjective.from_config(
            link_axes=config.loss.link_axes,
            balanced_links=config.loss.balanced_links,
            slack_links=config.loss.slack_links,
        )
        # Every training term; the trainer owns only the step. The loss reads its own config and input structs, so
        # it imports nothing of the model or the dataset — the trainer repacks each forward at the call site.
        self._loss = TrackingLoss(
            config.loss.to_tracking_loss_config(config.data.downsample), self.contrastive, self._objective
        )
        # The build phase (model + optimiser + feature installs) and the velocity feed live in one object beside
        # the config that chose them; `train` rebuilds it once the epoch budget is resolved.
        self._assembly = JointAssembly(config, self.contrastive, self._objective)
        self._velocity = self._assembly.velocity

    @staticmethod
    def _prediction(out: JointForward) -> PairPrediction:
        """The model's forward repacked into the loss's own input — the seam that keeps the loss off the model type."""
        return PairPrediction(
            out.edge_logits, out.detection_t, out.detection_t1, out.source_features, out.target_features
        )

    @staticmethod
    def _ground_truth(sample: PairSample) -> PairGroundTruth:
        """The dataset's sample repacked into the loss's own target — the seam that keeps the loss off the data type."""
        return PairGroundTruth(sample.source_centres, sample.target_centres, sample.edge_matrix, sample.frame_t)

    @staticmethod
    def _batched_prediction(batched: BatchedForward) -> BatchedPrediction:
        """The padded batched forward repacked for the loss — the `(B, ...)` blocks plus the per-pair unpadded views."""
        return BatchedPrediction(
            batched.logits,
            batched.detect_t,
            batched.detect_t1,
            batched.source_mask,
            batched.target_mask,
            [JointTrainer._prediction(forward) for forward in batched.unpad()],
        )

    def train(self, pairs: PairSplit, evaluator: ModelEvaluator, setup: RunSetup) -> float:
        """Train (or fine-tune) on the GT pairs, saving the best SELECTION score. Returns that best score.

        The loop runs in eval-sized windows streaming `eval_every` GT pairs one at a time (ragged node counts,
        no cross-pair batching); a full resume snapshot is written every window and ``setup.resume`` continues it.
        One pair per step makes the corpus the epoch, so the schedule is resolved here — the first moment the
        pair count is known — and every later reader sees steps.
        """
        steps_per_epoch = max(1, len(pairs.train))
        self.config = self.config.resolved(steps_per_epoch)
        self._assembly = JointAssembly(self.config, self.contrastive, self._objective)  # rebuild on the resolved budget
        self._velocity = self._assembly.velocity
        schedule = self.config.schedule
        save_to = setup.save_to
        torch.set_float32_matmul_precision("high")  # TF32 on the fp32 matmuls the bf16 autocast leaves alone
        torch.backends.cudnn.benchmark = True  # one static frame shape — cudnn picks the fastest algo once
        model, optimization = self._assembly.build(
            warm_start=setup.warm_start,
            init_weights=setup.init_weights,
            detector_from=setup.detector_from,
            spacing=pairs.spacing,
        )

        run = TrainingRun.open(self.config.model_dump(), setup)
        resume_path = save_to.with_suffix(RESUME_SUFFIX)
        done, restored = self._restore(model, optimization, resume_path, resume=setup.resume)
        best = restored if restored is not None else self._initial_best(model, evaluator, pairs, run, setup)

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
                PairOptions(with_previous=self.config.data.prior_velocity, augmentation=self.config.augmentation()),
            )
            losses = self._run_window(model, optimization, dataset, curriculum, step_base=done)
            done += window
            result = self._evaluate(model, evaluator, pairs.val)
            pipeline = result.pipeline
            # The configured objective (proxy by default; confusor margin above the ceiling) weighted by the
            # difficulty the window TRAINED at — a policy moving the diet must not be judged by a metric that
            # forgot it moved.
            base = self._selection_base(pipeline)
            selected = base * (curriculum.selection_weight() if curriculum else 1.0)
            if curriculum is not None:
                curriculum.update(pipeline.selection_score)
            improved = stop.update(selected) if stop is not None else selected >= best
            marker = ""
            if improved:
                best, marker = selected, " *saved"
                self._checkpoint().save_best(save_to, model)
            progress = RunProgress(done=done, best=best)
            self._checkpoint().save_snapshot(resume_path, model, self._auxiliary(), optimization, progress)
            run.window(done, {**self._metrics(result, best), **losses})
            rate = losses["it_per_s"]
            eta_hours = (schedule.steps - done) / rate / _SECONDS_PER_HOUR if rate else 0.0
            logger.info(
                "epoch %g/%g | step %5d/%d | train loss %.4f (edge %.4f det %.4f nce %.4f hn %.4f) | "
                "proxy %.4f (sel %.4f) | "
                "best %.4f%s | node R %.3f ratio %+.2f | mislinks %d inv %.2f P true %.3f vs chosen %.3f top1 %.3f | "
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
                pipeline.top1_fraction,
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
        self, model: JointModel, evaluator: ModelEvaluator, pairs: PairSplit, run: TrainingRun, setup: RunSetup
    ) -> float:
        """The score before a step — the bar a warm start must beat (window 0), and the base-calibration gate:
        a warm/chained run onto a miscalibrated base is aborted here, not read as a false null after training."""
        initial = self._evaluate(model, evaluator, pairs.val)
        logger.info(
            "init proxy %.4f (sel %.4f) | node R %.3f ratio %+.2f | mislinks %d inv %.2f P true %.3f vs chosen %.3f"
            " top1 %.3f | sanity AUC %.4f",
            initial.pipeline.score,
            initial.pipeline.selection_score,
            initial.pipeline.node_recall,
            initial.pipeline.node_ratio,
            initial.pipeline.mislinks,
            initial.pipeline.inverted_fraction,
            initial.pipeline.mean_p_true,
            initial.pipeline.mean_p_chosen,
            initial.pipeline.top1_fraction,
            initial.edge_auc,
        )
        # The band is grid-aware (BaseCalibration.upper_band): a finer decode over-detects by design, so its warm
        # base sits at a positive ratio the (1,4,4) +-0.5 band would false-reject. detector_from is a warm base
        # like any other now — its FROZEN detector's over-detection is exactly what the grid-scaled band admits.
        _, dy, dx = self.config.data.downsample
        expects_calibrated_base = setup.warm_start or setup.init_weights is not None or setup.detector_from is not None
        BaseCalibration.gate(
            initial.pipeline.node_ratio, expects_calibrated_base=expects_calibrated_base, inplane_area=dy * dx
        )
        initial_best = self._selection_base(initial.pipeline)
        run.window(0, self._metrics(initial, initial_best))
        self._checkpoint().save_best(setup.save_to, model)
        return initial_best

    def _selection_base(self, pipeline: EvalResult) -> float:
        return self.config.selection_scalar(pipeline.selection_score, pipeline.mean_p_true, pipeline.mean_p_chosen)

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
            "top1_fraction": result.pipeline.top1_fraction,
            "mislinks": result.pipeline.mislinks,
        }

    def _auxiliary(self) -> dict[str, nn.Module]:
        """The run's auxiliary trained modules by snapshot key (contrastive head, link objective) — one home so
        saving and restoring cannot disagree on either name."""
        return {"contrastive": self.contrastive, "objective": self._objective}

    def _curriculum(self, pairs: PairSplit) -> PairCurriculum | None:
        """The run's ONE policy over (which pair, how much it counts), or None for the uniform stream.

        Built once because every policy carries state across windows (a difficulty estimate, an EMA, a share).
        The three are mutually exclusive: paced needs a synthetic population, fixed a share of one, the sampler
        neither.
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
        step_base: int = 0,
    ) -> dict[str, float]:
        """Train one optimiser step per pair of `dataset`; returns each term's window mean plus `it_per_s`.

        `step_base` is the global step this window starts at — added to the window-local count so a step-scheduled
        feature (the velocity GT warmup) sees the run's true progress, not a per-window counter that resets.
        """
        model.train()
        if self.config.model.freeze_backbone_norm:  # re-assert after train() flips the whole tree back to train mode
            self._assembly.freeze_backbone_norm(model)
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
            outcomes = self._step_batch(
                model, optimization, batch + self._gpu_batch(gpu_count), len(batch), step=step_base + done
            )
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
        yield from self._prefetched(dataset, curriculum, curriculum.read_ahead())

    def _prefetched(
        self, dataset: PairDataset, curriculum: PairCurriculum, depth: int
    ) -> Iterator[tuple[PairDraw, PairSample]]:
        """Curriculum draws with up to `depth` NEXT pairs' decode overlapped under the current step's compute.

        The synchronous curriculum path idled the GPU ~one frame-decode per step: a sampler draws the next
        index from feedback on the previous step, so the uniform stream's read-ahead was disabled to keep that
        feedback fresh. `read_ahead()` is each policy's own statement of how stale a draw it tolerates — 0 when a
        per-pair `observe` moves the next draw (`DifficultySampler`), ≥1 for the window-fed mixtures whose draw
        sequence is byte-identical however far ahead it is read. A single worker per depth keeps at most `depth`
        pairs in flight, so `dataset.pair` is never called concurrently with itself at depth 1.
        """
        steps = len(dataset)
        if depth < 1:
            for step in range(steps):
                draw = curriculum.step()
                yield draw, dataset.pair(draw.index, augment_step=step)
            return
        with ThreadPoolExecutor(max_workers=depth) as pool:
            pending: deque[tuple[PairDraw, Future[PairSample]]] = deque()
            drawn = 0
            while drawn < steps and len(pending) < depth:
                draw = curriculum.step()
                pending.append((draw, pool.submit(dataset.pair, draw.index, augment_step=drawn)))
                drawn += 1
            while pending:
                draw, decoded = pending.popleft()
                if drawn < steps:
                    ahead = curriculum.step()
                    pending.append((ahead, pool.submit(dataset.pair, ahead.index, augment_step=drawn)))
                    drawn += 1
                yield draw, decoded.result()

    def _observe(self, curriculum: PairCurriculum | None, index: int, outcome: _PairOutcome) -> None:
        """Close the per-pair feedback loop — the step's own top-1 defect. A no-op on the uniform path."""
        if curriculum is None or index == _UNTRACKED_INDEX:
            return
        curriculum.observe(index, outcome.difficulty())

    def _step_batch(
        self,
        model: JointModel,
        optimization: _Optimization,
        batch: list[tuple[PairDraw, PairSample]],
        real_count: int,
        step: int = 0,
    ) -> list[_PairOutcome]:
        """One step over a batch — backbone forwarded ONCE over stacked windows, gradient the batch MEAN.

        The first `real_count` pairs are real and train BOTH heads. Generated scenes beyond train association
        always, and detection too when `gpu_scene_detection` is set — their 100% labels are the honest detection
        signal the real corpus (~98% of cells supervised as background) lacks, judged on the LB not sparse recall.
        """
        device = self.config.runtime.device
        scene_detection = self.config.data.gpu_scene_detection
        samples = [pair.to(device) for _, pair in batch]
        velocities = self._velocity.of_batch(model, samples, self._autocast(), step=step)
        if self._batched_eligible(samples, real_count, len(batch)):
            return self._batched_step(model, optimization, batch, samples, velocities)
        windows = torch.stack([torch.stack([s.frame_t, s.frame_t1], dim=0) for s in samples], dim=0)  # (B, 2, Z, Y, X)
        with self._attention(), self._autocast():
            forwards = model.forward_batch(
                windows, [s.source_centres for s in samples], [s.target_centres for s in samples], velocities
            )
            outcomes = [
                self._loss.per_pair(
                    self._prediction(out),
                    self._ground_truth(sample),
                    detection=index < real_count or scene_detection,
                )
                for index, (out, sample) in enumerate(zip(forwards, samples, strict=True))
            ]
        weights = self._loss.reliability_weights(outcomes)
        terms = [
            draw.weight * weight * out.total for (draw, _), out, weight in zip(batch, outcomes, weights, strict=True)
        ]
        (torch.stack(terms).sum() / len(batch)).backward()
        nn.utils.clip_grad_norm_(self._assembly.trained(model), self.config.optim.grad_clip)
        optimization.step()
        optimization.zero_grad()
        return outcomes

    def _batched_eligible(self, samples: list[PairSample], real_count: int, batch_len: int) -> bool:
        """Whether the batch takes the VECTORISED loss — the pure-default config, and the only one a graph captures.

        The batched loss reads the padded block for the edge and detection terms; anything that needs a per-pair
        quantity (a generated scene's zeroed detection, or an optional contrastive / hard-negative / reliability
        term) routes to the per-pair loop, which stays the reference implementation. Prior velocity now rides the
        batched block too — `batched_forward` threads the padded step through, so a velocity run is no longer forced
        onto the per-pair loop.

        Generated scenes stay on the per-pair loop even under `gpu_scene_detection` (where the loss WOULD equal the
        batched sum, `test_batched_equals_per_pair`): a crowded scene carries far more nodes than `_MAX_NODES`, and
        the batched path pads to that fixed block, so it raises where the ragged per-pair loop simply handles them.
        An oversized REAL pair (a dense corpus, a raised batch default) falls back the same way — `fits_batched`
        keeps the padded path from ever meeting a pair it cannot pad, so the default batch size can be raised safely.
        """
        loss = self.config.loss
        return (
            real_count == batch_len  # no generated scenes (batched pads to _MAX_NODES; a crowded scene overflows it)
            and all(JointModel.fits_batched(s.source_centres.shape[0], s.target_centres.shape[0]) for s in samples)
            and loss.contrastive_weight == 0.0
            and loss.hard_negative_weight == 0.0
            and not loss.reliability_weighting
        )

    def _batched_step(
        self,
        model: JointModel,
        optimization: _Optimization,
        batch: list[tuple[PairDraw, PairSample]],
        samples: list[PairSample],
        velocities: list[Tensor | None],
    ) -> list[_PairOutcome]:
        """One step with the loss VECTORISED over the padded batch — no per-pair Python loop over ragged logits.

        The edge and detection terms read the padded `(B, _MAX_NODES, ...)` block directly (`SoftmaxFocalBCE.
        batched`, `BalancedBCE.per_frame`), so the whole GPU forward-and-loss is batched tensor ops the compiler
        can graph. The per-pair outcomes returned for the sampler and the logs are packaged from the padded
        result AFTER the step — detached diagnostics, off the gradient path.
        """
        device = self.config.runtime.device
        windows = torch.stack([torch.stack([s.frame_t, s.frame_t1], dim=0) for s in samples], dim=0)
        with self._attention(), self._autocast():
            batched = model.batched_forward(
                windows, [s.source_centres for s in samples], [s.target_centres for s in samples], velocities
            )
            total, outcomes = self._loss.batched(  # (B,) for backward + per-pair diagnostics
                self._batched_prediction(batched), [self._ground_truth(sample) for sample in samples]
            )
            draw_weight = torch.tensor([draw.weight for draw, _ in batch], device=device, dtype=total.dtype)
            (draw_weight * total).sum().div(len(batch)).backward()
        nn.utils.clip_grad_norm_(self._assembly.trained(model), self.config.optim.grad_clip)
        optimization.step()
        optimization.zero_grad()
        return outcomes

    def _gpu_batch(self, count: int) -> list[tuple[PairDraw, PairSample]]:
        """`count` FRESH GPU-generated hard scene-pairs — untracked (they shape the gradient, not the feedback).

        The generator persists across windows (lazily created, seeded once) so every batch draws a NEW crowded
        population rather than replaying a pool — the whole point of generating on the GPU rather than loading.
        """
        if count <= 0:
            return []
        if not hasattr(self, "_scene_source"):
            device = self.config.runtime.device
            object.__setattr__(self, "_scene_source", GpuScenes(self.config.scene_config(), device))
            object.__setattr__(self, "_scene_rng", torch.Generator(device).manual_seed(self.config.runtime.seed))
        pairs = self._scene_source.batch(count, self._scene_rng)  # type: ignore[attr-defined]
        return [(PairDraw(index=_UNTRACKED_INDEX), pair) for pair in pairs]

    def _losses(self, model: JointModel, pair: PairSample) -> _PairOutcome:
        """One pair's loss tensors — the per-pair forward then `_losses_from`; used by the no-grad eval."""
        sample = pair.to(self.config.runtime.device)
        velocity = self._velocity.of(model, sample, self._autocast())
        with self._attention(), self._autocast():
            out = model.forward(sample.frame_t, sample.frame_t1, sample.source_centres, sample.target_centres, velocity)
            return self._loss.per_pair(self._prediction(out), self._ground_truth(sample))

    def _attention(self):
        """Pin SDPA to the math backend — the backbone is always compiled, and inductor needs the clean composite
        backward (the efficient SDPA backward has a broken compiled meta-kernel)."""
        return sdpa_kernel(SDPBackend.MATH)

    def _autocast(self) -> torch.autocast:
        """The run's precision context — bf16 on CUDA, a no-op elsewhere; one definition for every forward here."""
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.config.runtime.device == "cuda")

    def _evaluate(self, model: JointModel, evaluator: ModelEvaluator, val_targets: list[PairTarget]) -> _EvalResult:
        """Both trained heads through the shipped tracker on the proxy, plus their edge ranking on the val pairs."""
        model.eval()
        return _EvalResult(evaluator.evaluate_joint(model), self._edge_auc(model, val_targets))

    @torch.no_grad()
    def _edge_auc(self, model: JointModel, val_targets: list[PairTarget]) -> float:
        """Mean edge PRC-AUC over the held-out pairs — a COLLAPSE sanity check, not association quality.

        Scores are the pipeline's own (logits soft-maxed over each target's sources); no-link pairs are skipped,
        not zeroed. Read it as a floor alarm only: it ranks among GROUND-TRUTH node pairs where the linker sits at
        0.9947, so it is ~1.0 trained or not (0.6823 at random init, 0.9987 half an epoch later, then flat) — only
        a head that stopped ranking moves it. The number that can FAIL is the mislink inversion from `ModelEvaluator`.
        """
        model.eval()
        device, data = self.config.runtime.device, self.config.data
        scored: list[float] = []
        for target in val_targets:
            corpus = PairDataset(
                [target], 1, data.downsample, self.config.runtime.seed, PairOptions(with_previous=data.prior_velocity)
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
        progress = self._checkpoint().restore(resume_path, model, self._auxiliary(), optimization)
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
    # Defragment the CUDA caching allocator before the first allocation: the proxy-eval tracker's large
    # transient tensors on big validation movies otherwise fragment the default allocator into a >32GB
    # footprint that spills to host RAM. Expandable segments return freed blocks, holding the same pipeline
    # under ~21GB. setdefault so an explicit env override still wins; read lazily at first CUDA alloc.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = JointCli.build_parser().parse_args()
    config = JointTrainConfig.from_args(args)
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

    init_weights = proc / args.init_weights if args.init_weights else None
    detector_from = proc / args.detector_from if args.detector_from else None
    setup = RunSetup(
        save_to,
        warm_start=args.warm_start,
        resume=args.resume,
        split=split,
        init_weights=init_weights,
        detector_from=detector_from,
    )
    JointTrainer(config).train(pairs, evaluator, setup)


if __name__ == "__main__":
    main()
