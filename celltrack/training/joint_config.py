"""Every knob of a joint fine-tune, grouped by CONCERN and validated at construction.

One flat bag of twenty-five fields cannot say which knob belongs with which; this hierarchy does, the way
`TrackerConfig` already carries `LinkerConfig`/`BridgeConfig`. Each group is a pydantic model with real
bounds, because a config crosses a trust boundary — CLI flags and a loaded JSON are user input, and a
negative temperature or a probability of 3 should fail AT CONSTRUCTION, not deep inside a loss. Being
pydantic also makes the whole run one JSON document (`model_dump_json` / `model_validate_json`) and one
mlflow param table (`model_dump()` flattens to dotted keys in `core.tracking.Tracker._flat`).

The CLI stays FLAT — `--det-weight` is the user interface; the nesting is the model.

Every default here is the value the trainer ALREADY ran with, with the instrument that arbitrated it named
beside it where one did. A default nobody chose is not a decision anyone made: the published recipe's own
numbers (lr 1e-3, det-weight 1e1, threshold 0.99) are refuted by our measurements and are deliberately NOT
what this file ships.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from celltrack.losses.softmax_focal_bce import SOURCE_AXIS, TARGET_AXIS
from celltrack.models.lora import LoraConfig
from celltrack.operating_point import TrackerConfig

logger = logging.getLogger(__name__)

# Immutable + closed: a typo is a construction error, and a config cannot drift under a running loop.
_VALIDATED = ConfigDict(frozen=True, extra="forbid")

# The published pilkwang packs, under the (gitignored) data root's reference area — the warm-start sources.
# Seed 2 is also the proxy tracker's second seed, so the pair is named once here and read by both.
WARM_PACKS = {
    "seed1": Path("reference") / "pilkwang" / "split_0",
    "seed2": Path("reference") / "pilkwang" / "seed2" / "weights" / "unet_transformer" / "split_0",
}

# Each epoch-scale field and the step-scale field it replaces when set, so an override is logged, never silent.
_EPOCH_OVERRIDES = (("epochs", "steps"), ("evals_per_epoch", "eval_every"), ("patience_epochs", "patience"))


class LoraCfg(BaseModel):
    """Low-rank adapters over a FROZEN base — off by default, byte-identical to the parent when off.

    When on, `_prepare` freezes the whole warm-started model and injects rank-`rank` deltas into every leaf
    whose dotted path contains one of `targets`. The default targets the association locus: the whole edge
    transformer plus the U-Net decoder whose features it reads — the mislink signal is in the FEATURES, not the
    head alone (a1v/zni), so a head-only adaptation is expected to under-move it.
    """

    model_config = _VALIDATED

    enabled: bool = False
    rank: PositiveInt = 16
    alpha: float = Field(16.0, gt=0)
    targets: tuple[str, ...] = ("transformer", "decoder_blocks")

    def to_config(self) -> LoraConfig:
        """The `celltrack.models.lora` config this pydantic view configures — the trainer injects from it."""
        return LoraConfig(rank=self.rank, alpha=self.alpha, targets=self.targets)


class ModelCfg(BaseModel):
    """The object being trained: the backbone's shape, and which published pack a warm start continues."""

    model_config = _VALIDATED

    out_channels: PositiveInt = 32  # the published feature width the mounted transformer head expects
    layers: tuple[PositiveInt, ...] = (32, 64, 128)  # the published U-Net stage widths
    lora: LoraCfg = Field(default_factory=LoraCfg)  # frozen-base low-rank adapters; off = full model as before
    # WHICH published pack to continue. Warm-starting from seed1 produces that seed's descendant, and a
    # descendant adds nothing to an ensemble containing its parent (measured: +0.0075 alone, but 0.9315
    # against the pair's 0.9334). Training the same recipe from a DIFFERENT parent is how the result earns
    # a place — ensembles pay for disagreement, not for quality alone.
    warm_pack: Path = WARM_PACKS["seed1"]
    # Add a learned frame-position embedding to the backbone's per-voxel time attention — the one capability
    # the order-blind published backbone cannot be configured into. Off = the published attention, byte for
    # byte. On, the model can represent WHICH frame came first, i.e. a direction; it must LEARN to use it, so
    # this belongs to a long run, not the short warm-start optimum where a zero-initialised input never grows.
    temporal_position: bool = False


class DataCfg(BaseModel):
    """The input engine: how a frame is read, which pair the next step draws, and what rides along with it."""

    model_config = _VALIDATED

    downsample: tuple[PositiveInt, PositiveInt, PositiveInt] = (1, 4, 4)  # (z, y, x) — the published x4 in-plane
    # Draw each step's pair in proportion to its measured top-1 defect instead of uniformly with replacement
    # (see `DifficultySampler`). Off by default: with it off the loop is the uniform stream of yesterday.
    difficulty_sampling: bool = False
    # How many train videos to build the DETECTED-neighbourhood corpus over (0 = off, use the annotated pairs).
    # When set, those pairs REPLACE the annotated ones rather than mixing with them: the annotated corpus is
    # what carries the defect (0.99 in-gate candidates per source, 1.9% of sources contested, against 3.80 and
    # 95.4% at inference), so mixing it back in re-introduces exactly the uncontested rows this corpus exists
    # to escape — and the synthetic arm already measured mixing as dilution rather than curriculum.
    detected_videos: int = Field(0, ge=0)
    # Feed each source its PRIOR displacement (t-1 -> t) as three extra head-input columns, so the pair head can
    # ask whether a candidate continues the motion the cell was already on — the axis behind our documented
    # association failure (a fast mover's true successor lands FAR while a slower neighbour sits NEAR, and both
    # distance and the learned probability pick the near-wrong cell). Off by default and byte-identical off.
    # COSTS ~50% more wall-clock per step: an extra frame decode plus one extra no-grad backbone forward.
    prior_velocity: bool = False
    # Share of steps drawn from the FULLY-LABELLED synthetic corpus (`celltrack.data.synthetic_pairs`), whose
    # candidate set is complete by construction — the property our own annotation cannot have, and the one the
    # measured association failure turns on (all 38 dense mislinks chose an UNANNOTATED partner). 0.0 keeps the
    # corpus of yesterday; the synthetic pairs are not even enumerated unless this or `paced_curriculum` asks.
    synthetic_fraction: float = Field(0.0, ge=0.0, lt=1.0)
    # How many synthetic sequences to enumerate (each is 6 frames = 5 pairs); None takes all 2174 of them.
    synthetic_sequences: PositiveInt | None = None
    # How many hard scenes to GENERATE dynamically (crowded fast-movers with a known answer, the dense-hard
    # SceneConfig default) instead of loading the static npz. >0 makes the synthetic corpus the generator's; the
    # mix into real pairs is still `synthetic_fraction`. This is the contested signal the real corpus lacks.
    synthetic_scenes: int = Field(0, ge=0)
    # Fraction of EACH training batch generated fresh on the GPU per step (dynamic, no pool to memorise, ~4ms/8).
    # 0 = off; 0.5 = half of every batch is fresh hard scenes. The complement is drawn from the real corpus.
    gpu_scene_fraction: float = Field(0.0, ge=0.0, lt=1.0)
    # Whether generated scenes also train the DETECTION head. True is the HONEST signal real data lacks — a scene
    # labels 100% of its cells where the real corpus supervises ~98% AS background — so it counteracts, not
    # poisons, the detection-loss-punishes-good-detectors bias; annotated-recall may still read down because the
    # sparse eval only sees ~1-2% of cells. False trains association only (blob appearance never touches the head).
    gpu_scene_detection: bool = True
    # Replace the FIXED share above with the feedback controller (`PacedMixture`): the two populations
    # alternate structurally and an EMA'd difficulty moves a per-sample LOSS WEIGHT instead of a draw
    # probability, with save-best corrected for the difficulty the window trained at. Off by default.
    paced_curriculum: bool = False


class OptimCfg(BaseModel):
    """The update rule: AdamW's rate, the gradient ceiling, and whether the rate decays over the run."""

    model_config = _VALIDATED

    # Fine-tuning a CONVERGED model at its original training rate is the classic way to walk off its optimum,
    # which is what 3000 warm-start steps at 1e-4 did (never beat the init) — a warm-start arm passes 1e-5.
    # This is the from-scratch default the trainer already ran; the published recipe's 1e-3 is NOT it.
    lr: float = Field(1e-4, gt=0)
    grad_clip: float = Field(1.0, gt=0)
    # Pairs FORWARDED TOGETHER in one backbone pass and averaged into one optimiser step. A pair's target counts
    # are ragged, so the HEADS run per pair, but the backbone — the GPU cost — takes any batch dimension, so
    # stacking B pairs' windows fills the device where a per-pair forward left it ~40% idle for the whole
    # campaign. The gradient is the mean over the batch, so `lr` keeps its meaning across sizes. 1 reproduces the
    # per-pair update exactly (mean of one, step on every pair), so the old single-pair arms stay comparable; a
    # larger value both averages the update (steadier than one pair's noise) and uses the card.
    batch_size: PositiveInt = 1
    # Decay the rate to zero over the run (cosine). Warm-starting a CONVERGED model is exactly the case a
    # schedule is for: a flat rate keeps taking full-size steps away from an optimum the weights already sit
    # in, while an annealed one explores early and settles.
    cosine_lr: bool = False


class ContrastiveSite(StrEnum):
    """WHERE the contrastive term acts — the measured failure was placement, so this is the knob that carries it.

    On the shared FEATURES (what ran, and what degraded the model: proxy 0.8414 -> 0.7292, node recall
    0.966 -> 0.867 while the term itself converged) the objective reshapes the very representation the detect
    head reads. Through a PROJECTION it shapes an embedding of them instead, and DETACHED_PROJECTION is the
    strict form of that — the contrastive gradient stops at the head and reaches no backbone weight at all.
    """

    FEATURES = "features"
    PROJECTION = "projection"
    DETACHED_PROJECTION = "detached_projection"

    @property
    def projects(self) -> bool:
        """Whether the term reads a head of its own rather than the shared features directly."""
        return self is not ContrastiveSite.FEATURES

    @property
    def stop_gradient(self) -> bool:
        """Whether the trunk is detached below that head — the contrastive term shaping ONLY the embedding."""
        return self is ContrastiveSite.DETACHED_PROJECTION


class LossCfg(BaseModel):
    """The objective: how the three terms are weighted, and which voxels the detection term is allowed to see."""

    model_config = _VALIDATED

    neg_weight: float = Field(1e-2, gt=0)  # background-voxel down-weight in the detection BCE (published)
    # Weight on the (two-frame) detection term relative to the edge term. 1.0, not the published 1e1: the
    # detection term is load-bearing here (a proxy A/B measured 0.1 WORSE), so the terms are carried even.
    det_weight: float = Field(1.0, ge=0)
    # Weight on the InfoNCE term over the NODE FEATURES — the only part of the objective that asks the
    # backbone for discrimination rather than invariance. 0.0 by default: the term is still computed and
    # logged as a diagnostic, but contributes exactly nothing, so an unasked run is the run of yesterday.
    contrastive_weight: float = Field(0.0, ge=0)
    # WHERE that one weight acts. `FEATURES` is what ran and what the measurement condemns; it stays the
    # default so a run asking for nothing is still yesterday's run, and the term is off at weight 0 anyway.
    contrastive_site: ContrastiveSite = ContrastiveSite.FEATURES
    temperature: float = Field(0.07, gt=0)  # the InfoNCE softmax temperature over candidate targets
    # Weight on the hard-negative RANKING term (`HardNegativeMargin`) — each annotated source's true logit held
    # above its nearest wrong targets'. 0.0 by default, so the term is computed and logged as a diagnostic (its
    # magnitude is what a run matches the weight to) while an unasked run is byte-for-byte the run of yesterday.
    # The zni probe's ABSOLUTE form of this (push the decoy's probability to zero, UNet frozen) was refuted by
    # flattening; the margin form is invariant to depressing a whole row, so it can only be satisfied by re-ranking.
    hard_negative_weight: float = Field(0.0, ge=0)
    # How many nearest wrong targets each annotated source contributes — the zni probe's mined count, kept so the
    # two tools mine the same set and only the OBJECTIVE over it differs.
    hard_negatives: PositiveInt = 4
    # Sigmoid response above which an UNANNOTATED voxel stops being supervised as background (see
    # `BalancedBCE`). Our labels cover ~1-2% of a frame's cells, so the zero target calls thousands of real
    # cells background — and warm-starting a saturated detector aims exactly those gradients at its correct
    # detections (measured: node recall 0.966 -> 0.936, ratio -0.15, within half an epoch). None keeps the
    # objective of yesterday; `main` derives the value it passes from `TrackerConfig.threshold`.
    ignore_ambiguous_above: float | None = Field(None, gt=0, lt=1)
    # Whether the link loss ALSO normalises across targets ("one child per source") rather than across sources
    # alone. False keeps the frontier's form and yesterday's numbers exactly. True fixes two measured defects:
    # a single-source pair is degenerate under the source axis (identically-1.0 softmax, zero gradient, a
    # constant 66.67 loss — 28.8% of our real pairs), and a rival TARGET can otherwise only be pushed down when
    # another supervised source claims its column, which no unannotated mislink partner ever has.
    symmetric_links: bool = False
    # Whether the link loss counts one DECISION at a time and balances the true candidate against its rivals
    # inside it, instead of averaging over cells. The cell mean leaves the association objective unbalanced in
    # a way the DETECTION objective beside it already refuses: `BalancedBCE` weighs positives 1/n_pos and
    # negatives w/n_neg per frame, while the link loss lets the single true candidate be outnumbered by its
    # rivals — up to 38 to 1 on the detected corpus (targets/gap mean 8.3, max 39), with only the focal power
    # pushing back and by an amount nobody chose. False keeps the cell mean and every number measured under it.
    balanced_links: bool = False

    @property
    def link_axes(self) -> tuple[int, ...]:
        """The axes the link loss normalises over — the source axis always, the target axis when asked."""
        return (SOURCE_AXIS, TARGET_AXIS) if self.symmetric_links else (SOURCE_AXIS,)


class EvalCfg(BaseModel):
    """The in-loop scorer: the operating point the selector's tracker runs at, and what it is allowed to cost."""

    model_config = _VALIDATED

    # The SELECTOR runs the WHOLE shipped operating point, not a threshold grafted onto defaults.
    # This field used to be `threshold: float`, derived from `TrackerConfig().threshold` — the right instinct
    # applied to ONE field out of fourteen, while two of the other thirteen (`linker.name`,
    # `bidirectional_edges`) are exactly what separates the 0.892 tier from the 0.895 one we submit. Selection
    # exists to rank candidates the way deployment will, and a partially-specified operating point silently
    # ranked them under a linker we never ship. Carrying the object makes that state INEXPRESSIBLE rather than
    # merely discouraged.
    tracker: TrackerConfig = Field(default_factory=TrackerConfig.shipped)
    # The selector eval skips flip-TTA: it buys a faithful score the checkpoint choice doesn't need, at 4x cost.
    tta: bool = False


class ScheduleCfg(BaseModel):
    """The budget and the stopping policy, in BOTH scales — raw steps, and passes over the corpus.

    The loop consumes one GT pair per step, so a pass is `len(train_targets)` steps: `steps_per_epoch` is
    DERIVED from the split at `resolved` time, never a field, since a hand-set value rots the moment the
    split moves. Unset (None) epoch fields leave the step-scale values exactly as given.
    """

    model_config = _VALIDATED

    steps: PositiveInt = 1500
    # The eval is a full `CellTracker` pass over the four proxy movies, serial with training — keep it coarse.
    eval_every: PositiveInt = 500
    # Stop after this many consecutive non-improving evals; save-best means stopping early never costs
    # quality, only compute. <1 disables (runs the full `steps` budget).
    patience: int = Field(5, ge=0)
    es_min_delta: float = Field(0.0, ge=0)
    epochs: float | None = Field(None, gt=0)
    evals_per_epoch: PositiveInt | None = None
    patience_epochs: float | None = Field(None, gt=0)

    def resolved(self, steps_per_epoch: int) -> ScheduleCfg:
        """The same schedule expressed in steps — what the loop, the scheduler and early-stop all read."""
        eval_every = self._eval_every(steps_per_epoch)
        windows_per_epoch = self.evals_per_epoch if self.evals_per_epoch is not None else steps_per_epoch / eval_every
        steps = self.steps if self.epochs is None else round(self.epochs * steps_per_epoch)
        patience = self.patience
        if self.patience_epochs is not None:
            patience = max(1, round(self.patience_epochs * windows_per_epoch))
        self._log_schedule(steps_per_epoch, steps, eval_every, patience)
        return self.model_copy(update={"steps": steps, "eval_every": eval_every, "patience": patience})

    def _eval_every(self, steps_per_epoch: int) -> int:
        """Window length in steps; a corpus smaller than the requested evals still yields a one-step window."""
        if self.evals_per_epoch is None:
            return self.eval_every
        return max(1, steps_per_epoch // self.evals_per_epoch)

    def _overrides(self) -> list[str]:
        """Which raw step-scale values the epoch form replaced — reported so neither form is dropped in silence."""
        return [
            f"{epoch}={getattr(self, epoch)} overrides {raw}={getattr(self, raw)}"
            for epoch, raw in _EPOCH_OVERRIDES
            if getattr(self, epoch) is not None
        ]

    def _log_schedule(self, steps_per_epoch: int, steps: int, eval_every: int, patience: int) -> None:
        """One startup line carrying every derived value, so a run's schedule is self-documenting."""
        overrides = self._overrides()
        logger.info(
            "schedule: %d steps/epoch | %d steps (%.2f epochs) | eval every %d steps | patience %d windows%s",
            steps_per_epoch,
            steps,
            steps / steps_per_epoch,
            eval_every,
            patience,
            " | " + "; ".join(overrides) if overrides else "",
        )


class RuntimeCfg(BaseModel):
    """Where the run executes and how it is made repeatable — nothing here changes the objective."""

    model_config = _VALIDATED

    device: str = "cuda"
    seed: int = Field(0, ge=0)
    # Frame-decompress threads and pairs kept in flight, so the GPU step overlaps the reads. The joint loop read
    # its two frames INLINE — a measured ~16 ms each against a 160-440 ms step, so a tenth to a fifth of every
    # step was the GPU waiting on zstd.
    #
    # WHY NOT AS MANY AS POSSIBLE: because the loop cannot consume them. ONE reader delivers 34.9 pairs/s
    # (28.7 ms/pair measured on real data) while the loop consumes 2.3-6.2 pairs/s, so a single background
    # thread already has 6-15x more throughput than the step can take, and the pooled store roughly halves the
    # read again. Threads beyond that hide nothing and compete for cores with whatever else the box is running
    # — which during these very measurements was two other training jobs. 4 is MARGIN for a contended box and a
    # dense video, not a throughput requirement; 8 measures faster on the read alone (10.2 ms/pair) and is
    # indistinguishable end-to-end. 1 is the serial path, byte-identical, and is what the curriculum path uses
    # whatever this says.
    loader_threads: PositiveInt = 4
    loader_prefetch: PositiveInt = 8
    compile_backbone: bool = False  # torch.compile the U-Net (static shape); one-time warmup, then fused kernels


class JointTrainConfig(BaseModel):
    """A whole joint fine-tune as one validated object — six concerns, each its own group."""

    model_config = _VALIDATED

    model: ModelCfg = Field(default_factory=ModelCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    loss: LossCfg = Field(default_factory=LossCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    schedule: ScheduleCfg = Field(default_factory=ScheduleCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)

    def resolved(self, steps_per_epoch: int) -> JointTrainConfig:
        """The same run with its schedule in steps — resolved once, the moment the pair count is known."""
        return self.model_copy(update={"schedule": self.schedule.resolved(steps_per_epoch)})
