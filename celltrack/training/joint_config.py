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

import argparse
import logging
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from celltrack.data.augmentation import DEFAULT_AUGMENTATION, Augmentation
from celltrack.data.synthetic_scene import FAITHFUL_APPEARANCE, SceneConfig
from celltrack.losses.softmax_focal_bce import SOURCE_AXIS, TARGET_AXIS
from celltrack.losses.tracking_loss import TrackingLossConfig
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
    # BatchNorm3d ("batch", the published backbone byte-for-byte) or GroupNorm ("group"). BN stores the train
    # movies' running intensity statistics and applies them unchanged to val frames at eval, leaking the source
    # domain across the 44b6/6bba microscope gap; GroupNorm recomputes per sample and stores no source statistic,
    # so it is domain-robust. A from-scratch-only swap — a warm-started BN pack has no home for its running
    # buffers in a GroupNorm backbone, so `norm="group"` with `--warm-start` is rejected at build.
    norm: Literal["batch", "group"] = "batch"
    # Hold the warm pack's BatchNorm running stats FROZEN through a warm finetune. `model.train()` puts BN in
    # train mode, where it normalises by each step's small-batch statistics instead of the published running
    # mean/var the pretrained weights expect — so a warm detector's train-mode forward diverges from its eval
    # forward (measured: det loss spikes to ~9.5 vs the warm ~2-3), the shared backbone drifts fighting that
    # noise, and the association the edge head reads off it degrades (over-links, proxy collapses). Eval-mode BN
    # uses the stored stats in BOTH phases; the affine weight/bias still train. Warm-start only (GroupNorm has
    # no running stats to freeze; a from-scratch BN has none worth keeping).
    freeze_backbone_norm: bool = False
    # Add a learned per-head distance bias to the edge transformer's CROSS-ATTENTION — a zero-initialised
    # `DistanceAttentionBias` over the pair's separation in micrometres (see relative_position_transformer). Off =
    # the pack head byte for byte; on, geometry enters every attention layer the pack otherwise decides on
    # appearance alone (the dense-confusor axis). Zero-init keeps a warm start identical at step 0; the spacing is
    # threaded from the corpus and the reach is the linker's own gate, so no new distance scale is introduced.
    relative_position: bool = False
    # WHICH edge-head architecture to build from scratch. "pack" is the pilkwang SimpleNodeTransformer (feature
    # cross-attention, position-blind, first-order pair scoring) byte for byte. "hoct" is the HOCT two-stage head:
    # 3D-RoPE node self-attention then edge-to-edge self-attention with a line-to-line geometry bias — the
    # relative-position sight and higher-order edge comparison the pack head structurally cannot represent (the
    # dense-confusor discrimination axis). From-scratch only: a pack warm start carries no HOCT weights to continue.
    head: Literal["pack", "hoct"] = "pack"


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
    # Steps to bootstrap the velocity off GROUND-TRUTH `t-1 -> t` links before the model's own affinity takes
    # over (linear anneal). Breaks the from-scratch chicken-egg: the model's early prev-gap affinity is
    # confident-wrong, so a self-derived velocity points at the wrong cell and cannot fix the confusor it exists
    # for. Only meaningful with `prior_velocity`; 0 = off and byte-identical off (pure model-derived from step 0).
    # The CLI collapses the two into one decision: `--velocity-gt-warmup-steps N` (N > 0) turns `prior_velocity`
    # ON, so the from-scratch case is one flag and the warmup-0 dead-end cannot be reached by forgetting a flag.
    velocity_gt_warmup_steps: int = Field(0, ge=0)
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
    # Render generated scenes with the appearance-faithful preset (FAITHFUL_APPEARANCE) instead of the identical-
    # blob default. One decision, not a knob sweep: the eqdx KS audit found the default blobs appearance-degenerate
    # (why prior synth-training refutations were shallow — undertrained on a distribution the detector could shortcut).
    # The preset adds size SPREAD + faithful z-shape + brightness/noise so both heads must learn the real signal.
    # Off by default and byte-identical off. Reaches the numpy `synthetic_scenes` path fully; the GPU separable path
    # honours every field except per-cell size (one shared kernel for conv3d speed). See synthetic_scene.py.
    faithful_scenes: bool = False
    # Fraction of generated cells turned into CONSTRUCTED confusors (a fast source + a slow near-source distractor,
    # a labelled hard negative — see synthetic_scene.SceneConfig.confusor_rate). Uniform placement poses the
    # confusor only ~4%; joint_confusor_synth_v1 trained on trivially-easy edges and P_true never moved. >0 raises
    # the measured rate ~linearly (0.30 -> ~32%, confusor_audit). 0 (default) is the plain uniform generator.
    confusor_rate: float = Field(0.0, ge=0.0, le=0.5)
    # Replace the FIXED share above with the feedback controller (`PacedMixture`): the two populations
    # alternate structurally and an EMA'd difficulty moves a per-sample LOSS WEIGHT instead of a draw
    # probability, with save-best corrected for the difficulty the window trained at. Off by default.
    paced_curriculum: bool = False
    # Fold the four TEST movies into the TRAIN corpus (the stage-3 fine-tune). Selection is unaffected — the edge
    # AUC and save-best still read the validation four — so the leaderboard estimate stays a real held-out number;
    # only the pairs the loop optimises on grow. Legitimate because those movies ARE given and the hidden eval
    # overlaps them; the trap this avoids is SELECTING on them (proxy.py's documented anti-transfer), not training.
    include_test_in_train: bool = False
    # Label-preserving detection augmentation the pilkwang recipe has and ours lacked — the from-scratch recall
    # gap is warm weights recovering cells our own miss from IDENTICAL pixels (g8j8), i.e. an intensity/orientation
    # dependence a from-scratch run never had the variety to break. These CONFIG fields default to the identity, so
    # a programmatically-built config is byte-for-byte the un-augmented baseline; the CLI opts in for the recipe
    # (`--augment`, DEFAULT_AUGMENTATION) and `--no-augment` restores this identity. A pair shares ONE draw across
    # `t-1`/`t`/`t+1` (joint_dataset).
    aug_brightness: float = Field(0.0, ge=0.0)  # multiplicative jitter half-range
    aug_offset: float = Field(0.0, ge=0.0)  # additive jitter half-range
    aug_flip_axes: tuple[int, ...] = ()  # frame axes (0=z, 1=y, 2=x) each flipped with p=0.5

    def augmentation(self) -> Augmentation:
        """The augmentation policy these fields describe — identity by default, so an unasked run is unchanged."""
        return Augmentation(brightness=self.aug_brightness, offset=self.aug_offset, flip_axes=self.aug_flip_axes)

    def scene_config(self) -> SceneConfig:
        """The SceneConfig every generated-scene path renders from — one home for the faithful/default choice."""
        base = FAITHFUL_APPEARANCE if self.faithful_scenes else SceneConfig()
        return replace(base, confusor_rate=self.confusor_rate)


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
    # stacking B pairs' windows fills the device where a per-pair forward left it ~40% idle (GPU ~4% util,
    # Python-latency-bound) for the whole campaign. The gradient is the MEAN over the batch, so `lr` keeps its
    # meaning across sizes — no rescale when this moves. 16 is the DEFAULT because the batched backbone is
    # compute-bound by ~16 (throughput flat to ~48, a VRAM cliff past it): the smallest batch that saturates the
    # card, so it takes the full throughput at the least activation memory and the least departure from the
    # per-pair reference's gradient. 1 restores the exact per-pair update (mean of one, step on every pair) for a
    # run that must stay bitwise-comparable to the old single-pair arms; a batch-eligible config auto-falls back
    # to that same per-pair loop whenever a pair overflows the pad (`JointModel.fits_batched`), so 16 is safe.
    batch_size: PositiveInt = 16
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
    # Weight each pair's loss by SNR/(1+crowd) (`SampleWeight`), batch-normalised to mean 1: trust bright,
    # isolated supervision, ease off dim crowded cells the labels cannot adjudicate. A soft LOSS (it scales the
    # gradient), NOT a soft logit (see the calibration term). False = every pair weighs equally, yesterday's run.
    reliability_weighting: bool = False
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
    # Whether the link loss appends a learned "no source parents this target" row before the source-axis
    # softmax (Trackastra's `1 + Σexp`). It gives an unannotated rival's probability mass somewhere to go other
    # than a real source it does not belong to — the dense confusor — and un-degenerates the single-source pair.
    # Overrides `symmetric_links` (the slack row is a statement about parents, source-axis only). The slack
    # logit is one LEARNED scalar calibrated to the mean real logit, not the frontier's constant 0 (our logits
    # sit near -11). False keeps the plain focal BCE and every number measured under it.
    slack_links: bool = False

    @property
    def link_axes(self) -> tuple[int, ...]:
        """The axes the link loss normalises over — the source axis always, the target axis when asked."""
        return (SOURCE_AXIS, TARGET_AXIS) if self.symmetric_links else (SOURCE_AXIS,)

    def to_tracking_loss_config(self, downsample: tuple[int, int, int]) -> TrackingLossConfig:
        """The loss's own config this pydantic view fills — the trainer builds `TrackingLoss` from it.

        The reliability weight's crowd gate is the linker's admission radius; it lives in the shipped operating
        point, not the loss knobs, so it is resolved here rather than being a field the CLI could contradict.
        """
        return TrackingLossConfig(
            neg_weight=self.neg_weight,
            ignore_ambiguous_above=self.ignore_ambiguous_above,
            det_weight=self.det_weight,
            contrastive_weight=self.contrastive_weight,
            hard_negative_weight=self.hard_negative_weight,
            hard_negatives=self.hard_negatives,
            reliability_weighting=self.reliability_weighting,
            downsample=downsample,
            gate_um=TrackerConfig.shipped().linker.gate_um,
        )


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

    def scene_config(self) -> SceneConfig:
        """The generated-scene appearance, so a caller holding the whole config need not reach through `data`."""
        return self.data.scene_config()

    @staticmethod
    def _augmentation_from_args(args: argparse.Namespace) -> Augmentation:
        """Resolve the CLI's aug decision: the preset when on, the identity on `--no-augment`, with per-field overrides.

        The CLI turns augmentation on by default (the recipe wants it); an `--aug-*` flag left unset takes the
        preset's value, an explicit one overrides only that field. `--no-augment` is the byte-identical baseline.
        """
        if not args.augment:
            return Augmentation()
        preset = DEFAULT_AUGMENTATION
        return Augmentation(
            brightness=preset.brightness if args.aug_brightness is None else args.aug_brightness,
            offset=preset.offset if args.aug_offset is None else args.aug_offset,
            flip_axes=preset.flip_axes if args.aug_flip_axes is None else tuple(args.aug_flip_axes),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> JointTrainConfig:
        """Fan the CLI's flat flags out into the six concern groups — the parser is the interface, this the model."""
        augmentation = cls._augmentation_from_args(args)
        return cls(
            model=ModelCfg(
                warm_pack=WARM_PACKS[args.warm_pack],
                temporal_position=args.temporal_position,
                relative_position=args.relative_position,
                norm=args.norm,
                head=args.head,
                freeze_backbone_norm=args.freeze_backbone_norm,
                lora=LoraCfg(
                    enabled=args.lora, rank=args.lora_rank, alpha=args.lora_alpha, targets=tuple(args.lora_targets)
                ),
            ),
            data=DataCfg(
                downsample=tuple(args.downsample),
                detected_videos=args.detected_videos,
                difficulty_sampling=args.difficulty_sampling,
                prior_velocity=args.prior_velocity or args.velocity_gt_warmup_steps > 0,
                velocity_gt_warmup_steps=args.velocity_gt_warmup_steps,
                synthetic_fraction=args.synthetic_fraction,
                synthetic_sequences=args.synthetic_sequences,
                synthetic_scenes=args.synthetic_scenes,
                gpu_scene_fraction=args.gpu_scene_fraction,
                gpu_scene_detection=args.gpu_scene_detection,
                faithful_scenes=args.faithful_scenes,
                confusor_rate=args.confusor_rate,
                paced_curriculum=args.paced_curriculum,
                include_test_in_train=args.include_test_in_train,
                aug_brightness=augmentation.brightness,
                aug_offset=augmentation.offset,
                aug_flip_axes=augmentation.flip_axes,
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
                slack_links=args.slack_links,
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
                loader_threads=args.loader_threads,
                loader_prefetch=args.loader_prefetch,
            ),
        )

    def resolved(self, steps_per_epoch: int) -> JointTrainConfig:
        """The same run with its schedule in steps — resolved once, the moment the pair count is known."""
        return self.model_copy(update={"schedule": self.schedule.resolved(steps_per_epoch)})

    def augmentation(self) -> Augmentation:
        """The label-preserving augmentation policy — the trainer asks the run, not its data group."""
        return self.data.augmentation()
