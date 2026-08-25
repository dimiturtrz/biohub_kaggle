"""Assemble the trainable joint model and its optimiser from one configuration — the run's build phase.

Everything here happens ONCE, before the training loop: pick the model (chained from a prior stage, warm-started
from a pilkwang pack, or fresh), install the optional features (temporal position, prior-velocity widening,
relative-position bias, LoRA adapters), compile the backbone, and hand the right parameter set to the optimiser.
The order is load-bearing (see `build`), which is exactly why it is one object: a single place that knows the one
sequence keeping BOTH a warm start and a widened head correct.

The link objective and the contrastive head are OWNED by the trainer (the loss reads them every step) and passed
in here, because the same instances must ride the optimiser and move to the device — a second copy would train a
different head than the loss scores. The prior-velocity source, by contrast, is config-derived state produced
here (`velocity`) and read back by the trainer per step: the feature that widens the head at build time is the
same feature that feeds the history forward, so it has one home beside the widening.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from celltrack.losses.association_objective import AssociationObjective
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.models.lora import LoRA
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from celltrack.operating_point import TrackerConfig
from celltrack.training.contrastive_term import ContrastiveTerm
from celltrack.training.joint_config import JointTrainConfig, LoraCfg
from celltrack.training.prior_velocity_source import PriorVelocitySource
from celltrack.training.tunet_detector import _Optimization
from core.geometry import Spacing
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
# The fresh transformer's shape, matching the pilkwang head the warm-start path loads.
_HIDDEN_DIM, _N_HEADS, _N_BLOCKS = 128, 4, 4
_POS_FEATURE_DIM = 4 * _POS_EMBED_DIM  # sinusoidal embed of (t, z, y, x)


class JointAssembly:
    """Builds the run's trained model + optimiser from its configuration; owns the model-side feature installs.

    Holds the config plus the two trainer-owned auxiliary modules the optimiser must also own (the contrastive
    head and the link objective). Construct fresh whenever the config is re-resolved — it caches nothing but its
    inputs, so a rebuild is cheap and keeps the schedule horizon in step with the resolved budget.
    """

    def __init__(self, config: JointTrainConfig, contrastive: ContrastiveTerm, objective: AssociationObjective) -> None:
        self.config = config
        self.contrastive = contrastive
        self._objective = objective

    def build(
        self,
        *,
        warm_start: bool,
        init_weights: Path | None = None,
        detector_from: Path | None = None,
        spacing: Spacing | None = None,
    ) -> tuple[JointModel, _Optimization]:
        """The run's trained objects, built in the one order that keeps BOTH a warm start and a widened head.

        The order is load-bearing. `widen_linear_inputs` REPLACES the head's input projection with a new
        parameter object, so the sequence is: load the pretrained weights → widen → wrap for relative position →
        and only then hand `model.parameters()` to the optimiser — built any earlier it holds the discarded proj.
        """
        model = self._model(warm_start=warm_start, init_weights=init_weights, detector_from=detector_from).to(
            self.config.runtime.device
        )
        if detector_from is not None:  # head-only validation: the warm detector is frozen, only the head trains
            model.detector.requires_grad_(requires_grad=False)
            self.freeze_backbone_norm(model)
        pretrained = {id(parameter) for parameter in model.parameters()}  # the base, BEFORE the feature installs
        if self.config.model.temporal_position:  # motion sight; the position table lands on the model's device
            model.detector.install_temporal_position()
        self.velocity.widen(model)
        if self.config.model.relative_position:  # geometry sight; a zero-init per-head distance bias in the head
            if spacing is None:
                raise ValueError("--relative-position needs the corpus spacing; PairSplit.spacing is unset")
            model.install_relative_position(
                spacing,
                TrackerConfig.shipped().linker.gate_um,
                self.config.runtime.device,
                directional=self.config.model.relative_position_directional,
            )
        if self._lora.enabled:  # freeze the base, adapt the association locus; detection is anchored
            installed = [parameter for parameter in model.parameters() if id(parameter) not in pretrained]
            adapted = LoRA.inject(model, self._lora.to_config(), keep=installed)
            if adapted == 0:
                raise ValueError(f"LoRA targets {self._lora.targets} matched no layers")
            logger.info("LoRA: %d layers adapted (rank %d), %d params kept", adapted, self._lora.rank, len(installed))
            model = model.to(self.config.runtime.device)  # the new adapter params default to CPU — move them
        if self.config.runtime.device == "cuda":
            # channels_last_3d is a conv layout — scoped to `detector` (conv backbone) NOT the whole model: HOCT's
            # rank-3/4 head tables raise "required rank 5 tensor". Stub omits the memory_format overload; valid.
            model.detector = model.detector.to(memory_format=torch.channels_last_3d)  # type: ignore[no-matching-overload]
        # Compile the backbone ALWAYS (static 64^3 shape) — not an option: a crash is a bug to fix at its root,
        # not a reason to run eager. Default mode (fusion), not reduce-overhead: CUDA-graphing the backbone ALONE
        # leaves the step-wide launch gaps (heads/loss/optimiser stay eager) untouched, so it did not lift the
        # oscillating util and only added graph-capture warmup — the launch-latency needs the WHOLE step static,
        # which the ragged per-pair loss is not. The temporal-attention SDPA is forced to MATH at forward time
        # (see `_step`) — its efficient backward has a broken compiled meta-kernel.
        model.detector.unet = torch.compile(model.detector.unet, dynamic=False)  # type: ignore[bad-assignment]
        # The edge head stays EAGER: it is a tiny attention over the padded nodes, already collapsed to ONE
        # batched call, so compiling it fused little — and the SAME module runs ragged per-pair in the selector
        # eval (whole videos, every gap a different node count), where a static compile thrashes the recompile cap.
        if self.config.data.gpu_scene_fraction:  # fuse the uniform-N generated-scene edge head (measured 4x)
            model.compile_scene_head(self.config.scene_config().n_cells)
        self.contrastive.to(self.config.runtime.device)
        self._objective.to(self.config.runtime.device)
        optimizer = torch.optim.AdamW(self.trained(model), lr=self.config.optim.lr)
        return model, _Optimization(optimizer, self._schedule(optimizer))

    def trained(self, model: JointModel) -> list[nn.Parameter]:
        """Every parameter the optimiser owns and the clip covers — both heads, plus the contrastive head.

        At the `FEATURES` site the contrastive term holds none, so the list IS `model.parameters()` in the same
        order (an unasked run is byte-identical). Under LoRA the base is frozen, so only the adapter deltas — all
        that moves — are handed over.
        """
        if self._lora.enabled:
            return [*LoRA.adapter_parameters(model), *self.contrastive.parameters(), *self._objective.parameters()]
        trained = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return [*trained, *self.contrastive.parameters(), *self._objective.parameters()]

    @staticmethod
    def freeze_backbone_norm(model: JointModel) -> None:
        """Put every BatchNorm in the detector into eval mode so it normalises by the warm pack's stored running
        stats, not the current step's noisy batch stats. The affine weight/bias keep `requires_grad` and still
        train; only the running-stat update and the batch-stat normalisation are suppressed."""
        for module in model.detector.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):  # noqa: SLF001 — no public base spans BN1d/2d/3d+SyncBN
                module.eval()

    @property
    def velocity(self) -> PriorVelocitySource:
        """The run's history feed, reading the flag live — the config may be re-resolved before the model exists."""
        return PriorVelocitySource(
            enabled=self.config.data.prior_velocity, gt_warmup_steps=self.config.data.velocity_gt_warmup_steps
        )

    def _schedule(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler | None:
        """A cosine decay to zero over the whole run, or none for a flat rate."""
        if not self.config.optim.cosine_lr:
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.schedule.steps)

    @property
    def _lora(self) -> LoraCfg:
        """The run's adapter config, read live — keeps callers off a four-deep reach through `config.model.lora`."""
        return self.config.model.lora

    def _model(
        self, *, warm_start: bool, init_weights: Path | None = None, detector_from: Path | None = None
    ) -> JointModel:
        """A joint model to train — chained from a prior stage's checkpoint, warm-started from the pack, or fresh.

        `init_weights` continues a checkpoint THIS trainer wrote (the staged curriculum): `from_checkpoint`
        rebuilds both heads at the width the file was saved at — a compile prefix reloads transparently — so
        the stage inherits the last stage's weights and nothing else. CAVEAT: chaining a `--prior-velocity`
        checkpoint is broken — `build` widens the head a SECOND time over the already-widened proj (bd
        biohub_kaggle: double-widen). The non-velocity curriculum path is unaffected.

        `detector_from` warms ONLY the detector from a checkpoint and pairs it with a FRESH configured head —
        the one init path that crosses head classes (a pack-detector ckpt under a `--head hoct` run), so a head
        mechanism is read on identical detector features without re-paying the detector pretrain.
        """
        architecture, downsample = self.config.model, self.config.data.downsample
        if init_weights is not None:
            if warm_start:
                raise ValueError("--init-weights and --warm-start are mutually exclusive: one init, not two")
            logger.info("chained from prior-stage checkpoint at %s", init_weights)
            return JointModel.from_checkpoint(init_weights, self.config.runtime.device)
        if detector_from is not None:
            if warm_start or init_weights is not None:
                raise ValueError("--detector-from is its own init: not combinable with --warm-start / --init-weights")
            source = JointModel.from_checkpoint(detector_from, self.config.runtime.device)
            if source.downsample != downsample:
                raise ValueError(f"detector-from grid {source.downsample} != run grid {downsample}")
            head = EdgeTransformerScorer._head_cls(architecture.head)(  # noqa: SLF001 — the edge-head class
                feat_dim=architecture.out_channels + _POS_FEATURE_DIM,
                hidden_dim=_HIDDEN_DIM,
                n_heads=_N_HEADS,
                n_blocks=_N_BLOCKS,
            )
            logger.info("warm detector from %s, fresh %s head (detector frozen)", detector_from, architecture.head)
            return JointModel(source.detector, head, downsample)
        if warm_start:
            if architecture.norm != "batch" or architecture.head != "pack":
                raise ValueError("warm-start needs the pilkwang BN pack: --norm group / --head hoct are from-scratch")
            pack = DataRoot.from_config(_CONFIG).processed(_DATASET) / architecture.warm_pack
            scorer = EdgeTransformerScorer.from_pack(pack, self.config.runtime.device)
            logger.info("warm-started from pilkwang pack at %s", pack)
            return JointModel(scorer.detector, scorer.transformer, downsample)
        detector = TemporalUNetDetector(architecture.out_channels, architecture.layers, norm=architecture.norm)
        transformer = EdgeTransformerScorer._head_cls(architecture.head)(  # noqa: SLF001 — the edge-head class
            feat_dim=architecture.out_channels + _POS_FEATURE_DIM,
            hidden_dim=_HIDDEN_DIM,
            n_heads=_N_HEADS,
            n_blocks=_N_BLOCKS,
        )
        return JointModel(detector, transformer, downsample)
