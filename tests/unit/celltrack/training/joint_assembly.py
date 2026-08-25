"""Unit tests for the build phase — the model + optimiser assembly this repo owns, on CPU with no data.

`JointAssembly` is the pre-loop half of the joint trainer: it picks the model (fresh / warm / chained), installs
the optional features in the one load-bearing order, and hands the right parameter set to the optimiser. These
tests exercise that order and the parameter bookkeeping directly on the unit — the loop itself is covered beside
the trainer in `joint_detector.py`.
"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from celltrack.losses.association_objective import AssociationObjective
from celltrack.models.edge_transformer import _POS_EMBED_DIM
from celltrack.models.joint_model import JointModel
from celltrack.models.lora import LoRA
from celltrack.models.prior_velocity import PriorVelocity
from celltrack.training.contrastive_term import ContrastiveSite, ContrastiveTerm, ContrastiveTermConfig
from celltrack.training.joint_assembly import JointAssembly
from celltrack.training.joint_config import (
    DataCfg,
    JointTrainConfig,
    LoraCfg,
    LossCfg,
    ModelCfg,
    OptimCfg,
    RuntimeCfg,
    ScheduleCfg,
)

_CORPUS = 18024  # the real GT-pair count: one pair per step, so this is one epoch


def _assembly(config: JointTrainConfig) -> JointAssembly:
    """A `JointAssembly` wired exactly as the trainer builds it — the SHARED contrastive head and link objective.

    The two auxiliary modules must be the same instances the loss scores, so the trainer constructs them once and
    hands them to the assembly; a unit test arranges the same way, since the optimiser-ownership assertions here
    are about precisely those shared parameters.
    """
    contrastive = ContrastiveTerm(ContrastiveTermConfig.from_loss(config.loss, config.model.out_channels))
    objective = AssociationObjective.from_config(
        link_axes=config.loss.link_axes,
        balanced_links=config.loss.balanced_links,
        slack_links=config.loss.slack_links,
    )
    return JointAssembly(config, contrastive, objective)


def _cpu_config() -> JointTrainConfig:
    """A tiny CPU config: a two-level backbone, no downsample, one eval-sized window of two steps."""
    return JointTrainConfig(
        model=ModelCfg(out_channels=2, layers=(2, 4)),
        data=DataCfg(downsample=(1, 1, 1)),
        runtime=RuntimeCfg(device="cpu"),
        schedule=ScheduleCfg(steps=2, eval_every=2, patience=2),
    )


def _velocity_config() -> JointTrainConfig:
    """The tiny CPU config with the prior-velocity feature on — the only difference from the default run."""
    return _cpu_config().model_copy(update={"data": DataCfg(downsample=(1, 1, 1), prior_velocity=True)})


def _lora_velocity_config() -> JointTrainConfig:
    """Velocity ON under a frozen-base LoRA — the combo where the widened columns must survive the base freeze."""
    return _cpu_config().model_copy(
        update={
            "model": ModelCfg(out_channels=2, layers=(2, 4), lora=LoraCfg(enabled=True)),
            "data": DataCfg(downsample=(1, 1, 1), prior_velocity=True),
        }
    )


def _contrastive_config(site: ContrastiveSite) -> JointTrainConfig:
    """The tiny CPU config with the contrastive term ON at one placement — the only difference from the default."""
    return _cpu_config().model_copy(update={"loss": LossCfg(contrastive_weight=0.1, contrastive_site=site)})


def test_group_norm_rejects_a_warm_start():
    """`--norm group` with `--warm-start` fails fast: a pilkwang BN pack cannot load a bufferless GroupNorm.

    The guard fires before any pack is touched, so it needs no data root — a from-scratch-only architecture
    swap and a warm init are contradictory, and the run says so at build rather than crashing on a key mismatch.
    """
    assembly = _assembly(JointTrainConfig(model=ModelCfg(norm="group")))
    with pytest.raises(ValueError, match="from-scratch"):
        assembly._model(warm_start=True)


def test_fresh_build_threads_the_group_norm():
    """A from-scratch build carries the configured norm into the detector it constructs — the GPU arm's backbone.

    The warm path is barred (previous test); this is the path the from-scratch run actually takes, so the
    `--norm group` flag must reach the built detector rather than defaulting the published BN backbone back in.
    """
    assembly = _assembly(JointTrainConfig(model=ModelCfg(out_channels=2, layers=(2, 4), norm="group")))
    assert assembly._model(warm_start=False).detector.norm == "group"


def test_schedule_spans_the_resolved_step_count():
    """The cosine horizon is built from the RESOLVED budget — resolving epochs late must not leave it stale."""
    config = JointTrainConfig(schedule=ScheduleCfg(steps=10, epochs=2.0), optim=OptimCfg(cosine_lr=True)).resolved(
        _CORPUS
    )
    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-4)

    schedule = _assembly(config)._schedule(optimizer)

    assert schedule is not None
    assert schedule.T_max == 36048  # type: ignore[missing-attribute]


def test_build_keeps_the_velocity_columns_trainable_under_lora():
    """LoRA freezes the base, but the velocity's widened projection is a zero-init feature INSTALL, not base.

    `LoRA.inject`'s blanket freeze catches the installed columns, and `adapter_parameters` (requires_grad only)
    then drops them from the optimiser: left so, they sit at their zero init forever and the feature is a silent
    no-op that still reads ON in the log. `build` thaws them, so the widened projection both trains and rides
    the optimiser — while its new columns still start at exactly zero, so the warm start is intact at step 0.
    """
    assembly = _assembly(_lora_velocity_config())
    model, optimization = assembly.build(warm_start=False)
    proj = model.transformer.proj
    base = cast(nn.Linear, proj.base if isinstance(proj, LoRA.Linear) else proj)  # LoRA wraps the widened proj

    optimised = {id(parameter) for group in optimization.optimizer.param_groups for parameter in group["params"]}
    assert base.weight.requires_grad  # thawed — not left frozen by inject's blanket base freeze
    assert id(base.weight) in optimised  # and the optimiser actually owns it, so the columns train
    assert not base.weight[:, -PriorVelocity.DIM :].any()  # the velocity columns still start at exactly zero


def test_build():
    """ORDER: widening replaces the projection object, so the optimiser must be built AFTER it, over the new one.

    Built the other way round the run still trains — it just never moves the live projection, in silence. The
    pin is identity: the parameter the optimiser holds IS the widened layer's weight, and its new columns
    start at exactly zero so the (pretrained) head's output is unchanged at step 0.
    """
    model, optimization = _assembly(_velocity_config()).build(warm_start=False)
    projection = cast(nn.Linear, model.transformer.proj)

    assert projection.in_features == _cpu_config().model.out_channels + 4 * _POS_EMBED_DIM + PriorVelocity.DIM
    optimised = {id(parameter) for group in optimization.optimizer.param_groups for parameter in group["params"]}
    assert id(projection.weight) in optimised
    assert not projection.weight[:, -PriorVelocity.DIM :].any()


def test_trained():
    """A head nobody optimises is a head that never learns: its parameters ride the optimiser and the clip."""
    assembly = _assembly(_contrastive_config(ContrastiveSite.DETACHED_PROJECTION))
    model, optimization = assembly.build(warm_start=False)

    optimised = {id(parameter) for group in optimization.optimizer.param_groups for parameter in group["params"]}
    head = list(assembly.contrastive.parameters())

    assert head and {id(parameter) for parameter in head} <= optimised
    assert len(assembly.trained(model)) == len(list(model.parameters())) + len(head)


def test_freeze_backbone_norm():
    """BatchNorm in the detector goes to eval mode so it normalises by the warm pack's running stats, not the
    noisy batch — while its affine weight keeps `requires_grad` and still trains. Fed a detector carrying a real
    BatchNorm (the conftest stub-backbone has none), so the eval-mode + affine-thaw contract is what is pinned."""
    batchnorm = nn.BatchNorm3d(2)
    model = SimpleNamespace(detector=nn.Sequential(nn.Conv3d(2, 2, 1), batchnorm))
    assert batchnorm.training  # a fresh BatchNorm trains by default

    JointAssembly.freeze_backbone_norm(cast(JointModel, model))

    assert not batchnorm.training  # now normalises by stored running stats
    assert batchnorm.weight.requires_grad  # but the affine weight still trains


def test_velocity():
    """The history feed is config-derived: on when `--prior-velocity` is set, off otherwise, without a rebuild."""
    assert _assembly(_velocity_config()).velocity.enabled
    assert not _assembly(_cpu_config()).velocity.enabled
