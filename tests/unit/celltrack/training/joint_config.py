"""Unit tests for the joint run's config hierarchy — its defaults, its bounds, and its two scales.

Two things are pinned here. FIRST the defaults themselves: they are measured values (a leaderboard A/B, a
proxy A/B, a published recipe), so a silent drift in one of them is a shipped experiment nobody ran — the
test states each number and the instrument behind it. SECOND the epoch/step resolution, which is the only
behaviour the config carries beyond holding values.
"""

import pytest
from pydantic import ValidationError

from celltrack.losses.softmax_focal_bce import SOURCE_AXIS, TARGET_AXIS
from celltrack.operating_point import TrackerConfig
from celltrack.training.joint_cli import JointCli
from celltrack.training.joint_config import (
    WARM_PACKS,
    ContrastiveSite,
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

_CORPUS = 18024  # the real GT-pair count: one pair per step, so this is one epoch


def test_defaults_are_the_measured_recipe():
    """Every default is the value the trainer already ran with — NOT the published recipe's own numbers.

    lr 1e-3, det-weight 1e1 and threshold 0.99 are pilkwang's; ours are 1e-4 (a warm start passes 1e-5),
    1.0 (0.1 measured worse) and a permissive 0.5 selector threshold. A default nobody chose is not a
    decision anyone made, so each one is written down here rather than inherited.
    """
    config = JointTrainConfig()
    assert (config.model.out_channels, config.model.layers) == (32, (32, 64, 128))
    assert config.model.warm_pack == WARM_PACKS["seed1"]
    assert config.data.downsample == (1, 4, 4)
    assert (config.data.difficulty_sampling, config.data.prior_velocity) == (False, False)
    assert (config.optim.lr, config.optim.grad_clip, config.optim.cosine_lr) == (1e-4, 1.0, False)
    assert (config.loss.det_weight, config.loss.neg_weight) == (1.0, 1e-2)
    assert (config.loss.contrastive_weight, config.loss.temperature) == (0.0, 0.07)
    # The measured-bad placement stays the default BECAUSE the weight is zero: an unasked run is yesterday's.
    assert config.loss.contrastive_site is ContrastiveSite.FEATURES
    assert config.loss.ignore_ambiguous_above is None
    # The selector's threshold TRACKS the shipped tracker's rather than pinning a literal: selection must
    # rank candidates the way deployment does, and a copied constant would silently drift from it.
    # The selector runs the WHOLE shipped operating point, not a threshold on neutral defaults: ranking
    # checkpoints under a linker we do not deploy can discard the better model (an affinity-side change flips
    # sign between the per-frame and the global linker).
    assert (config.eval.tracker, config.eval.tta) == (TrackerConfig.shipped(), False)
    assert (config.schedule.steps, config.schedule.eval_every, config.schedule.patience) == (1500, 500, 5)
    assert config.schedule.es_min_delta == 0.0
    assert (config.schedule.epochs, config.schedule.evals_per_epoch, config.schedule.patience_epochs) == (None,) * 3
    assert (config.runtime.device, config.runtime.seed) == ("cuda", 0)


def test_from_args():
    """The parser's flat flags fan out into the concern each one belongs to — one override per group proven.

    `from_args` is the CLI-to-model seam: a flag lands in exactly one group and nowhere else. Overriding one
    flag per group and reading it back where it belongs catches a mis-wire (a flag routed to the wrong Cfg)
    that byte-identical defaults would hide.
    """
    args = JointCli.build_parser().parse_args(
        [
            "--lr",
            "0.005",
            "--velocity-gt-warmup-steps",
            "300",
            "--det-weight",
            "7.0",
            "--steps",
            "42",
            "--aug-brightness",
            "0.2",
            "--aug-flip-axes",
            "1",
            "2",
        ]
    )
    config = JointTrainConfig.from_args(args)
    assert config.optim.lr == 0.005
    assert config.data.velocity_gt_warmup_steps == 300
    assert config.data.aug_brightness == 0.2
    assert config.data.aug_flip_axes == (1, 2)
    assert config.loss.det_weight == 7.0
    assert config.schedule.steps == 42
    assert config.eval.tracker.threshold == args.eval_threshold  # tracked off the shipped operating point
    assert config.model.warm_pack == WARM_PACKS[args.warm_pack]


def test_data_cfg_augmentation():
    """`DataCfg.augmentation()` hands the dataset the policy its fields describe — identity when nothing is asked."""
    aug = DataCfg(aug_brightness=0.3, aug_offset=1.5, aug_flip_axes=(1, 2)).augmentation()
    assert (aug.brightness, aug.offset, aug.flip_axes) == (0.3, 1.5, (1, 2))
    assert DataCfg().augmentation().flip_axes == ()  # un-augmented by default


def test_joint_train_config_augmentation():
    """The run delegates to its data group, so the trainer asks the run and never reaches through it (demeter)."""
    run = JointTrainConfig(data=DataCfg(aug_flip_axes=(1, 2)))
    assert run.augmentation().flip_axes == (1, 2)


def test_to_config():
    """LoraCfg hands the trainer the celltrack.models.lora config it configures — same rank, alpha, targets."""
    config = LoraCfg(rank=8, alpha=32.0, targets=("transformer", "decoder_blocks")).to_config()
    assert config.rank == 8
    assert config.alpha == 32.0
    assert config.targets == ("transformer", "decoder_blocks")
    assert config.scaling() == 4.0


def test_to_tracking_loss_config():
    """LossCfg fills the loss's own decoupled config — its knobs pass through, and the crowd gate is resolved."""
    config = LossCfg(symmetric_links=True, neg_weight=0.3, det_weight=2.0).to_tracking_loss_config((1, 2, 2))
    assert config.link_axes == (SOURCE_AXIS, TARGET_AXIS)
    assert config.neg_weight == 0.3
    assert config.det_weight == 2.0
    assert config.downsample == (1, 2, 2)
    assert config.gate_um == TrackerConfig.shipped().linker.gate_um  # resolved from the shipped point, not a field


def test_projects():
    """Only the measured-bad site reads the features directly; both fixed placements interpose a head."""
    assert not ContrastiveSite.FEATURES.projects
    assert ContrastiveSite.PROJECTION.projects and ContrastiveSite.DETACHED_PROJECTION.projects


def test_stop_gradient():
    """The trunk is spared at exactly one site — the strict form of the fix, one field from the other two."""
    assert ContrastiveSite.DETACHED_PROJECTION.stop_gradient
    assert not ContrastiveSite.PROJECTION.stop_gradient
    assert not ContrastiveSite.FEATURES.stop_gradient


def test_bounds_reject_a_value_that_cannot_mean_what_it_says():
    """The groups are validated models, not bags: a rate of zero or a probability of 3 fails at construction."""
    with pytest.raises(ValidationError):
        OptimCfg(lr=0.0)
    with pytest.raises(ValidationError):
        LossCfg(temperature=0.0)
    with pytest.raises(ValidationError):
        LossCfg(ignore_ambiguous_above=3.0)
    with pytest.raises(ValidationError):
        EvalCfg(threshold=1.5)
    with pytest.raises(ValidationError):
        DataCfg(downsample=(0, 4, 4))
    with pytest.raises(ValidationError):
        ScheduleCfg(steps=0)


def test_unknown_field_is_rejected():
    """`extra="forbid"`: a mistyped knob is a construction error, not a value the run silently ignores."""
    with pytest.raises(ValidationError):
        JointTrainConfig(lr=1e-5)  # the flat name — it belongs to `optim` now, and the typo must say so


def test_json_round_trip():
    """The whole run is one JSON document, and reading it back gives the same config — provenance, validated."""
    config = JointTrainConfig(
        optim=OptimCfg(lr=1e-5),
        loss=LossCfg(ignore_ambiguous_above=0.97, contrastive_site=ContrastiveSite.DETACHED_PROJECTION),
        runtime=RuntimeCfg(device="cpu"),
    )
    assert JointTrainConfig.model_validate_json(config.model_dump_json()) == config


def test_model_dump_flattens_to_dotted_mlflow_params():
    """`Tracker._flat` turns the nesting into `group.knob` params, so every knob stays queryable across runs."""
    dumped = JointTrainConfig().model_dump()
    assert dumped["loss"]["det_weight"] == 1.0
    assert dumped["schedule"]["steps"] == 1500


def test_schedule_cfg_resolved():
    """No epoch field set means today's run: the step-scale numbers a caller passed reach the loop unchanged."""
    schedule = ScheduleCfg(steps=3000, eval_every=500, patience=5)
    assert schedule.resolved(_CORPUS) == schedule


def test_joint_train_config_resolved():
    """Resolving the run is resolving its schedule — an untouched one leaves the whole config equal to itself."""
    config = JointTrainConfig(schedule=ScheduleCfg(steps=3000, eval_every=500, patience=5))
    assert config.resolved(_CORPUS) == config


def test_resolved_turns_epochs_into_steps():
    """`epochs` is passes over the corpus, so the budget is that many times the derived steps-per-epoch."""
    assert ScheduleCfg(epochs=1.5).resolved(_CORPUS).steps == 27036


def test_resolved_turns_evals_per_epoch_into_a_window_length():
    """Four evals per epoch is a window of a quarter of the corpus, whatever the split's size happens to be."""
    assert ScheduleCfg(evals_per_epoch=4).resolved(_CORPUS).eval_every == 4506


def test_resolved_turns_patience_epochs_into_windows():
    """Patience counts EVAL WINDOWS, so an epoch of patience is however many windows that epoch holds."""
    assert ScheduleCfg(patience_epochs=2.0, evals_per_epoch=3).resolved(_CORPUS).patience == 6

    from_raw_windows = ScheduleCfg(patience_epochs=1.0, eval_every=9012).resolved(_CORPUS)
    assert from_raw_windows.patience == 2  # no evals_per_epoch: the window count follows the raw eval_every


def test_resolved_epoch_fields_win_over_their_raw_twins():
    """Both forms given is not an error and neither is dropped in silence — the epoch form wins, and it is logged."""
    schedule = ScheduleCfg(steps=3000, eval_every=500, patience=5, epochs=2.0, evals_per_epoch=2, patience_epochs=3.0)

    resolved = JointTrainConfig(schedule=schedule).resolved(_CORPUS).schedule

    assert (resolved.steps, resolved.eval_every, resolved.patience) == (36048, 9012, 6)
    assert schedule._overrides() == [
        "epochs=2.0 overrides steps=3000",
        "evals_per_epoch=2 overrides eval_every=500",
        "patience_epochs=3.0 overrides patience=5",
    ]


def test_resolved_keeps_a_window_when_the_corpus_is_smaller_than_the_eval_count():
    """A corpus of three pairs asked for ten evals would floor to a zero-step window; it floors to one instead."""
    resolved = ScheduleCfg(evals_per_epoch=10, patience_epochs=0.01).resolved(3)
    assert resolved.eval_every == 1
    assert resolved.patience == 1


def test_resolved_leaves_every_other_group_untouched():
    """Resolving is a schedule operation — the objective, the model and the runtime come through identical."""
    config = JointTrainConfig(model=ModelCfg(out_channels=2), loss=LossCfg(det_weight=0.5))

    resolved = config.resolved(_CORPUS)

    assert (resolved.model, resolved.loss, resolved.data, resolved.optim) == (
        config.model,
        config.loss,
        config.data,
        config.optim,
    )


def test_link_axes():
    """Which axes the link loss normalises over — sources always, targets only when a run asks.

    The default is the frontier's form and yesterday's numbers. Asking adds the target axis, which is what
    gives a single-source pair a gradient at all: over sources alone its softmax is identically 1.0.
    """
    assert LossCfg().link_axes == (SOURCE_AXIS,)
    assert LossCfg(symmetric_links=True).link_axes == (SOURCE_AXIS, TARGET_AXIS)
