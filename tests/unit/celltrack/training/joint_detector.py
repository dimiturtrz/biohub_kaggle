"""Unit tests for the joint-trainer loop this repo owns — a tiny CPU end-to-end over synthetic GT pairs.

The loss parts and the pair dataset are covered elsewhere; here the whole loop is run for two steps on the
shared tiny video fixture with a minimal (non-pilkwang-sized) backbone and the stubbed transformer, so a
real forward/backward/eval/save cycle over both heads is exercised without the GPU or the `external/` dep.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import zarr
from torch import nn

from celltrack.data.difficulty_sampler import DifficultySampler
from celltrack.data.frame_source import ZarrFrames
from celltrack.data.joint_dataset import PairDataset, PairOptions, PairTarget
from celltrack.data.pair_curriculum import FixedMixture, PacedMixture
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TestMovieProxy
from celltrack.losses.tracking_loss import _PairOutcome
from celltrack.models.edge_transformer import _POS_EMBED_DIM
from celltrack.models.joint_model import JointModel
from celltrack.models.prior_velocity import PriorVelocity
from celltrack.operating_point import TrackerConfig
from celltrack.training.joint_checkpoint import RESUME_SUFFIX
from celltrack.training.joint_config import (
    ContrastiveSite,
    DataCfg,
    JointTrainConfig,
    LossCfg,
    ModelCfg,
    OptimCfg,
    RuntimeCfg,
    ScheduleCfg,
)
from celltrack.training.joint_detector import JointTrainer
from celltrack.training.pair_split import PairSplit
from celltrack.training.run_tracking import RunSetup, TrainingSplit
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.geometry import Spacing
from tests.unit.celltrack.conftest import RecordingMlflow

_CORPUS = 18024  # the real GT-pair count: one pair per step, so this is one epoch


def test_schedule_spans_the_resolved_step_count():
    """The cosine horizon is built from the RESOLVED budget — resolving epochs late must not leave it stale."""
    config = JointTrainConfig(schedule=ScheduleCfg(steps=10, epochs=2.0), optim=OptimCfg(cosine_lr=True)).resolved(
        _CORPUS
    )
    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-4)

    schedule = JointTrainer(config)._schedule(optimizer)

    assert schedule is not None
    assert schedule.T_max == 36048  # type: ignore[missing-attribute]


def _outcome(edge_logits: torch.Tensor, edge_matrix: torch.Tensor) -> _PairOutcome:
    """A pair outcome whose loss terms are placeholders — `test_observe` reads only its edge tensors."""
    zero = torch.zeros(())
    return _PairOutcome(torch.tensor(3.0), zero, zero, zero, zero, edge_logits, edge_matrix)


def _cpu_config() -> JointTrainConfig:
    """A tiny CPU config: a two-level backbone, no downsample, one eval-sized window of two steps."""
    return JointTrainConfig(
        model=ModelCfg(out_channels=2, layers=(2, 4)),
        data=DataCfg(downsample=(1, 1, 1)),
        runtime=RuntimeCfg(device="cpu"),
        schedule=ScheduleCfg(steps=2, eval_every=2, patience=2),
    )


def _pairs(video_store: Path) -> list[PairTarget]:
    """Two synthetic pairs whose full-resolution centres sit inside the tiny `video_store` volume (Z=2, Y=4, X=4)."""
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    q_low, q_high = float(quantiles["0.001"]), float(quantiles["0.999"])
    return [
        PairTarget(
            frames=ZarrFrames(video_store, q_low, q_high),
            timepoint=0,
            source_centres=np.array([[1, 2, 2]], dtype=np.int64),
            target_centres=np.array([[1, 2, 2], [0, 1, 1]], dtype=np.int64),
            edge_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
        ),
        PairTarget(
            frames=ZarrFrames(video_store, q_low, q_high),
            timepoint=1,
            source_centres=np.array([[0, 0, 0], [1, 3, 3]], dtype=np.int64),
            target_centres=np.array([[1, 3, 3]], dtype=np.int64),
            edge_matrix=np.array([[0.0], [1.0]], dtype=np.float32),
            previous_centres=np.array([[1, 2, 2]], dtype=np.int64),  # the pair at t=0 is this one's history
        ),
    ]


def _evaluator(video_store: Path, in_bounds_tracks: AnnotatedTracks) -> ModelEvaluator:
    """The real pipeline selector over a one-movie proxy — CPU, permissive threshold, no length pruning."""
    return ModelEvaluator(
        proxy=TestMovieProxy(paths=(video_store,), truths=(in_bounds_tracks,), spacing=Spacing(z=1.0, y=1.0, x=1.0)),
        edge_scorer=cast(BlendedEdgeTransformerScorer, None),  # unused: a joint eval mounts the model's own head
        recipe=DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False),
        device="cpu",
        config=TrackerConfig(threshold=0.0, min_track_length=1),
    )


def test_curriculum(video_store: Path):
    """A policy exists only when asked for, and is sized to the pair corpus rather than to the step budget."""
    split = PairSplit(_pairs(video_store), [])
    assert JointTrainer(_cpu_config())._curriculum(split) is None

    sampler = JointTrainer(_sampling_config())._curriculum(split)

    assert isinstance(sampler, DifficultySampler)
    sampler.observe(0, 0.5)
    assert sampler.coverage() == 0.5  # one of the corpus's two pairs, not one of the run's two steps


def test_curriculum_mixes_when_a_synthetic_population_is_present(video_store: Path):
    """A synthetic share picks the fixed mixture; asking for pacing picks the controller instead."""
    targets = _pairs(video_store)
    split = PairSplit(targets, [], targets)
    mixed = _cpu_config().model_copy(update={"data": DataCfg(downsample=(1, 1, 1), synthetic_fraction=0.5)})
    paced = _cpu_config().model_copy(update={"data": DataCfg(downsample=(1, 1, 1), paced_curriculum=True)})

    assert isinstance(JointTrainer(mixed)._curriculum(split), FixedMixture)
    assert isinstance(JointTrainer(paced)._curriculum(split), PacedMixture)
    # ...and with no synthetic pairs to draw, neither can be built, however the flags read.
    assert JointTrainer(mixed)._curriculum(PairSplit(targets, [])) is None


def test_pairs_leaves_the_uniform_stream_untouched(video_store: Path):
    """With no sampler the window walks exactly `PairDataset.stream()`, index-free — the default path is unchanged."""
    dataset = PairDataset(_pairs(video_store), steps=3, downsample=(1, 1, 1), seed=0)
    walked = list(JointTrainer(_cpu_config())._pairs(dataset, None))

    assert [draw.index for draw, _ in walked] == [-1, -1, -1]
    assert {draw.weight for draw, _ in walked} == {1.0}
    for (_, pair), expected in zip(walked, dataset.stream(), strict=True):
        assert torch.equal(pair.frame_t, expected.frame_t)
        assert torch.equal(pair.edge_matrix, expected.edge_matrix)


def test_pairs_draws_through_the_sampler(video_store: Path):
    """With a sampler the window walks the drawn indices, and each pair is the corpus entry that index names."""
    dataset = PairDataset(_pairs(video_store), steps=4, downsample=(1, 1, 1), seed=0)
    sampler = DifficultySampler(count=2, seed=0)
    sampler.observe(0, 0.0)
    reference = DifficultySampler(count=2, seed=0)
    reference.observe(0, 0.0)

    walked = list(JointTrainer(_cpu_config())._pairs(dataset, sampler))

    assert [draw.index for draw, _ in walked] == [reference.draw() for _ in range(4)]
    for draw, pair in walked:
        assert torch.equal(pair.edge_matrix, dataset.pair(draw.index).edge_matrix)


def test_pairs_prefetches_without_changing_the_draw_sequence(video_store: Path):
    """A window-fed policy (read_ahead >= 1) walks the SAME indices, decoded a step ahead, as in lock-step."""
    dataset = PairDataset(_pairs(video_store), steps=6, downsample=(1, 1, 1), seed=0)
    mixture = FixedMixture(1, 1, 0.5, seed=0)
    reference = FixedMixture(1, 1, 0.5, seed=0)

    walked = list(JointTrainer(_cpu_config())._pairs(dataset, mixture))

    assert mixture.read_ahead() >= 1
    assert [draw.index for draw, _ in walked] == [reference.step().index for _ in range(6)]
    for draw, pair in walked:
        assert torch.equal(pair.edge_matrix, dataset.pair(draw.index).edge_matrix)


def test_observe(video_store: Path):
    """The step's own top-1 defect reaches the sampler; with no sampler the feedback call is a no-op."""
    sampler = DifficultySampler(count=2, seed=0)
    outcome = _outcome(torch.tensor([[0.0, 9.0]]), torch.tensor([[1.0, 0.0]]))  # the one annotated source is missed
    trainer = JointTrainer(_cpu_config())

    trainer._observe(None, -1, outcome)
    trainer._observe(sampler, -1, outcome)  # the uniform stream draws no index; the sampler must not be touched
    assert sampler.coverage() == 0.0
    trainer._observe(sampler, 0, outcome)

    assert sampler.coverage() == 0.5
    assert outcome.difficulty() == 1.0


def test_health():
    """Each policy names its own diagnostics, and an unasked run contributes none — the rows of yesterday."""
    assert DifficultySampler(count=4, seed=0).health() == {"coverage": 0.0, "sampling_entropy": 1.0}
    assert set(FixedMixture(2, 2, 0.5, seed=0).health()) == {"synthetic_share"}
    assert set(PacedMixture(2, 2, 4, seed=0).health()) == {"difficulty", "synthetic_weight", "real_weight"}


def test_train(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """The loop runs end to end — sample, step, pipeline eval, edge AUC, save — and returns the best proxy score."""
    torch.manual_seed(0)
    targets = _pairs(video_store)
    save_to = tmp_path / "joint.pt"

    best = JointTrainer(_cpu_config()).train(
        PairSplit(targets, targets), _evaluator(video_store, in_bounds_tracks), RunSetup(save_to)
    )

    assert isinstance(best, float)
    assert save_to.exists()


def test_train_records_each_window(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path, mlflow_backend: RecordingMlflow
):
    """The joint loop is wired to the tracker, and keeps each loss term and each eval lever as its own series."""
    torch.manual_seed(0)
    targets = _pairs(video_store)
    JointTrainer(_cpu_config()).train(
        PairSplit(targets, targets),
        _evaluator(video_store, in_bounds_tracks),
        RunSetup(tmp_path / "joint.pt", split=TrainingSplit(191, 4, 4)),
    )
    assert mlflow_backend.logged_params()["loss.det_weight"] == 1.0
    assert {
        "proxy_score",
        "selection_score",
        "best_score",
        "node_recall",
        "node_ratio",
        "edge_auc",
        "inverted_fraction",
        "mean_p_true",
        "mean_p_chosen",
        "mislinks",
        "train_loss",
        "edge_loss",
        "det_loss",
        "contrastive_loss",
        "it_per_s",
    } <= mlflow_backend.metric_keys()
    steps = {call[3] for call in mlflow_backend.calls if call[0] == "metric"}
    assert steps == {0, 2}  # the init eval at step 0, then the single two-step window
    assert ("end",) in mlflow_backend.calls
    assert {"coverage", "sampling_entropy"} & mlflow_backend.metric_keys() == set()  # off: no sampler row


def test_train_runs_the_epoch_schedule(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path, mlflow_backend: RecordingMlflow
):
    """Asked for two epochs over a two-pair corpus, the loop runs four steps in one-pair windows."""
    torch.manual_seed(0)
    targets = _pairs(video_store)
    schedule = ScheduleCfg(steps=2, eval_every=2, patience=0, epochs=2.0, evals_per_epoch=2)
    config = _cpu_config().model_copy(update={"schedule": schedule})

    JointTrainer(config).train(
        PairSplit(targets, targets), _evaluator(video_store, in_bounds_tracks), RunSetup(tmp_path / "joint.pt")
    )

    steps = {call[3] for call in mlflow_backend.calls if call[0] == "metric"}
    assert steps == {0, 1, 2, 3, 4}


def _trained_bytes(config: JointTrainConfig, video_store: Path, tracks: AnnotatedTracks, run_dir: Path) -> bytes:
    """One full run under `config`, from the same torch seed, returned as the raw checkpoint bytes.

    Each run gets its own directory under the SAME checkpoint name: torch's zip archive records the file stem,
    so two names would differ in the bytes for a reason that has nothing to do with the weights.
    """
    run_dir.mkdir()
    save_to = run_dir / "joint.pt"
    torch.manual_seed(0)
    targets = _pairs(video_store)
    JointTrainer(config).train(PairSplit(targets, targets), _evaluator(video_store, tracks), RunSetup(save_to))
    return save_to.read_bytes()


def _trained_weights(config: JointTrainConfig, video_store: Path, tracks: AnnotatedTracks, run_dir: Path) -> bytes:
    """One full run's TRAINED weights, off the resume snapshot — what the loop did, not what it chose to keep.

    `_trained_bytes` reads the best-checkpoint, which `save_best` fills with the INIT when no window improves
    on it. Comparing two such files can therefore compare two untrained models and read as agreement. The
    snapshot is written every window regardless of the score, so both heads' tensors here are the run's own.
    """
    run_dir.mkdir()
    save_to = run_dir / "joint.pt"
    torch.manual_seed(0)
    targets = _pairs(video_store)
    JointTrainer(config).train(PairSplit(targets, targets), _evaluator(video_store, tracks), RunSetup(save_to))
    state = torch.load(save_to.with_suffix(RESUME_SUFFIX), map_location="cpu", weights_only=False)
    heads = (state["detector_state"], state["transformer_state"])
    return b"".join(tensor.numpy().tobytes() for head in heads for tensor in head.values())


def test_train_is_reproducible(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """The same config twice from the same seed writes the same checkpoint bytes — the loop is deterministic.

    This is the anchor the config restructure is measured against: nothing about grouping the knobs may make
    a run depend on anything but its values, so a differing byte here is a real behaviour change.
    """
    first = _trained_bytes(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "first")
    assert _trained_bytes(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "second") == first


def test_train_off_is_byte_identical(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """Difficulty sampling off reproduces the uniform run exactly — same draws, same weights, same bytes."""
    default = _trained_bytes(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "default")
    explicit_off = _cpu_config().model_copy(update={"data": DataCfg(downsample=(1, 1, 1), difficulty_sampling=False)})
    assert _trained_bytes(explicit_off, video_store, in_bounds_tracks, tmp_path / "off") == default


def test_train_batched_default_is_not_the_per_pair_path(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path
):
    """The batched default actually ENGAGES batching — its weights differ from the explicit per-pair (batch 1) run.

    A default that silently fell back to per-pair would train identically to `batch_size=1` and the utilisation
    win would be a lie. Reading the resume snapshot (not the best-checkpoint, which `save_best` fills with the
    INIT when no window improves) makes a byte difference here a real difference in what the loop trained.
    """
    default = _trained_weights(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "default")
    per_pair = _cpu_config().model_copy(update={"optim": OptimCfg(lr=1e-4, batch_size=1)})
    assert _trained_weights(per_pair, video_store, in_bounds_tracks, tmp_path / "one") != default


def test_train_accumulate_averages_instead_of_stepping(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path
):
    """Accumulating changes the weights the run TRAINS — the pairs are averaged into one update, not applied singly.

    Anchored on two EXPLICIT batch sizes rather than the default, so the invariant (a larger batch averages into
    a different update) is independent of whatever the default happens to be. Read off the resume snapshot rather
    than the best-checkpoint, and that choice is the point: `save_best` writes the INIT whenever no window beats
    it, so two arms that both fail to improve produce identical files whatever they did to the weights. A
    byte-identity assertion over the best-checkpoint can therefore pass by comparing one untrained model with
    another. The snapshot holds the trained weights unconditionally, so a difference here is a real one.
    """
    single = _cpu_config().model_copy(update={"optim": OptimCfg(lr=1e-4, batch_size=1)})
    batched = _cpu_config().model_copy(update={"optim": OptimCfg(lr=1e-4, batch_size=2)})
    trained_single = _trained_weights(single, video_store, in_bounds_tracks, tmp_path / "single")
    assert _trained_weights(batched, video_store, in_bounds_tracks, tmp_path / "batched") != trained_single


def _velocity_config() -> JointTrainConfig:
    """The tiny CPU config with the prior-velocity feature on — the only difference from the default run."""
    return _cpu_config().model_copy(update={"data": DataCfg(downsample=(1, 1, 1), prior_velocity=True)})


def _sampling_config() -> JointTrainConfig:
    """The tiny CPU config drawing its pairs by difficulty — the only difference from the default run."""
    return _cpu_config().model_copy(update={"data": DataCfg(downsample=(1, 1, 1), difficulty_sampling=True)})


def test_prepare_widens_the_head_before_building_the_optimizer(video_store: Path):
    """ORDER: widening replaces the projection object, so the optimiser must be built AFTER it, over the new one.

    Built the other way round the run still trains — it just never moves the live projection, in silence. The
    pin is identity: the parameter the optimiser holds IS the widened layer's weight, and its new columns
    start at exactly zero so the (pretrained) head's output is unchanged at step 0.
    """
    model, optimization = JointTrainer(_velocity_config())._prepare(warm_start=False)
    projection = cast(nn.Linear, model.transformer.proj)

    assert projection.in_features == _cpu_config().model.out_channels + 4 * _POS_EMBED_DIM + PriorVelocity.DIM
    optimised = {id(parameter) for group in optimization.optimizer.param_groups for parameter in group["params"]}
    assert id(projection.weight) in optimised
    assert not projection.weight[:, -PriorVelocity.DIM :].any()


def test_losses_hands_the_velocity_to_the_model(video_store: Path, monkeypatch: pytest.MonkeyPatch):
    """The scored pair reaches `JointModel.forward` carrying a real, non-zero prior displacement; the batched
    no-grad pass that produced it forwards ZERO velocity for its own sources (one step back, no recursion)."""
    trainer = JointTrainer(_velocity_config())
    model, _ = trainer._prepare(warm_start=False)
    sample = PairDataset(_pairs(video_store), 1, (1, 1, 1), 0, PairOptions(with_previous=True)).pair(1)
    scored: list[torch.Tensor | None] = []
    gap_velocities: list[torch.Tensor] = []
    forward, forward_batch = JointModel.forward, JointModel.forward_batch

    def _recording(self: JointModel, *args: object, **kwargs: object) -> object:
        scored.append(cast(torch.Tensor | None, args[4] if len(args) > 4 else kwargs.get("source_velocity")))
        return forward(self, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    def _recording_batch(self: JointModel, *args: object, **kwargs: object) -> object:
        gap_velocities.extend(cast(list[torch.Tensor], args[3] if len(args) > 3 else kwargs["velocities"]))
        return forward_batch(self, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    monkeypatch.setattr(JointModel, "forward", _recording)
    monkeypatch.setattr(JointModel, "forward_batch", _recording_batch)

    trainer._losses(model, sample)

    assert len(scored) == 1  # only the scored pair reaches JointModel.forward — the gap pass is batched
    velocity = scored[0]
    assert velocity is not None and velocity.abs().sum() > 0  # the scored pair carries the prior displacement
    assert gap_velocities and all(not v.any() for v in gap_velocities)  # the previous gap carries no history


def test_train_prior_velocity_off_is_byte_identical(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path
):
    """Off explicitly reproduces the default run exactly — same reads, same head width, same weights, same bytes."""
    default = _trained_bytes(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "default")
    explicit_off = _cpu_config().model_copy(update={"data": DataCfg(downsample=(1, 1, 1), prior_velocity=False)})
    assert _trained_bytes(explicit_off, video_store, in_bounds_tracks, tmp_path / "off") == default


def _contrastive_config(site: ContrastiveSite) -> JointTrainConfig:
    """The tiny CPU config with the contrastive term ON at one placement — the only difference from the default."""
    return _cpu_config().model_copy(update={"loss": LossCfg(contrastive_weight=0.1, contrastive_site=site)})


def test_train_contrastive_off_is_byte_identical(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """The term off is the run of yesterday: same weights, same bytes — the placement work costs an unasked run nothing.

    Off is the DEFAULT site at weight zero; the projecting sites build a head, and building one draws from
    the RNG, so byte-identity is claimed for exactly the configuration that ships.
    """
    default = _trained_bytes(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "default")
    explicit_off = _cpu_config().model_copy(
        update={"loss": LossCfg(contrastive_weight=0.0, contrastive_site=ContrastiveSite.FEATURES)}
    )
    assert _trained_bytes(explicit_off, video_store, in_bounds_tracks, tmp_path / "off") == default


def _contrastive_gradients(config: JointTrainConfig, video_store: Path) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Backward the CONTRASTIVE term alone through a real pair; returns a backbone gradient and the head's.

    Alone is the point: the detection and edge terms reach the backbone by design, so a total-loss backward
    could not tell whether the contrastive gradient was among them.
    """
    torch.manual_seed(0)
    trainer = JointTrainer(config)
    model, _ = trainer._prepare(warm_start=False)
    sample = PairDataset(_pairs(video_store), 1, (1, 1, 1), 0).pair(0)

    trainer._losses(model, sample).contrastive.backward()

    backbone = next(parameter for parameter in model.detector.unet.parameters() if parameter.requires_grad)
    head = [parameter.grad for parameter in trainer.contrastive.parameters() if parameter.grad is not None]
    return backbone, head


def test_losses_detached_leaves_the_backbone_out_of_the_contrastive_term(video_store: Path):
    """The diagnosed fix in its strict form: through a detached head the term trains the embedding only."""
    backbone, head = _contrastive_gradients(_contrastive_config(ContrastiveSite.DETACHED_PROJECTION), video_store)

    assert backbone.grad is None or backbone.grad.abs().sum() == 0
    assert head and any(gradient.abs().sum() > 0 for gradient in head)


def test_losses_projected_still_reaches_the_backbone(video_store: Path):
    """The comparison arm, one field away: the head is interposed but the trunk is still shaped through it."""
    backbone, _ = _contrastive_gradients(_contrastive_config(ContrastiveSite.PROJECTION), video_store)

    assert backbone.grad is not None
    assert backbone.grad.abs().sum() > 0


def test_train_carries_the_projection_head_through_the_resume_snapshot(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path
):
    """A projecting run trains end to end, and its head is IN the snapshot — the optimiser has slots for it.

    Without this the resumed run rebuilds a random head under an optimiser state that expects the trained
    one, which no error reports: the contrastive term would silently restart from noise mid-run.
    """
    torch.manual_seed(0)
    targets = _pairs(video_store)
    save_to = tmp_path / "joint.pt"
    trainer = JointTrainer(_contrastive_config(ContrastiveSite.DETACHED_PROJECTION))

    trainer.train(PairSplit(targets, targets), _evaluator(video_store, in_bounds_tracks), RunSetup(save_to))

    snapshot = torch.load(save_to.with_suffix(".resume.pt"), weights_only=False)["contrastive"]
    assert set(snapshot) == set(trainer.contrastive.state_dict())
    assert all(torch.equal(snapshot[name], value) for name, value in trainer.contrastive.state_dict().items())


def test_prepare_hands_the_projection_head_to_the_optimizer(video_store: Path):
    """A head nobody optimises is a head that never learns: its parameters ride the optimiser and the clip."""
    trainer = JointTrainer(_contrastive_config(ContrastiveSite.DETACHED_PROJECTION))
    model, optimization = trainer._prepare(warm_start=False)

    optimised = {id(parameter) for group in optimization.optimizer.param_groups for parameter in group["params"]}
    head = list(trainer.contrastive.parameters())

    assert head and {id(parameter) for parameter in head} <= optimised
    assert len(trainer._trained(model)) == len(list(model.parameters())) + len(head)


def test_train_records_sampler_health(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path, mlflow_backend: RecordingMlflow
):
    """With sampling on, each window row carries the coverage and the collapse guard beside the loss terms."""
    torch.manual_seed(0)
    targets = _pairs(video_store)
    JointTrainer(_sampling_config()).train(
        PairSplit(targets, targets), _evaluator(video_store, in_bounds_tracks), RunSetup(tmp_path / "joint.pt")
    )
    assert {"coverage", "sampling_entropy"} <= mlflow_backend.metric_keys()


def test_train_hard_negatives_off_is_byte_identical(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path
):
    """The ranking term at weight zero is the run of yesterday: mining is computed and logged, and changes nothing."""
    default = _trained_bytes(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "default")
    explicit_off = _cpu_config().model_copy(update={"loss": LossCfg(hard_negative_weight=0.0, hard_negatives=4)})
    assert _trained_bytes(explicit_off, video_store, in_bounds_tracks, tmp_path / "off") == default


def test_losses_carries_the_hard_negative_term_into_the_total(video_store: Path):
    """Weighted on, the mined ranking penalty is non-zero, enters the total, and its gradient reaches the BACKBONE.

    Reaching the backbone is the whole experiment: every refuted association retrain here mined its negatives
    against FROZEN features, so a term that only moved the head would repeat that arm rather than test it.
    """
    torch.manual_seed(0)
    weighted = _cpu_config().model_copy(update={"loss": LossCfg(hard_negative_weight=1.0)})
    trainer = JointTrainer(weighted)
    model, _ = trainer._prepare(warm_start=False)
    sample = PairDataset(_pairs(video_store), 1, (1, 1, 1), 0).pair(0)

    outcome = trainer._losses(model, sample)

    assert outcome.hard_negative.item() > 0.0
    assert torch.isclose(outcome.total, outcome.edge + outcome.detection + outcome.hard_negative)
    outcome.hard_negative.backward()
    backbone = next(parameter for parameter in model.detector.unet.parameters() if parameter.requires_grad)
    assert backbone.grad is not None and backbone.grad.abs().sum() > 0
