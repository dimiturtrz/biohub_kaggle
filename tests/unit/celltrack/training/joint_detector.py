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

from celltrack.data.difficulty_sampler import DifficultySampler
from celltrack.data.joint_dataset import PairDataset, PairTarget
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TestMovieProxy
from celltrack.models.edge_transformer import _POS_EMBED_DIM
from celltrack.models.joint_model import JointModel
from celltrack.models.prior_velocity import PriorVelocity
from celltrack.tracker import TrackerConfig
from celltrack.training.joint_config import (
    DataCfg,
    JointTrainConfig,
    ModelCfg,
    OptimCfg,
    RuntimeCfg,
    ScheduleCfg,
)
from celltrack.training.joint_detector import JointTrainer, PairSplit, _PairOutcome
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
    """A pair outcome whose loss terms are placeholders — these tests are about the edge tensors beside them."""
    return _PairOutcome(
        total=torch.tensor(3.0, requires_grad=True),
        edge=torch.tensor(1.0),
        detection=torch.tensor(2.0),
        contrastive=torch.tensor(0.5),
        edge_logits=edge_logits,
        edge_matrix=edge_matrix,
    )


def test_logged():
    """Every term reaches the run's metric table under its own name, detached from the graph."""
    outcome = _outcome(torch.zeros(1, 1), torch.zeros(1, 1))
    assert outcome.logged() == {"train_loss": 3.0, "edge_loss": 1.0, "det_loss": 2.0, "contrastive_loss": 0.5}


def test_difficulty():
    """Hand-computed: of the two ANNOTATED sources one is ranked top-1 and one is not, so the defect is 1 - 1/2."""
    edge_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    edge_logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 5.0], [9.0, 0.0, 0.0]])  # row 2 is unannotated, ignored
    assert _outcome(edge_logits, edge_matrix).difficulty() == 0.5


def test_difficulty_is_none_without_annotated_sources():
    """A pair whose annotation links nothing has no defect to report, so the sampler is told to keep what it stored."""
    assert _outcome(torch.rand(2, 2), torch.zeros(2, 2)).difficulty() is None


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
            zarr_path=video_store,
            timepoint=0,
            q_low=q_low,
            q_high=q_high,
            source_centres=np.array([[1, 2, 2]], dtype=np.int64),
            target_centres=np.array([[1, 2, 2], [0, 1, 1]], dtype=np.int64),
            edge_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
        ),
        PairTarget(
            zarr_path=video_store,
            timepoint=1,
            q_low=q_low,
            q_high=q_high,
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


def test_sampler(video_store: Path):
    """A sampler exists only when asked for, and is sized to the pair corpus rather than to the step budget."""
    targets = _pairs(video_store)
    assert JointTrainer(_cpu_config())._sampler(targets) is None

    sampler = JointTrainer(_sampling_config())._sampler(targets)

    assert sampler is not None
    sampler.observe(0, 0.5)
    assert sampler.coverage() == 0.5  # one of the corpus's two pairs, not one of the run's two steps


def test_pairs_leaves_the_uniform_stream_untouched(video_store: Path):
    """With no sampler the window walks exactly `PairDataset.stream()`, index-free — the default path is unchanged."""
    dataset = PairDataset(_pairs(video_store), steps=3, downsample=(1, 1, 1), seed=0)
    walked = list(JointTrainer(_cpu_config())._pairs(dataset, None))

    assert [index for index, _ in walked] == [None, None, None]
    for (_, pair), expected in zip(walked, dataset.stream(), strict=True):
        assert torch.equal(pair.frame_t, expected.frame_t)
        assert torch.equal(pair.edge_matrix, expected.edge_matrix)


def test_pairs_draws_through_the_sampler(video_store: Path):
    """With a sampler the window walks the drawn indices, and each pair is the corpus entry that index names."""
    dataset = PairDataset(_pairs(video_store), steps=4, downsample=(1, 1, 1), seed=0)
    sampler = DifficultySampler(count=2, seed=0)
    sampler.observe(0, 0.0)  # pair 0 solved, pair 1 still optimistic -> every draw is pair 1

    walked = list(JointTrainer(_cpu_config())._pairs(dataset, sampler))

    assert [index for index, _ in walked] == [1, 1, 1, 1]
    assert torch.equal(walked[0][1].edge_matrix, dataset.pair(1).edge_matrix)


def test_observe(video_store: Path):
    """The step's own top-1 defect reaches the sampler; with no sampler the feedback call is a no-op."""
    sampler = DifficultySampler(count=2, seed=0)
    outcome = _outcome(torch.tensor([[0.0, 9.0]]), torch.tensor([[1.0, 0.0]]))  # the one annotated source is missed
    trainer = JointTrainer(_cpu_config())

    trainer._observe(None, None, outcome)
    trainer._observe(sampler, 0, outcome)

    assert sampler.coverage() == 0.5
    assert outcome.difficulty() == 1.0


def test_sampling():
    """The health numbers ride the window's metric dict only when the sampler is on — an unasked run logs as before."""
    trainer = JointTrainer(_cpu_config())
    assert trainer._sampling(None) == {}
    assert trainer._sampling(DifficultySampler(count=4, seed=0)) == {"coverage": 0.0, "sampling_entropy": 1.0}


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
    projection = model.transformer.proj

    assert projection.in_features == _cpu_config().model.out_channels + 4 * _POS_EMBED_DIM + PriorVelocity.DIM
    optimised = {id(parameter) for group in optimization.optimizer.param_groups for parameter in group["params"]}
    assert id(projection.weight) in optimised
    assert not projection.weight[:, -PriorVelocity.DIM :].any()


def test_losses_hands_the_velocity_to_the_model(video_store: Path, monkeypatch: pytest.MonkeyPatch):
    """A pair WITH a predecessor reaches `JointModel.forward` carrying a real, non-zero prior displacement."""
    trainer = JointTrainer(_velocity_config())
    model, _ = trainer._prepare(warm_start=False)
    sample = PairDataset(_pairs(video_store), 1, (1, 1, 1), 0, with_previous=True).pair(1)
    seen: list[torch.Tensor | None] = []
    forward = JointModel.forward

    def _recording(self: JointModel, *args: object, **kwargs: object) -> object:
        seen.append(cast(torch.Tensor | None, args[4] if len(args) > 4 else kwargs.get("source_velocity")))
        return forward(self, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

    monkeypatch.setattr(JointModel, "forward", _recording)

    trainer._losses(model, sample)

    assert len(seen) == 2  # the no-grad forward on the previous gap, then the scored pair
    assert seen[0] is not None and not seen[0].any()  # the previous gap carries no history of its own
    velocity = seen[1]
    assert velocity is not None
    assert velocity.abs().sum() > 0


def test_train_prior_velocity_off_is_byte_identical(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path
):
    """Off explicitly reproduces the default run exactly — same reads, same head width, same weights, same bytes."""
    default = _trained_bytes(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "default")
    explicit_off = _cpu_config().model_copy(update={"data": DataCfg(downsample=(1, 1, 1), prior_velocity=False)})
    assert _trained_bytes(explicit_off, video_store, in_bounds_tracks, tmp_path / "off") == default


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
