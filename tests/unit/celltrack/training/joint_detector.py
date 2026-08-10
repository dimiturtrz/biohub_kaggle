"""Unit tests for the joint-trainer loop this repo owns — a tiny CPU end-to-end over synthetic GT pairs.

The loss parts and the pair dataset are covered elsewhere; here the whole loop is run for two steps on the
shared tiny video fixture with a minimal (non-pilkwang-sized) backbone and the stubbed transformer, so a
real forward/backward/eval/save cycle over both heads is exercised without the GPU or the `external/` dep.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.data.difficulty_sampler import DifficultySampler
from celltrack.data.joint_dataset import PairDataset, PairTarget
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TestMovieProxy
from celltrack.tracker import TrackerConfig
from celltrack.training.joint_detector import JointTrainConfig, JointTrainer, PairSplit, _PairOutcome
from celltrack.training.run_tracking import RunSetup, TrainingSplit
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.geometry import Spacing
from tests.unit.celltrack.conftest import RecordingMlflow


def test_config_defaults_to_pilkwangs_recipe():
    """The zero-argument training config carries the ×4 downsample, their learning rate, and an even loss weight."""
    config = JointTrainConfig()
    assert config.downsample == (1, 4, 4)
    assert config.lr == 1e-4
    assert config.det_weight == 1.0
    assert config.contrastive_weight == 0.0  # the contrastive term is opt-in: an unasked run is unchanged


_CORPUS = 18024  # the real GT-pair count: one pair per step, so this is one epoch


def test_resolved():
    """No epoch field set means today's run: the step-scale numbers a caller passed reach the loop unchanged."""
    config = JointTrainConfig(steps=3000, eval_every=500, patience=5)
    assert config.resolved(_CORPUS) == config


def test_resolved_turns_epochs_into_steps():
    """`epochs` is passes over the corpus, so the budget is that many times the derived steps-per-epoch."""
    assert JointTrainConfig(epochs=1.5).resolved(_CORPUS).steps == 27036


def test_resolved_turns_evals_per_epoch_into_a_window_length():
    """Four evals per epoch is a window of a quarter of the corpus, whatever the split's size happens to be."""
    assert JointTrainConfig(evals_per_epoch=4).resolved(_CORPUS).eval_every == 4506


def test_resolved_turns_patience_epochs_into_windows():
    """Patience counts EVAL WINDOWS, so an epoch of patience is however many windows that epoch holds."""
    resolved = JointTrainConfig(patience_epochs=2.0, evals_per_epoch=3).resolved(_CORPUS)
    assert resolved.patience == 6

    from_raw_windows = JointTrainConfig(patience_epochs=1.0, eval_every=9012).resolved(_CORPUS)
    assert from_raw_windows.patience == 2  # no evals_per_epoch: the window count follows the raw eval_every


def test_resolved_epoch_fields_win_over_their_raw_twins():
    """Both forms given is not an error and neither is dropped in silence — the epoch form wins, and it is logged."""
    config = JointTrainConfig(
        steps=3000, eval_every=500, patience=5, epochs=2.0, evals_per_epoch=2, patience_epochs=3.0
    )

    resolved = config.resolved(_CORPUS)

    assert (resolved.steps, resolved.eval_every, resolved.patience) == (36048, 9012, 6)
    assert config._overrides() == [
        "epochs=2.0 overrides steps=3000",
        "evals_per_epoch=2 overrides eval_every=500",
        "patience_epochs=3.0 overrides patience=5",
    ]


def test_resolved_keeps_a_window_when_the_corpus_is_smaller_than_the_eval_count():
    """A corpus of three pairs asked for ten evals would floor to a zero-step window; it floors to one instead."""
    resolved = JointTrainConfig(evals_per_epoch=10, patience_epochs=0.01).resolved(3)
    assert resolved.eval_every == 1
    assert resolved.patience == 1


def test_schedule_spans_the_resolved_step_count():
    """The cosine horizon is built from the RESOLVED budget — resolving epochs late must not leave it stale."""
    config = JointTrainConfig(steps=10, epochs=2.0, cosine_lr=True).resolved(_CORPUS)
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
        steps=2,
        out_channels=2,
        layers=(2, 4),
        downsample=(1, 1, 1),
        device="cpu",
        eval_every=2,
        patience=2,
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


def test_config_defaults_to_uniform_sampling():
    """Difficulty sampling is opt-in — the zero-argument config still draws pairs uniformly with replacement."""
    assert JointTrainConfig().difficulty_sampling is False


def test_sampler(video_store: Path):
    """A sampler exists only when asked for, and is sized to the pair corpus rather than to the step budget."""
    targets = _pairs(video_store)
    assert JointTrainer(_cpu_config())._sampler(targets) is None

    sampler = JointTrainer(replace(_cpu_config(), difficulty_sampling=True))._sampler(targets)

    assert sampler is not None
    sampler.observe(0, 0.5)
    assert sampler.coverage() == 0.5  # one of the corpus's two pairs, not one of the run's two steps


def test_pairs_leaves_the_uniform_stream_untouched(video_store: Path):
    """With no sampler the window walks exactly `PairDataset.stream()`, index-free — the default path is unchanged."""
    dataset = PairDataset(_pairs(video_store), steps=3, downsample=(1, 1, 1), seed=0)
    walked = list(JointTrainer(_cpu_config())._pairs(dataset, None))

    assert [index for index, _ in walked] == [None, None, None]
    for (_, pair), expected in zip(walked, dataset.stream(), strict=True):
        assert all(torch.equal(actual, want) for actual, want in zip(pair, expected, strict=True))


def test_pairs_draws_through_the_sampler(video_store: Path):
    """With a sampler the window walks the drawn indices, and each pair is the corpus entry that index names."""
    dataset = PairDataset(_pairs(video_store), steps=4, downsample=(1, 1, 1), seed=0)
    sampler = DifficultySampler(count=2, seed=0)
    sampler.observe(0, 0.0)  # pair 0 solved, pair 1 still optimistic -> every draw is pair 1

    walked = list(JointTrainer(_cpu_config())._pairs(dataset, sampler))

    assert [index for index, _ in walked] == [1, 1, 1, 1]
    assert torch.equal(walked[0][1][4], dataset.pair(1)[4])


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
    """The joint loop is wired to the tracker, and keeps each loss term and the edge AUC as their own series."""
    torch.manual_seed(0)
    targets = _pairs(video_store)
    JointTrainer(_cpu_config()).train(
        PairSplit(targets, targets),
        _evaluator(video_store, in_bounds_tracks),
        RunSetup(tmp_path / "joint.pt", split=TrainingSplit(191, 4, 4)),
    )
    assert mlflow_backend.logged_params()["det_weight"] == 1.0
    assert {
        "proxy_score",
        "best_score",
        "node_recall",
        "node_ratio",
        "edge_auc",
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
    config = replace(_cpu_config(), steps=2, eval_every=2, patience=0, epochs=2.0, evals_per_epoch=2)

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


def test_train_off_is_byte_identical(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """Difficulty sampling off reproduces the uniform run exactly — same draws, same weights, same bytes."""
    default = _trained_bytes(_cpu_config(), video_store, in_bounds_tracks, tmp_path / "default")
    explicit_off = replace(_cpu_config(), difficulty_sampling=False)
    assert _trained_bytes(explicit_off, video_store, in_bounds_tracks, tmp_path / "off") == default


def test_train_records_sampler_health(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path, mlflow_backend: RecordingMlflow
):
    """With sampling on, each window row carries the coverage and the collapse guard beside the loss terms."""
    torch.manual_seed(0)
    targets = _pairs(video_store)
    JointTrainer(replace(_cpu_config(), difficulty_sampling=True)).train(
        PairSplit(targets, targets), _evaluator(video_store, in_bounds_tracks), RunSetup(tmp_path / "joint.pt")
    )
    assert {"coverage", "sampling_entropy"} <= mlflow_backend.metric_keys()
