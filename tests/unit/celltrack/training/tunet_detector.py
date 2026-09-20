"""Unit tests for the detector-training logic this repo owns — the loss and the end-to-end loop.

The count-normalised detection loss is exercised on synthetic logits; the loop is run for two steps on the
shared tiny video fixture with a minimal (non-pilkwang-sized) backbone, so a real forward/backward/eval/save
cycle is covered without the GPU or the full model.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.data.tunet_dataset import FrameTarget
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TestMovieProxy
from celltrack.models.edge_transformer import PrecomputedEdgeAffinity
from celltrack.operating_point import TrackerConfig
from celltrack.training.run_tracking import RunSetup, TrainingSplit
from celltrack.training.tunet_detector import (
    TUNetDetectorTrainer,
    TUNetTrainConfig,
    _Optimization,
)
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.data.video import ImageStatistics
from core.geometry import Spacing
from tests.unit.celltrack.conftest import RecordingMlflow


class _StubEdgeScorer:
    """A geometry-only stand-in for the pilkwang edge head — every gap unscored, so the linker uses pure geometry.

    Mounting the real `BlendedEdgeTransformerScorer` needs the two pilkwang packs (absent in CI); the linker
    treats an unscored gap (`probabilities -> None`) as distance-only, which is all this end-to-end loop test needs.
    """

    def affinities(self, path: Path, detections: TrackGraph, device: str) -> PrecomputedEdgeAffinity:
        return PrecomputedEdgeAffinity({})


def test_config_defaults_to_pilkwangs_recipe():
    """The zero-argument training config carries the ×4 downsample and their detector learning rate."""
    config = TUNetTrainConfig()
    assert config.downsample == (1, 4, 4)
    assert config.lr == 1e-4


def test_arg_parser():
    """The CLI parser defaults match the config and the re-pool knobs parse into the (z,y,x) tuple / cap."""
    defaults = TUNetTrainConfig.arg_parser().parse_args([])
    assert defaults.downsample == (1, 4, 4)  # bare run reproduces the shipped strided read
    assert defaults.vram_cap is None  # crawl-guard off unless asked

    repool = TUNetTrainConfig.arg_parser().parse_args(["--downsample", "1", "2", "2", "--vram-cap", "0.85"])
    assert repool.downsample == [1, 2, 2]
    assert repool.vram_cap == 0.85


def _one_param_optimization(*, with_schedule: bool) -> tuple[torch.nn.Parameter, _Optimization]:
    """A single-parameter SGD optimiser, optionally under a cosine schedule, for exercising _Optimization."""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) if with_schedule else None
    return param, _Optimization(optimizer, scheduler)


def test_zero_grad():
    """Clearing gradients through the bundle leaves the parameter with no pending gradient."""
    param, optimization = _one_param_optimization(with_schedule=False)
    param.grad = torch.tensor([5.0])
    optimization.zero_grad()
    assert param.grad is None


def test_step():
    """One bundled step moves the parameter along its gradient and advances the learning-rate schedule."""
    param, optimization = _one_param_optimization(with_schedule=True)
    (param * param).sum().backward()  # grad = 2 * param
    lr_before = optimization.optimizer.param_groups[0]["lr"]
    optimization.step()
    assert param.item() != 1.0
    assert optimization.optimizer.param_groups[0]["lr"] < lr_before


def test_state_dict():
    """The bundle's state carries both the optimiser and, when scheduled, the schedule."""
    _param, optimization = _one_param_optimization(with_schedule=True)
    state = optimization.state_dict()
    assert "optimizer" in state
    assert state["scheduler"] is not None


def test_load_state_dict():
    """Restoring a saved state replays the schedule so the resumed learning rate matches the saved one."""
    _param, saved = _one_param_optimization(with_schedule=True)
    for _ in range(3):
        saved.step()
    target_lr = saved.optimizer.param_groups[0]["lr"]
    _fresh_param, fresh = _one_param_optimization(with_schedule=True)
    fresh.load_state_dict(saved.state_dict())
    assert fresh.optimizer.param_groups[0]["lr"] == target_lr


def _cpu_config() -> TUNetTrainConfig:
    """A tiny CPU config: a two-level backbone, no downsample, evaluated at a permissive threshold."""
    return TUNetTrainConfig(
        steps=2,
        out_channels=2,
        layers=(2, 4),
        downsample=(1, 1, 1),
        device="cpu",
        threads=2,
        prefetch=2,
        batch_size=2,
        eval_every=2,
        eval_tracker=replace(TrackerConfig.shipped(), threshold=0.0),
        recipe=DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False),
    )


def _targets(video_store: Path, in_bounds_tracks: AnnotatedTracks) -> list[FrameTarget]:
    """One annotated frame of the tiny fixture video, at its own intensity quantiles."""
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    frame_coords = in_bounds_tracks.graph.coordinates
    return [
        FrameTarget(
            zarr_path=video_store,
            timepoint=0,
            q_low=float(quantiles["0.001"]),
            q_high=float(quantiles["0.999"]),
            coords=frame_coords[frame_coords[:, 0] == 0][:, 1:].astype(np.int64),
        )
    ]


def _evaluator(video_store: Path, in_bounds_tracks: AnnotatedTracks) -> ModelEvaluator:
    """The real pipeline selector over a one-movie proxy — CPU, permissive threshold, geometry-only edges."""
    return ModelEvaluator(
        proxy=TestMovieProxy(paths=(video_store,), truths=(in_bounds_tracks,), spacing=Spacing(z=1.0, y=1.0, x=1.0)),
        edge_scorer=cast(BlendedEdgeTransformerScorer, _StubEdgeScorer()),  # geometry-only; real head needs packs
        recipe=DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False),
        device="cpu",
        config=TrackerConfig(threshold=0.0, min_track_length=1),
    )


def test_train(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """The loop runs end to end — sample, forward, loss, step, eval, save — and returns a best score, saved."""
    torch.manual_seed(0)
    save_to = tmp_path / "detector.pt"
    best = TUNetDetectorTrainer(_cpu_config()).train(
        _targets(video_store, in_bounds_tracks), _evaluator(video_store, in_bounds_tracks), RunSetup(save_to)
    )
    assert isinstance(best, float)
    assert save_to.exists()


def test_train_records_each_window(
    video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path, mlflow_backend: RecordingMlflow
):
    """The loop is actually wired to the tracker: the config lands as params, each window as metrics at its step."""
    torch.manual_seed(0)
    TUNetDetectorTrainer(_cpu_config()).train(
        _targets(video_store, in_bounds_tracks),
        _evaluator(video_store, in_bounds_tracks),
        RunSetup(tmp_path / "detector.pt", split=TrainingSplit(191, 4, 4)),
    )
    assert mlflow_backend.logged_params()["steps"] == 2
    assert {
        "proxy_score",
        "best_score",
        "node_recall",
        "node_ratio",
        "train_loss",
        "it_per_s",
        "frames_per_s",
    } <= mlflow_backend.metric_keys()
    steps = {call[3] for call in mlflow_backend.calls if call[0] == "metric"}
    assert steps == {0, 2}  # the init eval at step 0, then the single two-step window
    assert ("end",) in mlflow_backend.calls


def test_voxel_um():
    """A training voxel is the imaging scale stretched by the run's downsample, per axis."""
    config = TUNetTrainConfig(downsample=(1, 4, 4))

    assert config.voxel_um() == (1.625, 1.625, 1.625)
    assert TUNetTrainConfig(downsample=(1, 2, 2)).voxel_um() == (1.625, 0.8125, 0.8125)


def test_offset_radius():
    """The supervised half-window is one cell radius in each axis' own voxels, never below one."""
    assert TUNetTrainConfig(downsample=(1, 4, 4)).offset_radius() == (2, 2, 2)
    assert TUNetTrainConfig(downsample=(1, 2, 2)).offset_radius() == (2, 5, 5)
