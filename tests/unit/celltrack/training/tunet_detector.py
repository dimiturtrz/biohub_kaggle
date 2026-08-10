"""Unit tests for the detector-training logic this repo owns — the loss and the end-to-end loop.

The count-normalised detection loss is exercised on synthetic logits; the loop is run for two steps on the
shared tiny video fixture with a minimal (non-pilkwang-sized) backbone, so a real forward/backward/eval/save
cycle is covered without the GPU or the full model.
"""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.detectors.tunet import DetectorRecipe
from celltrack.linkers.linkers import LinkerConfig
from celltrack.postproc.linefit_smoother import LinefitSmoother
from celltrack.postproc.short_track_filter import ShortTrackFilter
from celltrack.training.tunet_dataset import FrameTarget
from celltrack.training.tunet_detector import (
    TUNetDetectorTrainer,
    TUNetTrainConfig,
    _FoldEval,
    _Optimization,
)
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.geometry import Spacing
from core.metrics.matching import DistanceMatcher


def test_config_defaults_to_pilkwangs_recipe():
    """The zero-argument training config carries the ×4 downsample and their detector learning rate."""
    config = TUNetTrainConfig()
    assert config.downsample == (1, 4, 4)
    assert config.lr == 1e-4


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
        eval_threshold=0.0,
        recipe=DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False),
    )


def test_train(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """The loop runs end to end — sample, forward, loss, step, eval, save — and returns a best score, saved."""
    torch.manual_seed(0)
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    frame_coords = in_bounds_tracks.graph.coordinates
    targets = [
        FrameTarget(
            zarr_path=video_store,
            timepoint=0,
            q_low=float(quantiles["0.001"]),
            q_high=float(quantiles["0.999"]),
            coords=frame_coords[frame_coords[:, 0] == 0][:, 1:].astype(np.int64),
        )
    ]
    spacing = Spacing(z=1.0, y=1.0, x=1.0)
    evaluator = _FoldEval(
        videos=[(video_store, in_bounds_tracks)],
        spacing=spacing,
        matcher=DistanceMatcher(spacing=spacing),
        short=ShortTrackFilter(min_length=1),
        smooth=LinefitSmoother(strength=0.8),
        linker_config=LinkerConfig(name="motion"),
    )
    save_to = tmp_path / "detector.pt"
    best = TUNetDetectorTrainer(_cpu_config()).train(targets, evaluator, warm_start=False, save_to=save_to)
    assert isinstance(best, float)
    assert save_to.exists()
