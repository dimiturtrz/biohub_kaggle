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

from celltrack.linefit_smoother import LinefitSmoother
from celltrack.linkers import LinkerConfig
from celltrack.short_track_filter import ShortTrackFilter
from celltrack.training.tunet_dataset import FrameTarget
from celltrack.training.tunet_detector import (
    TUNetDetectorTrainer,
    TUNetTrainConfig,
    _FoldEval,
    _Optimization,
)
from celltrack.tunet import DetectorRecipe
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.geometry import Spacing
from core.metrics.matching import DistanceMatcher


def _flat_logits(shape: tuple[int, int, int], value: float) -> torch.Tensor:
    return torch.full(shape, value, requires_grad=True)


def test_detection_loss():
    """A single GT voxel against all-zero logits yields a finite, positive, differentiable loss."""
    logits = _flat_logits((4, 4, 4), 0.0)
    coords = torch.tensor([[1, 1, 1]])
    loss = TUNetDetectorTrainer._detection_loss(logits, coords, neg_weight=1e-2)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_confident_correct_prediction_beats_confident_wrong_one():
    """Firing high on the GT voxel and low elsewhere scores a lower loss than the inverse."""
    coords = torch.tensor([[1, 1, 1]])
    right = torch.full((4, 4, 4), -6.0)
    right[1, 1, 1] = 6.0
    wrong = torch.full((4, 4, 4), 6.0)
    wrong[1, 1, 1] = -6.0
    lower = TUNetDetectorTrainer._detection_loss(right, coords, 1e-2).item()
    higher = TUNetDetectorTrainer._detection_loss(wrong, coords, 1e-2).item()
    assert lower < higher


def test_out_of_bounds_coordinates_are_ignored():
    """GT coordinates outside the downsampled grid are dropped rather than indexing out of range."""
    logits = _flat_logits((4, 4, 4), 0.0)
    coords = torch.tensor([[1, 1, 1], [99, 0, 0], [-1, 0, 0]])
    assert torch.isfinite(TUNetDetectorTrainer._detection_loss(logits, coords, neg_weight=1e-2))


def test_empty_coordinates_give_an_all_negative_finite_loss():
    """A frame with no GT nodes is all negatives — the loss is defined and finite, not a divide-by-zero."""
    logits = _flat_logits((4, 4, 4), 0.0)
    coords = torch.empty((0, 3), dtype=torch.long)
    assert torch.isfinite(TUNetDetectorTrainer._detection_loss(logits, coords, neg_weight=1e-2))


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
