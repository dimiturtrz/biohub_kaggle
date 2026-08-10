"""Unit tests for the joint-trainer loop this repo owns — a tiny CPU end-to-end over synthetic GT pairs.

The loss parts and the pair dataset are covered elsewhere; here the whole loop is run for two steps on the
shared tiny video fixture with a minimal (non-pilkwang-sized) backbone and the stubbed transformer, so a
real forward/backward/eval/save cycle over both heads is exercised without the GPU or the `external/` dep.
"""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.data.joint_dataset import PairTarget
from celltrack.detectors.tunet import DetectorRecipe
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TestMovieProxy
from celltrack.tracker import TrackerConfig
from celltrack.training.joint_detector import JointTrainConfig, JointTrainer, PairSplit
from core.data.tracks import AnnotatedTracks
from core.data.video import ImageStatistics
from core.geometry import Spacing


def test_config_defaults_to_pilkwangs_recipe():
    """The zero-argument training config carries the ×4 downsample, their learning rate, and an even loss weight."""
    config = JointTrainConfig()
    assert config.downsample == (1, 4, 4)
    assert config.lr == 1e-4
    assert config.det_weight == 1.0


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


def test_train(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """The loop runs end to end — sample, step, pipeline eval, edge AUC, save — and returns the best proxy score."""
    torch.manual_seed(0)
    targets = _pairs(video_store)
    save_to = tmp_path / "joint.pt"

    best = JointTrainer(_cpu_config()).train(
        PairSplit(targets, targets),
        _evaluator(video_store, in_bounds_tracks),
        warm_start=False,
        save_to=save_to,
    )

    assert isinstance(best, float)
    assert save_to.exists()
