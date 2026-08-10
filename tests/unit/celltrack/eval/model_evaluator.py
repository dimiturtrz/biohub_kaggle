"""Unit tests for the shipped-pipeline model evaluator — the LB-aligned checkpoint selector the trainer uses.

`mount` is checked against stubbed proxy/edge-scorer loaders (the real ones need the pilkwang packs, absent in
CI). `evaluate` is run end to end on the tiny video fixture with a minimal detector and a geometry-only edge
scorer, so a real `CellTracker` pass over one movie is exercised without the GPU, the full model, or the packs —
asserting it returns a float score and finite node metrics in the `EvalResult` the trainer logs and selects on.
"""

import math
from pathlib import Path
from typing import cast

import pytest
import torch

from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.eval.model_evaluator import EvalResult, ModelEvaluator
from celltrack.eval.proxy import TestMovieProxy
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer, PrecomputedEdgeAffinity
from celltrack.models.joint_model import JointModel
from celltrack.models.temporal_unet_detector import TemporalUNetDetector as _Net
from celltrack.tracker import TrackerConfig
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.geometry import Spacing


class _StubEdgeScorer:
    """A geometry-only stand-in for the pilkwang edge head — every gap unscored, so the linker uses pure geometry."""

    def affinities(self, path: Path, detections: TrackGraph, device: str) -> PrecomputedEdgeAffinity:
        return PrecomputedEdgeAffinity({})


def _proxy(video_store: Path, truth: AnnotatedTracks) -> TestMovieProxy:
    """A one-movie proxy over the tiny video fixture, at unit spacing."""
    return TestMovieProxy(paths=(video_store,), truths=(truth,), spacing=Spacing(z=1.0, y=1.0, x=1.0))


def _evaluator(proxy: TestMovieProxy) -> ModelEvaluator:
    """A CPU evaluator over one movie: a geometry-only edge scorer, a permissive threshold, no length pruning."""
    return ModelEvaluator(
        proxy=proxy,
        edge_scorer=cast(BlendedEdgeTransformerScorer, _StubEdgeScorer()),
        recipe=DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False),
        device="cpu",
        config=TrackerConfig(threshold=0.0, min_track_length=1),
    )


def test_mount(monkeypatch: pytest.MonkeyPatch):
    """`mount` keeps the caller's proxy and mounts the two-seed edge affinity at the config's blend, both shared."""
    captured: dict[str, object] = {}

    def _fake_from_packs(
        packs: tuple[Path, ...], weights: tuple[float, ...], device: str, *, bidirectional: bool
    ) -> str:
        captured["packs"], captured["weights"], captured["device"] = packs, weights, device
        return "edge"

    monkeypatch.setattr(BlendedEdgeTransformerScorer, "from_packs", staticmethod(_fake_from_packs))

    evaluator = ModelEvaluator.mount(
        cast(TestMovieProxy, "proxy"),
        (Path("p1"), Path("p2")),
        DetectorRecipe(),
        "cpu",
        TrackerConfig(edge_blend=(0.8, 0.2)),
    )

    assert evaluator.proxy == "proxy"
    assert evaluator.edge_scorer == "edge"
    assert captured["packs"] == (Path("p1"), Path("p2"))
    assert captured["weights"] == (0.8, 0.2)


def test_evaluate(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """`evaluate` runs a real tracker over the movie and returns a float score with finite node metrics."""
    torch.manual_seed(0)
    detector = TemporalUNetDetector(out_channels=2, layers=(2, 4))

    result = _evaluator(_proxy(video_store, in_bounds_tracks)).evaluate(detector)

    assert isinstance(result, EvalResult)
    assert isinstance(result.score, float)
    assert result.selection_score <= result.score  # the clamp only ever removes the under-detection bonus
    assert math.isfinite(result.node_recall)
    assert math.isfinite(result.node_ratio)


def test_evaluate_joint(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """`evaluate_joint` scores BOTH trained heads — the model's own transformer, not the mounted pilkwang one."""
    torch.manual_seed(0)
    net = _Net(out_channels=2, layers=(2, 4))
    feat_dim = 2 + 4 * _POS_EMBED_DIM
    transformer = EdgeTransformerScorer._transformer_cls()(feat_dim=feat_dim, hidden_dim=8, n_heads=1, n_blocks=1)
    model = JointModel(net, transformer, downsample=(1, 1, 1))

    result = _evaluator(_proxy(video_store, in_bounds_tracks)).evaluate_joint(model)

    assert isinstance(result, EvalResult)
    assert isinstance(result.score, float)
    assert result.selection_score <= result.score
    assert math.isfinite(result.node_recall)
    assert math.isfinite(result.node_ratio)
