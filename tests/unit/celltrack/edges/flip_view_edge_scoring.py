"""Unit tests for four-view flip TTA on the edge head — the view set, the pooling, and the carried history."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from celltrack.edges.flip_view_edge_scoring import FlipViewEdgeScorer
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeGap, EdgeTransformerScorer, video_gaps
from celltrack.models.prior_velocity import GapHistory, PriorVelocity
from celltrack.models.temporal_unet_detector import DetectorRecipe, FlipView, TemporalUNetDetector
from core.data.tracks import TrackGraph

_RECIPE = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)


def _tiny_scorer(seed: int) -> EdgeTransformerScorer:
    """A minimal scorer with its own random weights, sized for CPU."""
    torch.manual_seed(seed)
    return EdgeTransformerScorer(TemporalUNetDetector(out_channels=2, layers=(2, 4)), _RECIPE).eval()


def _widened_scorer(seed: int) -> EdgeTransformerScorer:
    """A seed whose head was widened for the prior velocity — the case that carries a history between gaps."""
    torch.manual_seed(seed)
    scorer = EdgeTransformerScorer(TemporalUNetDetector(out_channels=2, layers=(2, 4)), _RECIPE)
    scorer.transformer = EdgeTransformerScorer._transformer_cls()(
        feat_dim=2 + 4 * _POS_EMBED_DIM + PriorVelocity.DIM, hidden_dim=8, n_heads=1, n_blocks=1
    )
    return scorer.eval()


def _video(tmp_path: Path, frames: int = 2) -> Path:
    """A synthetic OME-zarr with unit voxel scale and the quantiles the reader expects."""
    group = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    array = group.create_array("0", shape=(frames, 4, 8, 8), dtype="float32")
    array[:] = 0.0
    array[:, 1:3, 4, 5] = 1.0
    group.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}
    group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [{"scale": [1.0, 1.0, 1.0, 1.0]}]}]}]
    return tmp_path / "v.zarr"


def _detections() -> TrackGraph:
    """Two detections at t=0 and two at t=1 — a gap the scorer must return a (2, 2) matrix for."""
    coords = np.array([[0, 1, 4, 5], [0, 3, 2, 6], [1, 1, 4, 4], [1, 3, 5, 6]], dtype=np.int64)
    return TrackGraph(np.arange(4, dtype=np.int64), coords, np.empty((0, 2), dtype=np.int64))


def _gap(tmp_path: Path, detections: TrackGraph) -> EdgeGap:
    """The single scorable gap of the synthetic two-frame video."""
    return next(iter(video_gaps(_video(tmp_path), detections, "cpu")))


def test_gap_logits_pools_the_four_detector_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The head runs once per view of `FlipView.tta_ensemble` — one flip set for the detector and the edges."""
    scorer = _tiny_scorer(0)
    seen: list[FlipView] = []
    original = EdgeTransformerScorer._gap_window

    def _record(self: EdgeTransformerScorer, gap: EdgeGap, *, reverse: bool, view: FlipView) -> object:
        seen.append(view)
        return original(self, gap, reverse=reverse, view=view)

    monkeypatch.setattr(EdgeTransformerScorer, "_gap_window", _record)
    with torch.no_grad():
        FlipViewEdgeScorer(scorer).gap_logits(_gap(tmp_path, _detections()), GapHistory())
    assert tuple(seen) == FlipView.tta_ensemble()


def test_gap_logits(tmp_path: Path):
    """The pooled matrix keeps the single-view shape, stays finite, and is not the single view it replaces."""
    scorer, detections = _tiny_scorer(0), _detections()
    gap = _gap(tmp_path, detections)
    with torch.no_grad():
        pooled, _ = FlipViewEdgeScorer(scorer).gap_logits(gap, GapHistory())
        single, _ = scorer._seed_logits(gap, GapHistory())
    assert pooled.shape == single.shape == (2, 2)
    assert torch.isfinite(pooled).all()
    assert not torch.allclose(pooled, single)


def test_reverse_gap_logits(tmp_path: Path):
    """The reverse direction pools the same four views and comes back in `(t, s)`, the shape the fusion wants."""
    scorer, detections = _tiny_scorer(0), _detections()
    with torch.no_grad():
        reverse = FlipViewEdgeScorer(scorer).reverse_gap_logits(_gap(tmp_path, detections))
    assert reverse.shape == (2, 2)
    assert torch.isfinite(reverse).all()


def _moving_detections() -> TrackGraph:
    """Two cells over three frames, displaced each step — so the second gap's sources have a real history."""
    coords = np.array(
        [[0, 1, 2, 2], [0, 3, 5, 5], [1, 1, 3, 3], [1, 3, 6, 6], [2, 1, 4, 4], [2, 3, 7, 7]], dtype=np.int64
    )
    return TrackGraph(np.arange(6, dtype=np.int64), coords, np.empty((0, 2), dtype=np.int64))


def test_carried_history_is_the_identity_views(tmp_path: Path):
    """The history handed to the next gap comes from the unflipped view — a velocity is an input, not a score."""
    scorer = _widened_scorer(0)
    detections = _moving_detections()
    gap = next(iter(video_gaps(_video(tmp_path, frames=3), detections, "cpu")))
    with torch.no_grad():
        _, carried = FlipViewEdgeScorer(scorer).gap_logits(gap, GapHistory())
        expected = scorer._history_after(scorer._gap_window(gap, reverse=False))
    assert carried.coordinates is not None
    assert expected.coordinates is not None
    assert carried.logits is not None
    assert expected.logits is not None
    assert torch.equal(carried.coordinates, expected.coordinates)
    assert torch.equal(carried.logits, expected.logits)
