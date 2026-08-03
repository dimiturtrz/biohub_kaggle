"""Unit tests for the multi-seed edge-transformer blend — the pack loader and the logit-space blend.

Mirrors the single-seed tests: tiny UNet backbones and the real transformer head on a synthetic two-frame
video, so no real weights or GPU are needed. Two seeds are enough to exercise the convex logit combination.
"""

import json
from pathlib import Path

import numpy as np
import torch
import zarr

from celltrack.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.edge_scoring import EdgeTransformerScorer
from celltrack.tunet import DetectorRecipe, TemporalUNetDetector
from core.data.tracks import TrackGraph

_RECIPE = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)


def _tiny_scorer(seed: int) -> EdgeTransformerScorer:
    """A minimal scorer with its own random weights, sized for CPU."""
    torch.manual_seed(seed)
    return EdgeTransformerScorer(TemporalUNetDetector(out_channels=2, layers=(2, 4)), _RECIPE).eval()


def _save_pack(scorer: EdgeTransformerScorer, directory: Path) -> Path:
    """Serialise a scorer as a whole pack (detector + transformer state + config) the loader can mount."""
    directory.mkdir(parents=True, exist_ok=True)
    state = dict(scorer.detector.state_dict())
    state.update({f"transformer.{k}": v for k, v in scorer.transformer.state_dict().items()})
    torch.save(state, directory / "edge_predictor_best.pth")
    (directory / "config.json").write_text(
        json.dumps({"unet_out_channels": 2, "unet_layers": [2, 4], "downsample": [1, 1, 1], "pool_kernel_um": 1.0})
    )
    return directory


def _video(tmp_path: Path) -> Path:
    """A synthetic two-frame OME-zarr with unit voxel scale and the quantiles the reader expects."""
    group = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    array = group.create_array("0", shape=(2, 4, 8, 8), dtype="float32")
    array[:] = 0.0
    array[:, 1:3, 4, 4] = 1.0
    group.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}
    group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [{"scale": [1.0, 1.0, 1.0, 1.0]}]}]}]
    return tmp_path / "v.zarr"


def _detections() -> TrackGraph:
    """Two detections at t=0 and two at t=1 — a gap the scorer must return a (2, 2) matrix for."""
    coords = np.array([[0, 1, 4, 4], [0, 3, 4, 4], [1, 1, 4, 4], [1, 3, 4, 4]], dtype=np.int64)
    return TrackGraph(np.arange(4, dtype=np.int64), coords, np.empty((0, 2), dtype=np.int64))


def test_from_packs(tmp_path: Path):
    """The loader mounts one scorer per pack directory and pairs each with its blend weight."""
    packs = (_save_pack(_tiny_scorer(0), tmp_path / "a"), _save_pack(_tiny_scorer(1), tmp_path / "b"))
    blended = BlendedEdgeTransformerScorer.from_packs(packs, (0.8, 0.2))
    assert len(blended.scorers) == 2
    assert blended.weights == (0.8, 0.2)


def test_affinities(tmp_path: Path):
    """The blend yields a (2, 2) matrix soft-maxed over sources, and equals a single seed when its weight is 1."""
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    blended = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2)).affinities(_video(tmp_path), _detections(), "cpu")
    matrix = blended.probabilities(0)
    assert matrix is not None
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix.sum(axis=0), 1.0, atol=1e-5)  # softmax over the source dimension

    only_a = BlendedEdgeTransformerScorer((a, b), (1.0, 0.0)).affinities(_video(tmp_path), _detections(), "cpu")
    solo = a.affinities(_video(tmp_path), _detections(), "cpu")
    blended_matrix, solo_matrix = only_a.probabilities(0), solo.probabilities(0)
    assert blended_matrix is not None
    assert solo_matrix is not None
    assert np.allclose(blended_matrix, solo_matrix, atol=1e-5)
