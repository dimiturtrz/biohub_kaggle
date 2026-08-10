"""Unit tests for the edge-transformer mount's own logic — the affinity lookup, the pack loader, the scoring.

The `SimpleNodeTransformer` and `TemporalUNet3D` are third-party (the pinned `external/` checkout); these
cover the code this repo owns: the per-gap probability lookup, the whole-pack (de)serialisation, and the
detect→feature→transformer scoring path, all on a tiny synthetic video so no real weights or GPU are needed.
"""

import json
from pathlib import Path

import numpy as np
import torch
import zarr

from celltrack.models.edge_transformer import (
    _POS_EMBED_DIM,
    EdgeGap,
    EdgeTransformerScorer,
    NodeWindowSlot,
    PrecomputedEdgeAffinity,
)
from celltrack.models.prior_velocity import PriorVelocity
from celltrack.models.temporal_unet_detector import DetectorRecipe, TemporalUNetDetector
from core.data.tracks import TrackGraph

_RECIPE = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)


def _tiny_scorer() -> EdgeTransformerScorer:
    """A minimal scorer: a small UNet backbone and the real transformer head, sized for CPU."""
    torch.manual_seed(0)
    detector = TemporalUNetDetector(out_channels=2, layers=(2, 4))
    return EdgeTransformerScorer(detector, _RECIPE).eval()


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


def _node_feature_inputs() -> tuple[torch.Tensor, torch.Tensor, NodeWindowSlot]:
    """One node at voxel (1, 2, 2) of a 4³ grid whose channels there are distinctive."""
    feature_map = torch.zeros(3, 4, 4, 4)
    feature_map[:, 1, 2, 2] = torch.tensor([5.0, 6.0, 7.0])
    return feature_map, torch.tensor([[1.0, 2.0, 2.0]]), NodeWindowSlot(torch.tensor([4.0, 4.0, 4.0]), 0.0, "cpu")


def test_node_features():
    """A node's feature vector is its UNet channels at the voxel, concatenated with the position embedding."""
    channels = 3
    feature_map, grid_positions, slot = _node_feature_inputs()
    features = EdgeTransformerScorer.node_features(feature_map, grid_positions, slot)
    assert features.shape == (1, channels + 4 * _POS_EMBED_DIM)
    assert torch.allclose(features[0, :channels], torch.tensor([5.0, 6.0, 7.0]))
    indexed = feature_map[:, 1, 2, 2].unsqueeze(0)
    position = EdgeTransformerScorer._position_embedding(grid_positions, slot)
    assert torch.equal(features, torch.cat([indexed, position], dim=-1))  # no velocity == exactly the old tensor


def test_node_features_prior_velocity():
    """A supplied prior step appends three raw columns at the pack's own displacement scale, leaving the rest alone."""
    feature_map, grid_positions, slot = _node_feature_inputs()
    velocity = torch.tensor([[10.0, -20.0, 30.0]])
    without = EdgeTransformerScorer.node_features(feature_map, grid_positions, slot)
    with_velocity = EdgeTransformerScorer.node_features(feature_map, grid_positions, slot, velocity)
    assert with_velocity.shape == (1, without.shape[1] + PriorVelocity.DIM)
    assert torch.equal(with_velocity[:, : without.shape[1]], without)
    assert torch.equal(with_velocity[:, without.shape[1] :], velocity / PriorVelocity.SCALE)


def test_probabilities():
    """A gap with a stored matrix returns it by its source-timepoint key; an unscored gap returns None."""
    matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    affinity = PrecomputedEdgeAffinity({3: matrix})
    assert affinity.probabilities(3) is matrix
    assert affinity.probabilities(7) is None


def test_of():
    """Mounting in-memory heads keeps both objects — the given transformer, not the freshly built one."""
    torch.manual_seed(0)
    detector = TemporalUNetDetector(out_channels=2, layers=(2, 4))
    transformer = EdgeTransformerScorer._transformer_cls()(
        feat_dim=2 + 4 * _POS_EMBED_DIM, hidden_dim=8, n_heads=1, n_blocks=1
    )
    scorer = EdgeTransformerScorer.of(detector, transformer, _RECIPE)
    assert scorer.detector is detector
    assert scorer.transformer is transformer
    assert scorer.recipe is _RECIPE


def test_from_pack(tmp_path: Path):
    """The whole-pack loader takes `unet.*`/`detect_head.*` into the detector and `transformer.*` into the head."""
    scorer = _tiny_scorer()
    state = dict(scorer.detector.state_dict())
    state.update({f"transformer.{k}": v for k, v in scorer.transformer.state_dict().items()})
    torch.save(state, tmp_path / "edge_predictor_best.pth")
    (tmp_path / "config.json").write_text(
        json.dumps({"unet_out_channels": 2, "unet_layers": [2, 4], "downsample": [1, 1, 1], "pool_kernel_um": 1.0})
    )
    loaded = EdgeTransformerScorer.from_pack(tmp_path)
    assert loaded.recipe.downsample == (1, 1, 1)
    assert torch.equal(scorer.transformer.state_dict()["proj.weight"], loaded.transformer.state_dict()["proj.weight"])


def test_gap_logits_reverse(tmp_path: Path):
    """The reversed pass scores the swapped pair: a `(t, s)` matrix that is not the forward one transposed."""
    scorer = _tiny_scorer()
    source = TemporalUNetDetector._open_source(_video(tmp_path))
    src_positions = np.array([[1, 4, 4], [3, 4, 4]], dtype=np.float32)
    tgt_positions = np.array([[1, 4, 4], [2, 4, 4], [3, 4, 4]], dtype=np.float32)
    gap = EdgeGap(source, 0, src_positions, tgt_positions, "cpu")
    with torch.no_grad():
        forward = scorer._gap_logits(gap)
        reverse = scorer._gap_logits(gap, reverse=True)
    assert forward.shape == (2, 3)
    assert reverse.shape == (3, 2)
    assert not torch.allclose(reverse.T, forward)  # a second question of the head, not a re-read of the first


def test_affinities(tmp_path: Path):
    """Scoring a two-node→two-node gap yields a (2, 2) matrix soft-maxed over sources — columns sum to one."""
    affinity = _tiny_scorer().affinities(_video(tmp_path), _detections(), "cpu")
    matrix = affinity.probabilities(0)
    assert matrix is not None
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix.sum(axis=0), 1.0, atol=1e-5)  # softmax over the source dimension
    assert affinity.probabilities(1) is None  # only the t=0 gap has a following frame to score
