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
from celltrack.models.prior_velocity import GapHistory, PriorVelocity
from celltrack.models.temporal_unet_detector import DetectorRecipe, TemporalUNetDetector
from core.data.tracks import TrackGraph

_RECIPE = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
_OUT_CHANNELS = 2


def _tiny_scorer(*, prior_velocity: bool = False) -> EdgeTransformerScorer:
    """A minimal scorer: a small UNet backbone and the real transformer head, sized for CPU.

    `prior_velocity` mounts the head at the WIDENED input width a `--prior-velocity` run saves, which is the
    only thing that tells the scorer to carry history — there is no flag.
    """
    torch.manual_seed(0)
    detector = TemporalUNetDetector(out_channels=_OUT_CHANNELS, layers=(2, 4))
    scorer = EdgeTransformerScorer(detector, _RECIPE)
    if prior_velocity:
        scorer.transformer = EdgeTransformerScorer._transformer_cls()(
            feat_dim=_OUT_CHANNELS + 4 * _POS_EMBED_DIM + PriorVelocity.DIM, hidden_dim=8, n_heads=1, n_blocks=1
        )
    return scorer.eval()


def _video(tmp_path: Path, frames: int = 2) -> Path:
    """A synthetic OME-zarr of `frames` frames with unit voxel scale and the quantiles the reader expects."""
    group = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    array = group.create_array("0", shape=(frames, 4, 8, 8), dtype="float32")
    array[:] = 0.0
    array[:, 1:3, 4, 4] = 1.0
    group.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}
    group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [{"scale": [1.0, 1.0, 1.0, 1.0]}]}]}]
    return tmp_path / "v.zarr"


def _detections() -> TrackGraph:
    """Two detections at t=0 and two at t=1 — a gap the scorer must return a (2, 2) matrix for."""
    coords = np.array([[0, 1, 4, 4], [0, 3, 4, 4], [1, 1, 4, 4], [1, 3, 4, 4]], dtype=np.int64)
    return TrackGraph(np.arange(4, dtype=np.int64), coords, np.empty((0, 2), dtype=np.int64))


def _moving_detections() -> TrackGraph:
    """Two cells over THREE frames, both displaced each step — a second gap whose sources have a real history."""
    coords = np.array(
        [[0, 1, 2, 2], [0, 3, 5, 5], [1, 1, 3, 3], [1, 3, 6, 6], [2, 1, 4, 4], [2, 3, 7, 7]], dtype=np.int64
    )
    return TrackGraph(np.arange(6, dtype=np.int64), coords, np.empty((0, 2), dtype=np.int64))


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


def test_over(tmp_path: Path):
    """Every scorable `t -> t+1` gap of a video, in ascending time — the enumeration every scorer shares."""
    detections = _detections()
    video = _video(tmp_path, frames=3)

    gaps = list(EdgeGap.over(video, detections, "cpu"))

    assert [gap.timepoint for gap in gaps] == sorted(gap.timepoint for gap in gaps)
    assert len(gaps) >= 1


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


def test_uses_prior_velocity():
    """Whether history is carried is read off the mounted projection's width — there is no flag to get wrong."""
    assert not _tiny_scorer().uses_prior_velocity
    assert _tiny_scorer(prior_velocity=True).uses_prior_velocity


def test_affinities_without_prior_velocity_is_unchanged(tmp_path: Path):
    """An unwidened head computes exactly what it did before history existed: the stateless per-gap logits."""
    scorer = _tiny_scorer()
    video, detections = _video(tmp_path, frames=3), _moving_detections()

    matrices = scorer.affinities(video, detections, "cpu")

    with torch.no_grad():
        for gap in EdgeGap.over(video, detections, "cpu"):
            expected = torch.softmax(scorer._gap_logits(gap), dim=0).numpy()
            scored = matrices.probabilities(gap.timepoint)
            assert scored is not None and np.array_equal(scored, expected)


def test_affinities_prior_velocity_carries_history_across_gaps(tmp_path: Path):
    """A widened head scores every gap — and the SECOND gap's logits are moved by the first gap's history.

    The first gap has no predecessor and so takes the zero velocity an empty history yields, which makes it
    bit-identical to the stateless pass; the second gap is not, because the state crossed the boundary.
    """
    scorer = _tiny_scorer(prior_velocity=True)
    video, detections = _video(tmp_path, frames=3), _moving_detections()
    first, second = EdgeGap.over(video, detections, "cpu")

    matrices = scorer.affinities(video, detections, "cpu")

    with torch.no_grad():
        stateless_first = torch.softmax(scorer._gap_logits(first), dim=0).numpy()
        stateless_second = torch.softmax(scorer._gap_logits(second), dim=0).numpy()
    scored_second = matrices.probabilities(second.timepoint)
    assert scored_second is not None
    assert np.isfinite(scored_second).all()
    assert np.allclose(scored_second.sum(axis=0), 1.0, atol=1e-5)
    scored_first = matrices.probabilities(first.timepoint)
    assert scored_first is not None
    assert np.array_equal(scored_first, stateless_first)  # no predecessor == zeros
    assert not np.allclose(scored_second, stateless_second)


def test_reverse_pass_takes_no_history(tmp_path: Path):
    """The reversed direction of a widened head reads zeros — its sources' only prior gap is the one under test."""
    scorer = _tiny_scorer(prior_velocity=True)
    video, detections = _video(tmp_path, frames=3), _moving_detections()
    first, second = EdgeGap.over(video, detections, "cpu")

    with torch.no_grad():
        history = scorer._seed_logits(first, GapHistory())[1]
        reverse = scorer._gap_logits(second, reverse=True)
        window = scorer._gap_window(second, reverse=True)
        zeroed, _ = scorer._velocity_columns(window, GapHistory())

    assert history.logits is not None  # the forward direction DID carry state into the second gap
    assert torch.equal(reverse, scorer._head_logits(window, GapHistory()))
    assert zeroed is not None and not zeroed.any()
    assert reverse.shape == (2, 2)
    assert torch.isfinite(reverse).all()
