"""Unit tests for the multi-seed edge-transformer blend — the pack loader and the logit-space blend.

Mirrors the single-seed tests: tiny UNet backbones and the real transformer head on a synthetic two-frame
video, so no real weights or GPU are needed. Two seeds are enough to exercise the convex logit combination.
"""

import json
from pathlib import Path

import numpy as np
import torch
import zarr

from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer, EdgeBlendOptions
from celltrack.edges.seed_moment_alignment import SeedMomentAlignment
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeGap, EdgeTransformerScorer, video_gaps
from celltrack.models.prior_velocity import GapHistory, PriorVelocity
from celltrack.models.temporal_unet_detector import DetectorRecipe, TemporalUNetDetector
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


def test_fuse():
    """Fusion is the harmonic mean, hand-computed — and a pair both directions call impossible fuses to 0."""
    forward = torch.tensor([[0.8, 0.2], [0.5, 0.0]])
    reverse = torch.tensor([[0.4, 0.6], [0.0, 0.0]])
    fused = BlendedEdgeTransformerScorer.fuse(forward, reverse)
    expected = torch.tensor([[2 * 0.8 * 0.4 / 1.2, 2 * 0.2 * 0.6 / 0.8], [0.0, 0.0]])
    assert torch.allclose(fused, expected)
    assert not torch.isnan(fused).any()
    assert fused[0, 0] < forward[0, 0]  # dominated by the smaller direction, unlike an arithmetic mean


def test_affinities_disabled_is_the_forward_blend(tmp_path: Path):
    """With fusion off the output is exactly today's: the seed-blended logits soft-maxed over the sources."""
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    detections, video = _detections(), _video(tmp_path)
    matrix = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2)).affinities(video, detections, "cpu").probabilities(0)
    positions = detections.positions().astype(np.float32)
    gap = EdgeGap(TemporalUNetDetector._open_source(video), 0, positions[:2], positions[2:], "cpu")
    with torch.no_grad():
        weighted = torch.stack([weight * scorer._gap_logits(gap) for scorer, weight in ((a, 0.8), (b, 0.2))])
    expected = torch.softmax(weighted.sum(dim=0), dim=0).numpy()
    assert matrix is not None
    assert np.array_equal(matrix, expected)


def test_affinities_bidirectional(tmp_path: Path):
    """Enabling fusion changes the matrix, keeps its shape, and introduces no NaNs."""
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    detections, video = _detections(), _video(tmp_path)
    plain = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2)).affinities(video, detections, "cpu").probabilities(0)
    fused = (
        BlendedEdgeTransformerScorer((a, b), (0.8, 0.2), bidirectional=True)
        .affinities(video, detections, "cpu")
        .probabilities(0)
    )
    assert plain is not None
    assert fused is not None
    assert fused.shape == plain.shape
    assert np.isfinite(fused).all()
    assert not np.allclose(fused, plain)


def _widened_scorer(seed: int) -> EdgeTransformerScorer:
    """A seed whose head was widened for the prior velocity — the state a `--prior-velocity` run saves."""
    torch.manual_seed(seed)
    scorer = EdgeTransformerScorer(TemporalUNetDetector(out_channels=2, layers=(2, 4)), _RECIPE)
    scorer.transformer = EdgeTransformerScorer._transformer_cls()(
        feat_dim=2 + 4 * _POS_EMBED_DIM + PriorVelocity.DIM, hidden_dim=8, n_heads=1, n_blocks=1
    )
    return scorer.eval()


def _moving_detections() -> TrackGraph:
    """Two cells over THREE frames, displaced each step — the second gap's sources have a real history."""
    coords = np.array(
        [[0, 1, 2, 2], [0, 3, 5, 5], [1, 1, 3, 3], [1, 3, 6, 6], [2, 1, 4, 4], [2, 3, 7, 7]], dtype=np.int64
    )
    return TrackGraph(np.arange(6, dtype=np.int64), coords, np.empty((0, 2), dtype=np.int64))


def test_affinities_prior_velocity_is_per_seed(tmp_path: Path):
    """A widened blend scores every gap, and a one-member blend still equals that seed scored on its own.

    Each seed carries its OWN history — a velocity has to be the one that seed's weights produced — so the
    single-seed case (what a joint run's selector eval mounts) must reduce exactly to the solo scorer.
    """
    a = _widened_scorer(0)
    detections, video = _moving_detections(), _video(tmp_path, frames=3)

    blended = BlendedEdgeTransformerScorer((a,), (1.0,)).affinities(video, detections, "cpu")
    solo = a.affinities(video, detections, "cpu")

    for timepoint in (0, 1):
        matrix, expected = blended.probabilities(timepoint), solo.probabilities(timepoint)
        assert matrix is not None
        assert expected is not None
        assert np.isfinite(matrix).all()
        assert np.array_equal(matrix, expected)


def test_bidirectional_reverse_carries_no_history(tmp_path: Path):
    """The reverse half of a fused gap is the STATELESS pass: its sources' only prior gap is the one being scored.

    So the second gap's fused matrix must reproduce from the history-carrying forward and a reverse computed
    with no history at all — zeros for a widened head, never the forward direction's history, which belongs to
    the other frame's nodes.
    """
    a = _widened_scorer(0)
    scorer = BlendedEdgeTransformerScorer((a,), (1.0,), bidirectional=True)
    detections, video = _moving_detections(), _video(tmp_path, frames=3)
    first, second = video_gaps(video, detections, "cpu")

    fused = scorer.affinities(video, detections, "cpu").probabilities(1)

    with torch.no_grad():
        _, history = a._seed_logits(first, GapHistory())
        forward = torch.softmax(a._seed_logits(second, history)[0], dim=0)
        reverse = torch.softmax(a._gap_logits(second, reverse=True), dim=0).T
    assert fused is not None
    assert np.array_equal(fused, BlendedEdgeTransformerScorer.fuse(forward, reverse).numpy())


def test_with_bidirectional():
    """Re-pointing the knob keeps the mounted seeds — a sweep re-uses the loaded weights."""
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    scorer = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2))
    switched = scorer.with_bidirectional(bidirectional=True)
    assert switched.bidirectional
    assert not scorer.bidirectional
    assert switched.scorers is scorer.scorers
    assert switched.weights == scorer.weights


def test_alignment_and_view_tta_default_off(tmp_path: Path):
    """Both new mechanisms are opt-in: an unspecified scorer computes the byte-identical matrix it always did."""
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    detections, video = _detections(), _video(tmp_path)
    scorer = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2))
    assert not scorer.options.align_seed_moments
    assert not scorer.options.view_tta

    matrix = scorer.affinities(video, detections, "cpu").probabilities(0)
    positions = detections.positions().astype(np.float32)
    gap = EdgeGap(TemporalUNetDetector._open_source(video), 0, positions[:2], positions[2:], "cpu")
    with torch.no_grad():
        weighted = torch.stack([weight * scorer._gap_logits(gap) for scorer, weight in ((a, 0.8), (b, 0.2))])
    assert matrix is not None
    assert np.array_equal(matrix, torch.softmax(weighted.sum(dim=0), dim=0).numpy())


def test_alignment_changes_the_blend(tmp_path: Path):
    """Turning alignment on rescales seed 1 onto seed 0 before the weighted sum, so the matrix moves."""
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    detections, video = _detections(), _video(tmp_path)
    plain = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2)).affinities(video, detections, "cpu").probabilities(0)
    aligned = (
        BlendedEdgeTransformerScorer((a, b), (0.8, 0.2), options=EdgeBlendOptions(align_seed_moments=True))
        .affinities(video, detections, "cpu")
        .probabilities(0)
    )
    assert plain is not None
    assert aligned is not None
    assert np.isfinite(aligned).all()
    assert not np.allclose(aligned, plain)


def test_alignment_is_a_no_op_on_a_single_seed(tmp_path: Path):
    """With nothing to align against, the reference seed passes through untouched."""
    a = _tiny_scorer(0)
    detections, video = _detections(), _video(tmp_path)
    plain = BlendedEdgeTransformerScorer((a,), (1.0,)).affinities(video, detections, "cpu").probabilities(0)
    aligned = (
        BlendedEdgeTransformerScorer((a,), (1.0,), options=EdgeBlendOptions(align_seed_moments=True))
        .affinities(video, detections, "cpu")
        .probabilities(0)
    )
    assert plain is not None
    assert aligned is not None
    assert np.array_equal(aligned, plain)


def test_view_tta_changes_the_blend(tmp_path: Path):
    """Turning the four-view edge pool on changes the matrix and keeps it a valid over-sources distribution."""
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    detections, video = _detections(), _video(tmp_path)
    plain = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2)).affinities(video, detections, "cpu").probabilities(0)
    pooled = (
        BlendedEdgeTransformerScorer((a, b), (0.8, 0.2), options=EdgeBlendOptions(view_tta=True))
        .affinities(video, detections, "cpu")
        .probabilities(0)
    )
    assert plain is not None
    assert pooled is not None
    assert np.allclose(pooled.sum(axis=0), 1.0, atol=1e-5)
    assert not np.allclose(pooled, plain)


def test_view_tta_reaches_the_reverse_direction(tmp_path: Path):
    """The pool applies to both directions, so a bidirectional run fuses two pooled normalisations."""
    a = _tiny_scorer(0)
    detections, video = _detections(), _video(tmp_path)
    fused = (
        BlendedEdgeTransformerScorer((a,), (1.0,), bidirectional=True, options=EdgeBlendOptions(view_tta=True))
        .affinities(video, detections, "cpu")
        .probabilities(0)
    )
    forward_only = (
        BlendedEdgeTransformerScorer((a,), (1.0,), options=EdgeBlendOptions(view_tta=True))
        .affinities(video, detections, "cpu")
        .probabilities(0)
    )
    assert fused is not None
    assert forward_only is not None
    assert np.isfinite(fused).all()
    assert not np.allclose(fused, forward_only)


def test_with_bidirectional_carries_the_opt_ins():
    """Re-pointing the fusion knob must not silently drop the other two — a sweep would then compare two things."""
    a = _tiny_scorer(0)
    scorer = BlendedEdgeTransformerScorer(
        (a,), (1.0,), options=EdgeBlendOptions(align_seed_moments=True, view_tta=True)
    )
    switched = scorer.with_bidirectional(bidirectional=True)
    assert switched.options.align_seed_moments
    assert switched.options.view_tta


def test_with_options():
    """Re-pointing the transforms keeps the mounted seeds AND the fusion setting — the config path's other half."""
    a = _tiny_scorer(0)
    scorer = BlendedEdgeTransformerScorer((a,), (1.0,), bidirectional=True)
    switched = scorer.with_options(EdgeBlendOptions(align_seed_moments=True, view_tta=True))
    assert switched.scorers == scorer.scorers
    assert switched.bidirectional
    assert (switched.options.align_seed_moments, switched.options.view_tta) == (True, True)
    assert scorer.options == EdgeBlendOptions()  # the original is untouched — a sweep re-points, not mutates


def test_seed_scores(tmp_path: Path):
    """Every gap comes back as raw per-seed logits plus each seed's own probabilities — the blend, un-collapsed.

    The logits must be the seeds' RAW ones (pre-weighting), and the probabilities each seed's own softmax, so
    a caller can recover the shipped blended matrix from what is reported and see that nothing was normalised
    away on the trip out.
    """
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    detections, video = _detections(), _video(tmp_path)
    scorer = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2))

    reported = scorer.seed_scores(video, detections, "cpu")

    positions = detections.positions().astype(np.float32)
    gap = EdgeGap(TemporalUNetDetector._open_source(video), 0, positions[:2], positions[2:], "cpu")
    with torch.no_grad():
        expected = [seed._gap_logits(gap) for seed in (a, b)]
    assert len(reported) == 1
    assert reported[0].timepoint == 0
    assert reported[0].logits.shape == (2, 2, 2)
    assert np.allclose(reported[0].logits, torch.stack(expected).numpy())
    assert np.allclose(reported[0].probabilities, torch.softmax(torch.stack(expected), dim=1).numpy())

    blended = scorer.affinities(video, detections, "cpu").probabilities(0)
    assert blended is not None
    weighted = 0.8 * reported[0].logits[0] + 0.2 * reported[0].logits[1]
    assert np.allclose(blended, torch.softmax(torch.from_numpy(weighted), dim=0).numpy(), atol=1e-6)


def test_seed_scores_bidirectional_fuses_each_seed_alone(tmp_path: Path):
    """Under fusion each seed's probabilities are ITS forward fused with ITS reverse — not the blend's.

    A per-seed number taken forward-only while the blend was bidirectional would differ from the matrix the
    linker consumed for a reason that has nothing to do with the seeds disagreeing.
    """
    a = _tiny_scorer(0)
    detections, video = _detections(), _video(tmp_path)

    reported = BlendedEdgeTransformerScorer((a,), (1.0,), bidirectional=True).seed_scores(video, detections, "cpu")

    positions = detections.positions().astype(np.float32)
    gap = EdgeGap(TemporalUNetDetector._open_source(video), 0, positions[:2], positions[2:], "cpu")
    with torch.no_grad():
        forward = torch.softmax(a._gap_logits(gap), dim=0)
        reverse = torch.softmax(a._gap_logits(gap, reverse=True), dim=0).T
    assert np.allclose(reported[0].probabilities[0], BlendedEdgeTransformerScorer.fuse(forward, reverse).numpy())


def test_seed_logit_moments(tmp_path: Path):
    """The reporter returns one entry per gap, per seed, holding those seeds' actual raw logit mean and std."""
    a, b = _tiny_scorer(0), _tiny_scorer(1)
    detections, video = _detections(), _video(tmp_path)
    reported = BlendedEdgeTransformerScorer((a, b), (0.8, 0.2)).seed_logit_moments(video, detections, "cpu")

    positions = detections.positions().astype(np.float32)
    gap = EdgeGap(TemporalUNetDetector._open_source(video), 0, positions[:2], positions[2:], "cpu")
    with torch.no_grad():
        expected = tuple(SeedMomentAlignment.moments(seed._gap_logits(gap)) for seed in (a, b))

    assert len(reported) == 1
    assert reported[0].timepoint == 0
    assert reported[0].moments == expected
