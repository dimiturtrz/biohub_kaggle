"""Unit tests for the training-time prior-velocity feed — where the head's history input comes from.

The maths of the feature is covered beside `PriorVelocity`; here the questions are the wiring ones: off is
off (no tensor at all, not a zero one), a pair without a predecessor yields exact zeros, and a pair with one
yields the model's own previous-gap displacement.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.data.frame_source import ZarrFrames
from celltrack.data.joint_dataset import PairDataset, PairSample, PairTarget
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeGap, EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.models.prior_velocity import GapHistory, PriorVelocity
from celltrack.models.temporal_unet_detector import DetectorRecipe, TemporalUNetDetector
from celltrack.training.prior_velocity_source import PriorVelocitySource
from core.data.tracks import TrackGraph
from core.data.video import ImageStatistics

_OUT_CHANNELS = 2
_DOWNSAMPLE = (1, 1, 1)
_RECIPE = DetectorRecipe(downsample=_DOWNSAMPLE, pool_kernel_um=1.0, tta=False)
_WIDENED = _OUT_CHANNELS + 4 * _POS_EMBED_DIM + PriorVelocity.DIM


def _model(feat_dim: int = _OUT_CHANNELS + 4 * _POS_EMBED_DIM) -> JointModel:
    """A tiny joint model over the stubbed backbone and head, at the node-feature width its caller expects."""
    torch.manual_seed(0)
    detector = TemporalUNetDetector(_OUT_CHANNELS, (2, 4))
    transformer = EdgeTransformerScorer._transformer_cls()(feat_dim=feat_dim, hidden_dim=4, n_heads=1, n_blocks=1)
    return JointModel(detector, transformer, _DOWNSAMPLE)


def _sample(video_store: Path, *, previous: bool) -> PairSample:
    """One pair over the tiny video fixture — at t=1 with t=0 as history, or at t=0 with none."""
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    target = PairTarget(
        frames=ZarrFrames(video_store, float(quantiles["0.001"]), float(quantiles["0.999"])),
        timepoint=1 if previous else 0,
        source_centres=np.array([[0, 0, 0], [1, 3, 3]], dtype=np.int64),
        target_centres=np.array([[1, 3, 3]], dtype=np.int64),
        edge_matrix=np.array([[0.0], [1.0]], dtype=np.float32),
        previous_centres=np.array([[1, 2, 2]], dtype=np.int64) if previous else None,
    )
    return PairDataset([target], 1, _DOWNSAMPLE, seed=0, with_previous=True).pair(0)


def _sample_with_gt_links(video_store: Path) -> PairSample:
    """The `previous=True` pair, plus a GT `t-1 -> t` matrix: previous node parents source 0, not source 1.

    A single previous node p=1 over sources s=2. The `[[1, 0]]` matrix makes source 0's incoming mass 1 (its GT
    displacement is exact) and source 1's mass 0 (unparented → zero velocity), the two classes the warmup blend
    must keep apart from the model's own degenerate all-ones softmax, which would hand source 1 a full-mass guess.
    """
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    target = PairTarget(
        frames=ZarrFrames(video_store, float(quantiles["0.001"]), float(quantiles["0.999"])),
        timepoint=1,
        source_centres=np.array([[0, 0, 0], [1, 3, 3]], dtype=np.int64),
        target_centres=np.array([[1, 3, 3]], dtype=np.int64),
        edge_matrix=np.array([[0.0], [1.0]], dtype=np.float32),
        previous_centres=np.array([[1, 2, 2]], dtype=np.int64),
        previous_edge_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    return PairDataset([target], 1, _DOWNSAMPLE, seed=0, with_previous=True).pair(0)


def test_widen_is_a_no_op_while_the_feature_is_off():
    """Off, the head keeps the exact projection object it was built with — the run of yesterday, untouched."""
    model = _model()
    projection = model.transformer.proj

    PriorVelocitySource(enabled=False).widen(model)

    assert model.transformer.proj is projection


def test_widen():
    """On, the projection grows by the velocity's three inputs, and those columns start at exactly zero.

    Zero is what makes a warm start survive the widening: the layer's output is bitwise identical to the
    pretrained one at step 0 for any velocity, so the feature is learned from zero gain upward.
    """
    model = _model()
    original = cast(torch.nn.Linear, model.transformer.proj)
    weights = original.weight.detach().clone()

    PriorVelocitySource(enabled=True).widen(model)

    widened = cast(torch.nn.Linear, model.transformer.proj)
    assert widened.in_features == original.in_features + PriorVelocity.DIM
    assert torch.equal(widened.weight[:, : original.in_features], weights)
    assert not widened.weight[:, -PriorVelocity.DIM :].any()


def test_of_is_none_while_the_feature_is_off(video_store: Path):
    """Off, the head is handed no velocity at all — not a zero one — so its input is what it always was."""
    source = PriorVelocitySource(enabled=False)
    assert source.of(_model(), _sample(video_store, previous=True), torch.autocast("cpu", enabled=False)) is None


def test_of_is_zero_without_a_predecessor(video_store: Path):
    """A video's first pair has no history, and takes the same exact zeros an unmatched node degrades to."""
    source = PriorVelocitySource(enabled=True)

    velocity = source.of(_model(), _sample(video_store, previous=False), torch.autocast("cpu", enabled=False))

    assert velocity is not None
    assert velocity.shape == (2, 3)
    assert not velocity.any()


def test_of(video_store: Path):
    """With one previous node the softmax over sources is degenerate, so each source's velocity is hand-computable.

    A single source at `t-1` makes `softmax(logits, dim=0)` all ones whatever the head says, so every source's
    expected incoming displacement is exactly `c_source - c_previous` — the value below, independent of weights.
    """
    model = _model(feat_dim=_WIDENED)
    source = PriorVelocitySource(enabled=True)

    velocity = source.of(model, _sample(video_store, previous=True), torch.autocast("cpu", enabled=False))

    assert velocity is not None
    assert velocity.tolist() == [[-1.0, -2.0, -2.0], [0.0, 1.0, 1.0]]


def test_of_batch(video_store: Path):
    """`of_batch` returns, per pair, exactly what the single-pair `of` returns — the speed-up changes no number.

    Two pairs WITH a predecessor put `B = 2` through the batched previous-gap forward; the third, WITHOUT one,
    exercises the no-history skip that must land its zeros at the right index rather than shifting the batch. A
    padded forward that leaked a neighbour's nodes, or a scatter that misaligned, would break an equality here.
    """
    model = _model(feat_dim=_WIDENED)
    source = PriorVelocitySource(enabled=True)
    samples = [
        _sample(video_store, previous=True),
        _sample(video_store, previous=True),
        _sample(video_store, previous=False),
    ]
    precision = torch.autocast("cpu", enabled=False)

    batched = source.of_batch(model, samples, precision)
    per_pair = [source.of(model, sample, precision) for sample in samples]

    assert len(batched) == len(samples)
    for got, want in zip(batched, per_pair, strict=True):
        assert (got is None) == (want is None)
        if want is not None:
            assert got is not None
            assert torch.equal(got, want)


def test_gt_warmup_off_is_the_model_velocity(video_store: Path):
    """`gt_warmup_steps=0` (default), and any step past the window, both hand back the model's own velocity.

    Byte-identical off: the GT matrix rides on the sample but is never read, so source 1 keeps the model's
    degenerate `[0, 1, 1]` rather than the zero its unparented column would force.
    """
    model = _model(feat_dim=_WIDENED)
    sample = _sample_with_gt_links(video_store)
    precision = torch.autocast("cpu", enabled=False)

    off = PriorVelocitySource(enabled=True, gt_warmup_steps=0).of_batch(model, [sample], precision, step=0)[0]
    past = PriorVelocitySource(enabled=True, gt_warmup_steps=100).of_batch(model, [sample], precision, step=100)[0]

    assert off is not None and past is not None
    assert off.tolist() == [[-1.0, -2.0, -2.0], [0.0, 1.0, 1.0]]
    assert torch.equal(off, past)


def test_gt_warmup_full_is_the_ground_truth_velocity(video_store: Path):
    """At step 0 of the warmup, alpha=1: the velocity is the GT-link displacement, softmax-free.

    Source 0's parented column gives its exact `c0 - c_prev = [-1, -2, -2]`; source 1's empty column gives the
    zero its incoming mass earns — NOT the model's full-mass `[0, 1, 1]`, the confident-wrong direction the
    warmup exists to keep out of the head while the model's own affinity is still garbage.
    """
    model = _model(feat_dim=_WIDENED)
    source = PriorVelocitySource(enabled=True, gt_warmup_steps=100)

    samples = [_sample_with_gt_links(video_store)]
    velocity = source.of_batch(model, samples, torch.autocast("cpu", enabled=False), step=0)[0]

    assert velocity is not None
    assert velocity.tolist() == [[-1.0, -2.0, -2.0], [0.0, 0.0, 0.0]]


def test_gt_warmup_anneals_linearly_between_gt_and_model(video_store: Path):
    """Halfway through the warmup, alpha=0.5: the velocity is the midpoint of the GT and model vectors.

    Source 0 agrees in both regimes; source 1 is GT `[0,0,0]` vs model `[0,1,1]`, so the blend lands exactly
    halfway at `[0, 0.5, 0.5]` — the linear hand-back the head's input distribution rides to the self-derived one.
    """
    model = _model(feat_dim=_WIDENED)
    source = PriorVelocitySource(enabled=True, gt_warmup_steps=100)

    samples = [_sample_with_gt_links(video_store)]
    velocity = source.of_batch(model, samples, torch.autocast("cpu", enabled=False), step=50)[0]

    assert velocity is not None
    assert velocity.tolist() == [[-1.0, -2.0, -2.0], [0.0, 0.5, 0.5]]


def _same_nodes() -> TrackGraph:
    """The `_sample` pair's nodes as a detection graph: `previous_centres` at t=0, `source_centres` at t=1.

    Same coordinates, same row order, so the gap the scorer enumerates for `t=0 -> t=1` is exactly the gap the
    trainer's `previous_frame`/`previous_centres` describe. The t=2 node only exists to give the sources' own
    gap a successor to be scored against.
    """
    coords = np.array([[0, 1, 2, 2], [1, 0, 0, 0], [1, 1, 3, 3], [2, 1, 3, 3]], dtype=np.int64)
    return TrackGraph(np.arange(4, dtype=np.int64), coords, np.empty((0, 2), dtype=np.int64))


def test_training_and_inference_velocity_agree(video_store: Path):
    """The SAME weights and the SAME previous gap give the trainer and the shipped scorer the same velocity.

    This is the drift the whole feature turns on: a head trained on a history it is never scored with is a head
    scored on an input distribution it never saw. Both regimes go through `GapHistory.velocity` — one softmax
    over the sources of one zero-velocity forward — so the equality below is structural, not a coincidence of
    these numbers, and a second definition of the step would break it.
    """
    model = _model(feat_dim=_WIDENED)
    scorer = EdgeTransformerScorer.of(model.detector, model.transformer, _RECIPE).eval()
    previous_gap, scored_gap = EdgeGap.over(video_store, _same_nodes(), "cpu")

    training = PriorVelocitySource(enabled=True).of(model, _sample(video_store, previous=True), nullcontext())
    with torch.no_grad():
        _, history = scorer._seed_logits(previous_gap, GapHistory())
    inference = history.velocity(torch.as_tensor(scored_gap.src_positions))

    assert training is not None
    assert torch.equal(inference, training)
