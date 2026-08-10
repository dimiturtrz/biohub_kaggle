"""Unit tests for the training-time prior-velocity feed — where the head's history input comes from.

The maths of the feature is covered beside `PriorVelocity`; here the questions are the wiring ones: off is
off (no tensor at all, not a zero one), a pair without a predecessor yields exact zeros, and a pair with one
yields the model's own previous-gap displacement.
"""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.data.joint_dataset import PairDataset, PairSample, PairTarget
from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.models.prior_velocity import PriorVelocity
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from celltrack.training.prior_velocity_source import PriorVelocitySource
from core.data.video import ImageStatistics

_OUT_CHANNELS = 2
_DOWNSAMPLE = (1, 1, 1)


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
        zarr_path=video_store,
        timepoint=1 if previous else 0,
        q_low=float(quantiles["0.001"]),
        q_high=float(quantiles["0.999"]),
        source_centres=np.array([[0, 0, 0], [1, 3, 3]], dtype=np.int64),
        target_centres=np.array([[1, 3, 3]], dtype=np.int64),
        edge_matrix=np.array([[0.0], [1.0]], dtype=np.float32),
        previous_centres=np.array([[1, 2, 2]], dtype=np.int64) if previous else None,
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
    model = _model(feat_dim=_OUT_CHANNELS + 4 * _POS_EMBED_DIM + PriorVelocity.DIM)
    source = PriorVelocitySource(enabled=True)

    velocity = source.of(model, _sample(video_store, previous=True), torch.autocast("cpu", enabled=False))

    assert velocity is not None
    assert velocity.tolist() == [[-1.0, -2.0, -2.0], [0.0, 1.0, 1.0]]
