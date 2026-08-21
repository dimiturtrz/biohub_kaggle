"""Unit test for the joint forward — one backbone pass yields both detection maps and the edge logits.

The real backbone and transformer live in the pinned `external/` checkout (absent in CI); the autouse stubs
in `conftest` stand in with the same shape contracts, so this exercises the joint plumbing (single UNet pass
→ two detect-head reads + node-feature indexing → transformer) without them.
"""

from pathlib import Path

import torch
from torch import nn

from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.models.joint_model import _MAX_NODES, JointModel
from celltrack.models.prior_velocity import PriorVelocity
from celltrack.models.relative_position_transformer import RelativePositionEdgeTransformer
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from core.geometry import Spacing

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)
_GATE_UM = 10.0


def _joint_model(extra_features: int = 0) -> JointModel:
    out_channels = 2
    detector = TemporalUNetDetector(out_channels, (2, 4))  # stubbed backbone via conftest
    feat_dim = out_channels + 4 * _POS_EMBED_DIM + extra_features
    transformer = EdgeTransformerScorer._transformer_cls()(feat_dim=feat_dim, hidden_dim=8, n_heads=1, n_blocks=1)
    return JointModel(detector, transformer, downsample=(1, 1, 1))


class _StubHead(nn.Module):
    """The one structure the geometry wrapper reads for its head count — `blocks[0].cross_attn`.

    The suite-wide stub transformer is a bare linear with no attention blocks, so a model whose head must be
    geometry-wrapped stands one of these in instead (the wrapper's own forward/state fidelity is proven in
    `relative_position_transformer`; here it is only the model-level install + persistence seam).
    """

    def __init__(self) -> None:
        super().__init__()
        block = nn.Module()
        block.cross_attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.blocks = nn.ModuleList([block])


def _head_model() -> JointModel:
    return JointModel(TemporalUNetDetector(2, (2, 4)), _StubHead(), downsample=(1, 1, 1))


def test_forward():
    """One pair → both detection maps (frame shape), the s×u edge logits, and the node features behind them."""
    model = _joint_model().eval()
    frame_t, frame_t1 = torch.zeros(4, 8, 8), torch.zeros(4, 8, 8)
    source_positions = torch.tensor([[2, 4, 4]])  # 1 source
    target_positions = torch.tensor([[2, 4, 4], [1, 2, 2]])  # 2 targets
    out = model.forward(frame_t, frame_t1, source_positions, target_positions)
    assert out.detection_t.shape == (4, 8, 8) and out.detection_t1.shape == (4, 8, 8)
    assert out.edge_logits.shape == (1, 2)
    feature_dim = 2 + 4 * _POS_EMBED_DIM  # the appearance channels, then the sinusoidal (t, z, y, x) embed
    assert out.source_features.shape == (1, feature_dim) and out.target_features.shape == (2, feature_dim)


def test_forward_batch():
    """A batched forward computes each pair EQUIVALENTLY to a per-pair forward — the batching is transparent.

    This is the load-bearing property: a batch=B run must train on the same per-pair quantities a batch=1 run
    does (only the optimiser step is shared). The edge transformer runs as ONE padded, key-masked call over the
    whole batch rather than a per-pair loop (the 2.5x throughput win), so each pair's [:s, :u] block equals its
    unbatched logits up to floating point — the padded attention and the batch-size-dependent detection conv
    each select a different kernel, ~1e-6, far below anything training reads. The scene path (a compiled
    uniform-N head) keeps the per-pair loop.
    """
    torch.manual_seed(0)
    model = _joint_model().eval()
    frames = [(torch.randn(4, 8, 8), torch.randn(4, 8, 8)) for _ in range(2)]
    sources = [torch.tensor([[2, 4, 4]]), torch.tensor([[1, 2, 2]])]
    targets = [torch.tensor([[2, 4, 4], [1, 2, 2]]), torch.tensor([[3, 5, 5]])]

    with torch.no_grad():
        singles = [model.forward(ft, ft1, s, t) for (ft, ft1), s, t in zip(frames, sources, targets, strict=True)]
        windows = torch.stack([torch.stack([ft, ft1], dim=0) for ft, ft1 in frames], dim=0)  # (2, 2, Z, Y, X)
        batched = model.forward_batch(windows, sources, targets, [None, None])

    for single, batch in zip(singles, batched, strict=True):
        assert torch.allclose(single.edge_logits, batch.edge_logits, atol=1e-4)
        assert torch.allclose(single.detection_t, batch.detection_t, atol=1e-4)
        assert torch.allclose(single.source_features, batch.source_features, atol=1e-4)


def test_batched_forward():
    """The padded batched forward carries every pair on ONE `_MAX_NODES` shape, with the real-length masks."""
    torch.manual_seed(0)
    model = _joint_model().eval()
    frames = [(torch.randn(4, 8, 8), torch.randn(4, 8, 8)) for _ in range(2)]
    sources = [torch.tensor([[2, 4, 4]]), torch.tensor([[1, 2, 2]])]
    targets = [torch.tensor([[2, 4, 4], [1, 2, 2]]), torch.tensor([[3, 5, 5]])]
    windows = torch.stack([torch.stack([ft, ft1], dim=0) for ft, ft1 in frames], dim=0)

    with torch.no_grad():
        batched = model.batched_forward(windows, sources, targets)

    assert batched.logits.shape == (2, _MAX_NODES, _MAX_NODES)  # padded to the fixed shape
    assert batched.source_lengths == (1, 1) and batched.target_lengths == (2, 1)
    assert batched.source_mask[:, :2].tolist() == [[True, False], [True, False]]  # real then pad


def test_fits_batched():
    """A pair fits the padded path iff BOTH its node counts are within `_MAX_NODES` — the boundary `_pad_fixed` guards.

    This is the eligibility guard behind a raised default batch size: an oversized pair (either axis) must fall
    back to the ragged per-pair loop rather than reach the pad that would raise, so the two are the same boundary.
    """
    assert JointModel.fits_batched(_MAX_NODES, _MAX_NODES)  # exactly at the pad is still one block
    assert not JointModel.fits_batched(_MAX_NODES + 1, 1)  # sources overflow
    assert not JointModel.fits_batched(1, _MAX_NODES + 1)  # targets overflow


def test_forward_batch_oversized_routes_ragged():
    """A pair whose node set overflows `_MAX_NODES` routes to the ragged per-pair `_heads` path, not the pad.

    This is the boundary `fits_batched` names from the other side: without the fallback, `batched_forward`'s
    `_pad_fixed` raises on any pair above the fixed pad (the v1 dense-corpus crash at 769 nodes). The batched
    block only shortcuts the `_heads` reference, so an oversized pair must still forward — returning its real
    `(s, u)` block — rather than crash. One oversized pair in the batch pulls the whole batch to the ragged path.
    """
    model = _joint_model().eval()
    frame_t, frame_t1 = torch.zeros(4, 8, 8), torch.zeros(4, 8, 8)
    windows = torch.stack([torch.stack([frame_t, frame_t1], dim=0)], dim=0)  # (1, 2, Z, Y, X)
    oversized_sources = [torch.tensor([[2, 4, 4]]).repeat(_MAX_NODES + 1, 1)]  # 769 sources — one over the pad
    targets = [torch.tensor([[2, 4, 4], [1, 2, 2]])]

    with torch.no_grad():
        out = model.forward_batch(windows, oversized_sources, targets, [None])  # must not raise on the pad

    assert len(out) == 1
    assert out[0].edge_logits.shape == (_MAX_NODES + 1, 2)  # the real oversized (s, u), carried ragged


def test_unpad():
    """`BatchedForward.unpad` slices each pair's [:s, :u] block back out — equal to the per-pair forward."""
    torch.manual_seed(0)
    model = _joint_model().eval()
    frames = [(torch.randn(4, 8, 8), torch.randn(4, 8, 8)) for _ in range(2)]
    sources = [torch.tensor([[2, 4, 4]]), torch.tensor([[1, 2, 2]])]
    targets = [torch.tensor([[2, 4, 4], [1, 2, 2]]), torch.tensor([[3, 5, 5]])]
    windows = torch.stack([torch.stack([ft, ft1], dim=0) for ft, ft1 in frames], dim=0)

    with torch.no_grad():
        unpadded = model.batched_forward(windows, sources, targets).unpad()

    assert [f.edge_logits.shape for f in unpadded] == [(1, 2), (1, 1)]  # each pair's real (s, u)


def test_compile_scene_head():
    """After compiling the scene head for a node count, a pair of THAT count routes to the compiled transformer
    and computes the same edge logits the eager head does — the routing is transparent, only the launch differs."""
    model = _joint_model().eval()
    frame_t, frame_t1 = torch.zeros(4, 8, 8), torch.zeros(4, 8, 8)
    source_positions = torch.tensor([[2, 4, 4], [1, 2, 2]])  # 2 sources — the count we compile for
    target_positions = torch.tensor([[2, 4, 4], [1, 2, 2]])

    with torch.no_grad():
        eager = model.forward(frame_t, frame_t1, source_positions, target_positions)
        model.compile_scene_head(2)
        routed = model.forward(frame_t, frame_t1, source_positions, target_positions)

    assert model._scene_nodes == 2
    assert torch.allclose(eager.edge_logits, routed.edge_logits, atol=1e-4)


def test_forward_prior_velocity():
    """A source prior step widens both frames' features by three columns — the targets' being the zero vector."""
    model = _joint_model(extra_features=PriorVelocity.DIM).eval()
    frame_t, frame_t1 = torch.zeros(4, 8, 8), torch.zeros(4, 8, 8)
    source_positions = torch.tensor([[2, 4, 4]])
    target_positions = torch.tensor([[2, 4, 4], [1, 2, 2]])
    velocity = torch.tensor([[0.0, 100.0, -100.0]])
    out = model.forward(frame_t, frame_t1, source_positions, target_positions, velocity)
    feature_dim = 2 + 4 * _POS_EMBED_DIM
    assert out.source_features.shape == (1, feature_dim + PriorVelocity.DIM)
    assert torch.equal(out.source_features[:, feature_dim:], velocity / PriorVelocity.SCALE)
    assert torch.equal(out.target_features[:, feature_dim:], torch.zeros(2, PriorVelocity.DIM))
    assert out.edge_logits.shape == (1, 2)


def test_forward_batch_prior_velocity():
    """A batched forward carrying per-source velocities equals the per-pair velocity forward — the +3 velocity
    columns thread through the padded path the same way the base features do, so the 2x-batched velocity arm
    trains on the same quantities the per-pair one did (the property `celltrack-decisive-arm-speed-floor` needs).
    """
    torch.manual_seed(0)
    model = _joint_model(extra_features=PriorVelocity.DIM).eval()
    frames = [(torch.randn(4, 8, 8), torch.randn(4, 8, 8)) for _ in range(2)]
    sources = [torch.tensor([[2, 4, 4]]), torch.tensor([[1, 2, 2]])]
    targets = [torch.tensor([[2, 4, 4], [1, 2, 2]]), torch.tensor([[3, 5, 5]])]
    velocities = [torch.tensor([[0.0, 100.0, -100.0]]), torch.tensor([[50.0, 0.0, 25.0]])]

    with torch.no_grad():
        singles = [
            model.forward(ft, ft1, s, t, v)
            for (ft, ft1), s, t, v in zip(frames, sources, targets, velocities, strict=True)
        ]
        windows = torch.stack([torch.stack([ft, ft1], dim=0) for ft, ft1 in frames], dim=0)
        batched = model.forward_batch(windows, sources, targets, velocities)

    for single, batch in zip(singles, batched, strict=True):
        assert torch.allclose(single.edge_logits, batch.edge_logits, atol=1e-4)
        assert torch.allclose(single.source_features, batch.source_features, atol=1e-4)


def test_relative_position_config():
    """A plain head reports no geometry; a wrapped one reports spacing/reach that rebuilds the same config."""
    assert _head_model().relative_position_config() is None  # plain head persists as before

    model = _head_model()
    model.install_relative_position(_SPACING, _GATE_UM, "cpu")
    persisted = model.relative_position_config()
    assert persisted is not None
    assert persisted["gate_um"] == _GATE_UM
    assert list(persisted["spacing"]) == [_SPACING.z, _SPACING.y, _SPACING.x]

    rebuilt = JointModel._relative_config(persisted)
    assert rebuilt.enabled and rebuilt.gate_um == _GATE_UM
    assert tuple(rebuilt.spacing.as_array()) == tuple(_SPACING.as_array())


def test_install_relative_position():
    """Installing the geometry wrapper swaps the plain head for a zero-init distance-bias wrapper on the model."""
    model = _head_model()
    assert not isinstance(model.transformer, RelativePositionEdgeTransformer)

    model.install_relative_position(_SPACING, _GATE_UM, "cpu")

    assert isinstance(model.transformer, RelativePositionEdgeTransformer)
    assert model.relative_position_config() is not None  # now reports its geometry to the checkpoint


def test_serialisable():
    """The model reports its own checkpoint half in ONE call — both state dicts plus the facts to rebuild the heads.

    This is the seam `JointCheckpoint.payload` reads instead of plucking attributes across the model's internals
    (the feature-envy fix). A plain head reports `head="pack"`, `norm="batch"` and no relative geometry; once the
    geometry wrapper is installed the same call reports the geometry, so a checkpoint rebuilds the wrapped head.
    """
    model = _head_model()
    owned = model.serialisable()
    assert set(owned) >= {
        "detector_state",
        "transformer_state",
        "temporal_position",
        "relative_position",
        "norm",
        "head",
    }
    assert owned["head"] == "pack" and owned["norm"] == "batch"
    assert owned["relative_position"] is None

    model.install_relative_position(_SPACING, _GATE_UM, "cpu")
    assert model.serialisable()["relative_position"] is not None  # the wrapper's geometry now travels with the payload


def test_from_checkpoint(tmp_path: Path):
    """A checkpoint written by the trainer rebuilds both heads at the shape its weights were trained in."""
    out_channels = 2
    detector = TemporalUNetDetector(out_channels, (2, 4))
    transformer = EdgeTransformerScorer._transformer_cls()(
        feat_dim=out_channels + 4 * _POS_EMBED_DIM, hidden_dim=128, n_heads=4, n_blocks=4
    )
    saved = JointModel(detector, transformer, downsample=(1, 4, 4))
    path = tmp_path / "joint.pt"
    torch.save(  # exactly the payload JointTrainer._save_checkpoint writes
        {
            "detector_state": saved.detector.state_dict(),
            "transformer_state": saved.transformer.state_dict(),
            "config": {"out_channels": out_channels, "layers": [2, 4], "downsample": [1, 4, 4]},
        },
        path,
    )

    loaded = JointModel.from_checkpoint(path)

    assert loaded.downsample == (1, 4, 4)
    for restored, original in zip(
        loaded.detector.state_dict().values(), saved.detector.state_dict().values(), strict=True
    ):
        assert torch.equal(restored, original)
