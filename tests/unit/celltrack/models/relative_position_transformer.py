"""Unit tests for the wrapper that puts the bias into the pack head's cross-attention.

The load-bearing case is `test_forward`: with the bias ENABLED but still
at its zero initialisation, the wrapper's logits must equal the unmodified head's. That single equality
proves two things at once — that a pretrained head does not move on the step the feature is switched on, and
that the wrapper's re-implemented attention stack is a faithful copy of the head's forward rather than an
approximation of it.

The wrapper reaches INSIDE the pack head (its blocks' `norm1` / `cross_attn` / `mlp`), which the suite-wide
`_StubTransformer` — a single linear — deliberately does not have. So these tests run against the REAL
`SimpleNodeTransformer` when the pinned `external/` checkout is present, and against `_ReferenceHead` (the
same submodule names, the same forward, at a size that fits a unit test) when it is not, so CI covers the
same paths without the third-party dependency.
"""

import sys
from typing import override

import torch
from torch import Tensor, nn

from celltrack.models.relative_position_bias import RelativePositionConfig
from celltrack.models.relative_position_transformer import RelativePositionEdgeTransformer
from celltrack.models.temporal_unet_detector import _EXT_SRC
from core.geometry import Spacing

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)
_GATE_UM = 10.0
_FEAT_DIM, _HIDDEN_DIM, _HEADS = 6, 8, 2
_PAIR_CHUNK = 2
_REL_DIVISOR = 100.0
_UNBATCHED_NDIM = 2


class _ReferenceBlock(nn.Module):
    """The pack's `CrossAttentionBlock` submodule names and forward, for a run without the `external/` checkout."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(_HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(_HIDDEN_DIM)
        self.cross_attn = nn.MultiheadAttention(_HIDDEN_DIM, _HEADS, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM), nn.GELU(), nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM))

    @override
    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor | None = None) -> Tensor:
        attended, _ = self.cross_attn(
            self.norm1(q), self.norm1(kv), self.norm1(kv), key_padding_mask=None if kv_mask is None else ~kv_mask
        )
        q = q + attended
        return q + self.mlp(self.norm2(q))


class _ReferenceHead(nn.Module):
    """The pack's `SimpleNodeTransformer` structure and forward — bi-directional stack, then the pair MLP."""

    def __init__(self, feat_dim: int, hidden_dim: int, n_heads: int, n_blocks: int) -> None:
        super().__init__()
        self.pair_chunk_size = _PAIR_CHUNK
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([_ReferenceBlock() for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    @override
    def forward(
        self,
        feat_t: Tensor,
        feat_t1: Tensor,
        coords_t: Tensor,
        coords_t1: Tensor,
        mask_t: Tensor | None = None,
        mask_t1: Tensor | None = None,
    ) -> Tensor:
        unbatched = feat_t.ndim == _UNBATCHED_NDIM
        if unbatched:
            feat_t, feat_t1 = feat_t.unsqueeze(0), feat_t1.unsqueeze(0)
            coords_t, coords_t1 = coords_t.unsqueeze(0), coords_t1.unsqueeze(0)
        q = self.norm_in(self.proj(feat_t))
        k = self.norm_in(self.proj(feat_t1))
        for block in self.blocks:
            q = block(q, k, kv_mask=mask_t1)
            k = block(k, q, kv_mask=mask_t)
        q, k = self.norm_out(q), self.norm_out(k)
        expanded_q = q.unsqueeze(2).expand(-1, -1, k.shape[1], -1)
        expanded_k = k.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
        rel = (coords_t.unsqueeze(2) - coords_t1.unsqueeze(1)) / _REL_DIVISOR
        logits = self.pair_mlp(torch.cat([expanded_q, expanded_k, rel], dim=-1)).squeeze(-1)
        return logits.squeeze(0) if unbatched else logits


def _head_cls() -> type[nn.Module]:
    """The real pack head from the pinned checkout, or the structural reference when it is absent."""
    if str(_EXT_SRC) not in sys.path:
        sys.path.insert(0, str(_EXT_SRC))
    try:
        from tracking_cellmot.models import SimpleNodeTransformer  # type: ignore[missing-import]  # noqa: PLC0415
    except ImportError:
        return _ReferenceHead
    return SimpleNodeTransformer


def _config(*, enabled: bool) -> RelativePositionConfig:
    return RelativePositionConfig(spacing=_SPACING, gate_um=_GATE_UM, enabled=enabled)


def _head() -> nn.Module:
    """A tiny pack-shaped head at random but fixed weights, in eval so dropout cannot mask a difference."""
    torch.manual_seed(0)
    head = _head_cls()(feat_dim=_FEAT_DIM, hidden_dim=_HIDDEN_DIM, n_heads=_HEADS, n_blocks=2)
    head.pair_chunk_size = _PAIR_CHUNK  # pyrefly: ignore[bad-argument-type]  # the pack head stores a plain int
    return head.eval()


def _inputs(sources: int = 3, targets: int = 4) -> tuple[torch.Tensor, ...]:
    """One unbatched frame pair: node features and full-resolution `(z, y, x)` voxel coordinates."""
    torch.manual_seed(1)
    return (
        torch.randn(sources, _FEAT_DIM),
        torch.randn(targets, _FEAT_DIM),
        torch.rand(sources, 3) * 20.0,
        torch.rand(targets, 3) * 20.0,
    )


def test_forward_disabled_is_the_pack_head():
    """OFF by default: the wrapper delegates, so the logits are the published head's own, bit for bit."""
    head = _head()
    wrapper = RelativePositionEdgeTransformer(head, _config(enabled=False))
    assert wrapper.bias is None

    with torch.no_grad():
        assert torch.equal(wrapper(*_inputs()), head(*_inputs()))


def test_forward():
    """ENABLED but untrained: the bias is exactly zero, so a pretrained head's logits do not move.

    This is the whole reason the bias is learned-and-zeroed rather than RoPE — a rotation of q and k would
    change every score the instant it was switched on, discarding the published weights' warm start.

    The tolerance is float noise, not slack: an all-zero additive attention mask is mathematically the
    identity but sends `MultiheadAttention` down its masked kernel instead of its fast path, which sums in a
    different order (~2e-8 here). The DISABLED path above is the bitwise one.
    """
    head = _head()
    wrapper = RelativePositionEdgeTransformer(head, _config(enabled=True))
    inputs = _inputs()

    with torch.no_grad():
        torch.testing.assert_close(wrapper.forward(*inputs), head(*inputs), rtol=0.0, atol=1e-6)


def test_forward_under_grad():
    """With grad on — the trainer's path, where every block and pair chunk is re-run under checkpointing —
    the logits are the same as inference's, and the bias receives gradient, so the new signal is learnable.
    """
    wrapper = RelativePositionEdgeTransformer(_head(), _config(enabled=True))
    with torch.no_grad():
        wrapper.bias.weight.normal_(std=0.5)
    inputs = _inputs()

    logits = wrapper(*inputs)
    logits.sum().backward()

    with torch.no_grad():
        torch.testing.assert_close(wrapper(*inputs), logits.detach(), rtol=0.0, atol=1e-6)
    assert not torch.equal(wrapper.bias.weight.grad, torch.zeros_like(wrapper.bias.weight))


def test_forward_batched_with_padding():
    """A padded batch keeps its shape and ignores the padded keys, the bias folded into one additive mask."""
    head = _head()
    wrapper = RelativePositionEdgeTransformer(head, _config(enabled=True))
    features_t, features_t1, coords_t, coords_t1 = _inputs()
    batch = (features_t.unsqueeze(0), features_t1.unsqueeze(0), coords_t.unsqueeze(0), coords_t1.unsqueeze(0))
    mask_t = torch.ones(1, coords_t.shape[0], dtype=torch.bool)
    mask_t1 = torch.tensor([[True, True, True, False]])

    with torch.no_grad():
        logits = wrapper(*batch, mask_t, mask_t1)
        unpadded = wrapper(batch[0], batch[1][:, :3], batch[2], batch[3][:, :3], mask_t, mask_t1[:, :3])

    assert logits.shape == (1, 3, 4)
    torch.testing.assert_close(logits[:, :, :3], unpadded, rtol=0.0, atol=1e-6)


def test_forward_separation_changes_attention():
    """Once the bias is non-zero, two pairs identical in APPEARANCE but different in SEPARATION score differently.

    Appearance is held exactly equal — both targets carry the same feature vector — so any difference in the
    logits can only have come through the geometry the attention now sees.
    """
    wrapper = RelativePositionEdgeTransformer(_head(), _config(enabled=True))
    with torch.no_grad():
        wrapper.bias.weight.normal_(std=1.0)
    torch.manual_seed(2)
    feature = torch.randn(1, _FEAT_DIM)
    features_t1 = feature.repeat(2, 1)  # identical appearance
    coords_t = torch.zeros(1, 3)
    near_then_far = torch.tensor([[0.0, 1.0, 0.0], [0.0, 30.0, 0.0]])

    with torch.no_grad():
        logits = wrapper(feature, features_t1, coords_t, near_then_far)

    assert not torch.isclose(logits[0, 0], logits[0, 1])


def test_load_pretrained():
    """A published `SimpleNodeTransformer` checkpoint loads into the wrapper's head key for key, unchanged."""
    published = _head().state_dict()
    wrapper = RelativePositionEdgeTransformer(_head(), _config(enabled=True))
    with torch.no_grad():
        wrapper.transformer.proj.weight.add_(1.0)  # move it away first, so the load is what restores it

    wrapper.load_pretrained(published)

    assert all(torch.equal(published[key], value) for key, value in wrapper.pretrained_state().items())
    assert torch.equal(wrapper.bias.weight, torch.zeros_like(wrapper.bias.weight))


def test_pretrained_state():
    """The state a checkpoint should carry is the pack's own key space — no `transformer.` prefix, no bias."""
    wrapper = RelativePositionEdgeTransformer(_head(), _config(enabled=True))

    state = wrapper.pretrained_state()

    assert state.keys() == _head().state_dict().keys()
    assert not any(key.startswith(("transformer.", "bias.")) for key in state)


def test_head_count():
    """The head count is read off the wrapped block's own attention, never re-declared by a caller."""
    assert RelativePositionEdgeTransformer.head_count(_head()) == _HEADS
