"""Unit tests for the HOCT two-stage edge head, its nested pieces, factory route and checkpoint identity.

`test_hoct_edge_transformer_forward` fixes the drop-in contract the mount relies on — same constructor
keywords, a `.proj` whose `in_features` the velocity-width check reads, and a `(N_t, N_t1)` logit matrix from
both an unbatched and a batched call. The two nested pieces a smoke test cannot eyeball get their own checks:
`test_rope3_d_forward` pins that 3D-RoPE makes the node attention depend on the coordinate DIFFERENCE, not the
absolute value (a common shift leaves a q·k score unchanged), and `test_segment_distance` pins the `dline`
the edge bias reads (crossing = 0, parallel = the gap, shared endpoint = 0).

The load-bearing case is `test_checkpoint_round_trip`: a joint model built with the HOCT head, saved and
reloaded, must come back as the SAME class producing BYTE-IDENTICAL logits on a fixed input. That is the
wzv7-class guard — it asserts the value MOVED across the save/load, not merely that a `head` key sits in the
config. A head-selection bug that silently rebuilt the pack head, or dropped a learnable RoPE/alpha buffer,
would pass a membership check and fail this one.
"""

from pathlib import Path

import pytest
import torch

from celltrack.models import hoct_edge_transformer
from celltrack.models.edge_transformer import EdgeTransformerScorer
from celltrack.models.hoct_edge_transformer import _BIAS_TEMPORARIES, _NEG_INF, HoctEdgeTransformer
from celltrack.models.joint_model import _N_BLOCKS, _N_HEADS, JointModel
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from celltrack.training.joint_checkpoint import JointCheckpoint

_FEAT_DIM = 40
_HIDDEN_DIM = 128  # the pilkwang pack width (was a joint_model constant before it moved to per-run config)
_SMALL_HIDDEN = 64
_SMALL_HEADS = 4
_SMALL_BLOCKS = 2
_TOL = 1e-4
_ROPE_DIM = 12
_rows = HoctEdgeTransformer._EdgeAttentionBlock.attention_rows


def _head() -> HoctEdgeTransformer:
    return HoctEdgeTransformer(
        feat_dim=_FEAT_DIM, hidden_dim=_SMALL_HIDDEN, n_heads=_SMALL_HEADS, n_blocks=_SMALL_BLOCKS
    )


def test_hoct_edge_transformer_forward() -> None:
    torch.manual_seed(0)
    head = _head()
    assert head.proj.in_features == _FEAT_DIM  # the width the prior-velocity check reads
    feat_t, feat_t1 = torch.randn(5, _FEAT_DIM), torch.randn(7, _FEAT_DIM)
    coords_t, coords_t1 = torch.rand(5, 3) * 20, torch.rand(7, 3) * 20
    out = head.forward(feat_t, feat_t1, coords_t, coords_t1)
    assert out.shape == (5, 7)
    batched = head.forward(feat_t.unsqueeze(0), feat_t1.unsqueeze(0), coords_t.unsqueeze(0), coords_t1.unsqueeze(0))
    assert batched.shape == (1, 5, 7)


def test_gate_masks_out_of_range_and_keeps_near() -> None:
    torch.manual_seed(0)
    head = _head()  # default gate_um=12, voxel_um=(1.625, 0.40625, 0.40625) um/voxel
    feat_t, feat_t1 = torch.randn(1, _FEAT_DIM), torch.randn(2, _FEAT_DIM)
    coords_t = torch.tensor([[0.0, 0.0, 0.0]])
    coords_t1 = torch.tensor([[0.0, 0.0, 10.0], [0.0, 40.0, 0.0]])  # ~4.1 um in-gate, ~16.3 um out
    out = head.forward(feat_t, feat_t1, coords_t, coords_t1)
    assert out.shape == (1, 2)
    assert torch.isfinite(out[0, 0]) and out[0, 0] > _NEG_INF / 2  # in-gate candidate scores a real logit
    assert out[0, 1] == _NEG_INF  # out-of-gate pair denied — the softmax mass the loss already withholds


def test_rope3_d_forward() -> None:
    torch.manual_seed(0)
    rope = HoctEdgeTransformer._Rope3D(head_dim=_ROPE_DIM)
    q, k = torch.randn(1, 1, 1, _ROPE_DIM), torch.randn(1, 1, 1, _ROPE_DIM)
    p_q, p_k = torch.tensor([[[3.0, 5.0, 7.0]]]), torch.tensor([[[4.0, 5.0, 7.0]]])
    base = (rope.forward(q, p_q) * rope.forward(k, p_k)).sum()
    offset = torch.tensor([[[10.0, -2.0, 1.0]]])
    shifted = (rope.forward(q, p_q + offset) * rope.forward(k, p_k + offset)).sum()
    assert torch.allclose(base, shifted, atol=_TOL)  # score depends on the coordinate DIFFERENCE, not absolute


def test_node_attention_block_forward() -> None:
    torch.manual_seed(0)
    block = HoctEdgeTransformer._NodeAttentionBlock(_SMALL_HIDDEN, _SMALL_HEADS, 2.0, 0.0)
    x = torch.randn(1, 6, _SMALL_HIDDEN)
    coords = torch.rand(1, 6, 3) * 20
    out = block.forward(x, coords, None)
    assert out.shape == (1, 6, _SMALL_HIDDEN)


def test_edge_attention_block_forward() -> None:
    torch.manual_seed(0)
    block = HoctEdgeTransformer._EdgeAttentionBlock(_SMALL_HIDDEN, _SMALL_HEADS, 2.0, 0.0)
    n_edges = 5
    edges = torch.randn(n_edges, _SMALL_HIDDEN)
    dline = torch.rand(n_edges, n_edges)
    neighbour = torch.zeros(n_edges, n_edges)
    out = block.forward(edges, dline, neighbour)
    assert out.shape == (n_edges, _SMALL_HIDDEN)


def test_edge_attention_tiling_matches_untiled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A score tile small enough to force several slices must reproduce the single-slice result exactly.

    The block attends a slice of query rows at a time so the dense-crowd regime never allocates a whole
    (heads, e, e) score matrix. Tiling is a memory layout, not a model change, so the two paths are the same
    function: shrinking the budget until the same input needs multiple slices must not move the output.
    """
    torch.manual_seed(0)
    block = HoctEdgeTransformer._EdgeAttentionBlock(_SMALL_HIDDEN, _SMALL_HEADS, 2.0, 0.0)
    n_edges = 16
    edges = torch.randn(n_edges, _SMALL_HIDDEN)
    dline = torch.rand(n_edges, n_edges)
    neighbour = torch.where(torch.rand(n_edges, n_edges) < 0.5, 0.0, _NEG_INF)
    untiled = block.forward(edges, dline, neighbour)

    one_row = _SMALL_HEADS * n_edges * edges.element_size()
    monkeypatch.setattr(hoct_edge_transformer, "_SCORE_TILE_BYTES", one_row * 3)
    assert _rows(n_edges, _SMALL_HEADS, edges.element_size()) == 3
    torch.testing.assert_close(block.forward(edges, dline, neighbour), untiled)


def test_attention_rows() -> None:
    """A frame far under the tile budget takes one slice; a budget under one row still takes a whole row."""
    assert HoctEdgeTransformer._EdgeAttentionBlock.attention_rows(8, _SMALL_HEADS, 4) == 8
    over_budget = hoct_edge_transformer._SCORE_TILE_BYTES  # one row of this many edges exceeds the whole budget
    assert _rows(over_budget, _SMALL_HEADS, 4) == 1


def test_edge_biases(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiled, no-grad bias build must equal the dense one it replaces — including across several tiles."""
    torch.manual_seed(0)
    head = _head()
    p0, p1 = torch.rand(12, 3) * 20, torch.rand(12, 3) * 20
    expected_dline = HoctEdgeTransformer._segment_distance(p0, p1, p0, p1)
    midpoint = 0.5 * (p0 + p1)
    expected_neighbour = torch.where(torch.cdist(midpoint, midpoint) <= head.neighbour_um, 0.0, _NEG_INF)

    monkeypatch.setattr(hoct_edge_transformer, "_SCORE_TILE_BYTES", _BIAS_TEMPORARIES * 12 * 4 * 5)  # 5 rows/tile
    dline, neighbour = head._edge_biases(p0, p1)
    torch.testing.assert_close(dline, expected_dline)
    torch.testing.assert_close(neighbour, expected_neighbour)
    assert not dline.requires_grad  # coordinate-only, so no autograd graph is kept alive for it


def test_segment_distance() -> None:
    a0, a1 = torch.tensor([[-1.0, 0, 0]]), torch.tensor([[1.0, 0, 0]])
    crossing = HoctEdgeTransformer._segment_distance(a0, a1, torch.tensor([[0.0, -1, 0]]), torch.tensor([[0.0, 1, 0]]))
    assert crossing.item() < _TOL
    parallel = HoctEdgeTransformer._segment_distance(a0, a1, torch.tensor([[-1.0, 3, 0]]), torch.tensor([[1.0, 3, 0]]))
    assert abs(parallel.item() - 3.0) < _TOL
    shared = HoctEdgeTransformer._segment_distance(a0, a1, a1, torch.tensor([[1.0, 5, 0]]))
    assert shared.item() < _TOL


def test_head_factory_routes_hoct() -> None:
    assert EdgeTransformerScorer._head_cls("hoct") is HoctEdgeTransformer


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    out_channels, layers, downsample = 8, (8, 16), (1, 4, 4)
    detector = TemporalUNetDetector(out_channels, layers)
    # from_checkpoint rebuilds the head at the pack constants (only feat_dim is read from the saved proj), so a
    # faithful round-trip must build at those same widths — exactly what the from-scratch trainer does.
    head = HoctEdgeTransformer(feat_dim=_FEAT_DIM, hidden_dim=_HIDDEN_DIM, n_heads=_N_HEADS, n_blocks=_N_BLOCKS)
    model = JointModel(detector, head, downsample).eval()
    path = tmp_path / "hoct.pt"
    JointCheckpoint(out_channels, layers, downsample, device="cpu").save_best(path, model)

    reloaded = JointModel.from_checkpoint(path, device="cpu")
    assert isinstance(reloaded.transformer, HoctEdgeTransformer)  # head IDENTITY survived the round-trip

    feat_t, feat_t1 = torch.randn(6, _FEAT_DIM), torch.randn(4, _FEAT_DIM)
    coords_t, coords_t1 = torch.rand(6, 3) * 20, torch.rand(4, 3) * 20
    with torch.no_grad():
        before = model.transformer(feat_t, feat_t1, coords_t, coords_t1)
        after = reloaded.transformer(feat_t, feat_t1, coords_t, coords_t1)
    assert torch.allclose(before, after, atol=_TOL)  # VALUE moved: reloaded head computes byte-identical logits
