"""HOCT-style two-stage association head — a drop-in for the pilkwang ``SimpleNodeTransformer``.

The pack head the confusor defeats scores a source→target pair *first-order*: it cross-attends the two
frames' feature vectors (position-blind) and reads geometry only as a **radial** offset ``(c_t − c_j)/100``
fed to an MLP. Two rivals for one target that carry near-identical UNet features therefore get near-identical
scores — the head has no way to compare one candidate edge's geometry against a competitor's.

This head follows HOCT (arxiv 2607.11754, #1 CTC Linking) — the frontier method whose own diagnosis is our
exact failure (candidate graphs with near-random homophily, node embeddings uninformative). It adds the two
stages the pack head lacks:

* **Stage 1 — 3D-RoPE node self-attention.** Each frame's nodes self-attend with a *rotary* positional
  encoding over ``(z, y, x)``, so the QK product reads relative position (and thus, across the two frames,
  velocity) natively rather than through a bolted-on scalar. ``Ln`` layers per frame.
* **Stage 2 — edge-to-edge self-attention with a line-to-line bias.** Every in-gate candidate edge becomes a
  token. Edges whose midpoints are spatially near attend to each other under
  ``A_h(m,n) = q·k/√D + σ_h·softplus(α_h)·dline(e_m, e_n)``, where ``dline`` is the minimum distance between
  the two 3D candidate-edge *segments* and ``σ_h`` alternates ±1 across heads (attractive / repulsive). A true
  track edge lies near-parallel to the coherent local flow (small ``dline`` to its neighbours); a spurious
  crossing edge cuts across it. ``Le`` layers, then each edge token projects to its scalar logit.

The rotary encoding, the two attention blocks and the segment-distance geometry are all private to this head
(nothing else consumes them) and so live nested within it. The forward signature, the ``.proj`` projection
(whose ``in_features`` the mount reads to detect the widened prior-velocity columns) and the
``(…, N_t, N_t1)`` logit shape all match ``SimpleNodeTransformer`` exactly, so the head mounts and trains at
the three existing wiring sites with no change to the scorer or the trainer.
"""

from __future__ import annotations

from typing import cast, override

import torch
from jaxtyping import Float
from torch import Tensor, nn

_AXES = 3  # RoPE rotates the head dimension over (z, y, x)
_ROPE_PAIR = 2  # a rotary "pair" is two feature dims rotated together
_NEG_INF = -1.0e9  # additive mask value for out-of-gate / out-of-neighbourhood attention
_EPS = 1e-9  # degenerate-segment guard in the parametric min-distance solve
_UNBATCHED_NDIM = 2  # a bare (N, D) feature tensor, promoted to (1, N, D) for the batched path

# Budget for ONE edge-attention score tile. The dense-crowd regime this head exists for gates ~10k candidate
# edges per frame pair, and a full (heads, e, e) score matrix there is 3.3GB in fp32 — the allocation that
# OOMed the detected-node arm at batch 1 with 8.4GB free. The block therefore attends a slice of query rows at
# a time, sized so one tile stays an order of magnitude under that headroom while the dense case still needs
# only ~13 slices. Below the budget the arithmetic yields a single slice, so small frames keep the flat path.
_SCORE_TILE_BYTES = 256 * 1024**2


class HoctEdgeTransformer(nn.Module):
    """Two-stage association head: RoPE node self-attention, then line-to-line edge-to-edge self-attention.

    Drop-in for ``SimpleNodeTransformer`` — same constructor keywords, same six-argument forward, same
    ``.proj`` projection and ``(…, N_t, N_t1)`` logit output. ``n_blocks`` sets both stage depths. The private
    building blocks nest below: ``_Rope3D`` (rotary position), ``_NodeAttentionBlock`` (RoPE self-attention
    over one frame's nodes) and ``_EdgeAttentionBlock`` (edge-to-edge self-attention carrying the ``dline``
    line-to-line bias).
    """

    HEAD_NAME = "hoct"  # the checkpoint config value that rebuilds this class (`EdgeTransformerScorer._head_cls`)

    class _Rope3D(nn.Module):
        """Rotary position encoding over three spatial axes, applied to the leading ``rot_dim`` of each head.

        The head dimension is split into three equal rotary blocks (z, y, x); each block rotates its feature
        pairs by an angle proportional to that coordinate. Any dimensions past ``3 · ⌊head_dim / 6⌋`` pass
        through unrotated (standard partial-RoPE), so a head width that is not a multiple of six is still valid.
        """

        @staticmethod
        def _frequencies(rot_dim: int, base: float) -> Float[Tensor, "half"]:
            """The inverse-frequency bank for one axis' rotary block — geometric over the axis' rotary pairs."""
            half = rot_dim // (_AXES * _ROPE_PAIR)
            exponents = torch.arange(half, dtype=torch.float32) / max(half, 1)
            return base**-exponents

        def __init__(self, head_dim: int, base: float = 100.0) -> None:
            super().__init__()
            self.rot_dim = _AXES * _ROPE_PAIR * (head_dim // (_AXES * _ROPE_PAIR))
            self.log_freqs = nn.Parameter(self._frequencies(self.rot_dim, base).log())

        @override
        def forward(self, x: Float[Tensor, "b h n d"], coords: Float[Tensor, "b n 3"]) -> Float[Tensor, "b h n d"]:
            """Rotate the leading ``rot_dim`` features of ``x`` by the angles the node coordinates induce."""
            if self.rot_dim == 0:
                return x
            rotated, passthrough = x[..., : self.rot_dim], x[..., self.rot_dim :]
            freqs = self.log_freqs.exp()  # (half,)
            angles = coords.unsqueeze(-1) * freqs  # (b, n, 3, half)
            angles = angles.flatten(-2, -1).unsqueeze(1)  # (b, 1, n, 3*half) broadcast over heads
            cos, sin = angles.cos(), angles.sin()
            first, second = rotated[..., 0::2], rotated[..., 1::2]
            out = torch.empty_like(rotated)
            out[..., 0::2] = first * cos - second * sin
            out[..., 1::2] = first * sin + second * cos
            return torch.cat([out, passthrough], dim=-1)

    class _NodeAttentionBlock(nn.Module):
        """One RoPE self-attention block over a frame's nodes — pre-norm attention and MLP with residuals."""

        def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = hidden_dim // n_heads
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
            self.out = nn.Linear(hidden_dim, hidden_dim)
            self.rope = HoctEdgeTransformer._Rope3D(self.head_dim)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
                nn.Dropout(dropout),
            )

        @override
        def forward(
            self, x: Float[Tensor, "b n d"], coords: Float[Tensor, "b n 3"], key_mask: Float[Tensor, "b n"] | None
        ) -> Float[Tensor, "b n d"]:
            """Self-attend the nodes with rotary position, then apply the MLP — both residual."""
            batch, nodes, _ = x.shape
            qkv = self.qkv(self.norm1(x)).view(batch, nodes, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (b, h, n, head_dim)
            q, k = self.rope(q, coords), self.rope(k, coords)
            attn_mask = None
            if key_mask is not None:
                attn_mask = torch.where(key_mask[:, None, None, :], 0.0, _NEG_INF)
            attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attended = attended.transpose(1, 2).reshape(batch, nodes, -1)
            x = x + self.out(attended)
            return x + self.mlp(self.norm2(x))

    class _EdgeAttentionBlock(nn.Module):
        """One edge-to-edge self-attention block — the line-to-line bias is the ``dline`` term added to scores."""

        def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = hidden_dim // n_heads
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
            self.out = nn.Linear(hidden_dim, hidden_dim)
            sign = torch.where(torch.arange(n_heads) % 2 == 0, 1.0, -1.0)  # σ_h alternates ± across heads
            self.register_buffer("sign", sign.view(1, n_heads, 1, 1))
            self.alpha = nn.Parameter(torch.zeros(1, n_heads, 1, 1))  # softplus(α) is the bias magnitude, ≥0
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
                nn.Dropout(dropout),
            )

        @staticmethod
        def attention_rows(count: int, n_heads: int, itemsize: int) -> int:
            """Query rows whose ``(n_heads, rows, count)`` score tile fits the budget — at least one, at most all."""
            return max(1, min(count, _SCORE_TILE_BYTES // max(1, n_heads * count * itemsize)))

        @override
        def forward(
            self, edges: Float[Tensor, "e d"], dline: Float[Tensor, "e e"], neighbour: Float[Tensor, "e e"]
        ) -> Float[Tensor, "e d"]:
            """Attend each edge to its spatial neighbours, biased by the segment distance to each one."""
            count, _ = edges.shape
            qkv = self.qkv(self.norm1(edges)).view(count, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(1, 2, 0, 3)  # each (h, e, head_dim)
            sign = cast(Tensor, self.sign)  # a registered buffer is Tensor | Module to the checker
            magnitude = sign.squeeze(0) * torch.nn.functional.softplus(self.alpha.squeeze(0))  # (h, 1, 1)
            attended = torch.empty_like(q)
            rows = self.attention_rows(count, self.n_heads, q.element_size())
            for lo in range(0, count, rows):
                hi = min(lo + rows, count)
                scores = torch.einsum("hid,hjd->hij", q[:, lo:hi], k) / self.head_dim**0.5
                scores = scores + magnitude * dline[lo:hi].unsqueeze(0) + neighbour[lo:hi].unsqueeze(0)
                attended[:, lo:hi] = torch.einsum("hij,hjd->hid", scores.softmax(dim=-1), v)
            attended = attended.transpose(0, 1).reshape(count, -1)
            edges = edges + self.out(attended)
            return edges + self.mlp(self.norm2(edges))

    @staticmethod
    def _segment_distance(
        p0: Float[Tensor, "*e 3"], p1: Float[Tensor, "*e 3"], q0: Float[Tensor, "*f 3"], q1: Float[Tensor, "*f 3"]
    ) -> Float[Tensor, "*e *f"]:
        """Minimum distance between every ``p`` segment and every ``q`` segment — the ``dline`` of the edge bias.

        Segments are ``p0→p1`` and ``q0→q1``; the closest points are found by the standard clamped parametric
        solve (Ericson, *Real-Time Collision Detection*) and broadcast so the last ``p`` axis pairs against the
        last ``q`` axis. Degenerate (zero-length) segments fall out of the clamping without a special case.
        """
        p0, p1 = p0.unsqueeze(-2), p1.unsqueeze(-2)  # (..., e, 1, 3)
        q0, q1 = q0.unsqueeze(-3), q1.unsqueeze(-3)  # (..., 1, f, 3)
        d1, d2, r = p1 - p0, q1 - q0, p0 - q0
        a = (d1 * d1).sum(-1)
        e = (d2 * d2).sum(-1)
        f = (d2 * r).sum(-1)
        b = (d1 * d2).sum(-1)
        c = (d1 * r).sum(-1)
        denom = a * e - b * b
        s = torch.where(
            denom > _EPS, ((b * f - c * e) / denom.clamp_min(_EPS)).clamp(0.0, 1.0), torch.zeros_like(denom)
        )
        t = (b * s + f) / e.clamp_min(_EPS)
        t_clamped = t.clamp(0.0, 1.0)
        s = ((b * t_clamped - c) / a.clamp_min(_EPS)).clamp(0.0, 1.0)
        closest = (p0 + s.unsqueeze(-1) * d1) - (q0 + t_clamped.unsqueeze(-1) * d2)
        return closest.norm(dim=-1)

    def __init__(  # noqa: PLR0913 — mirrors SimpleNodeTransformer's constructor kwargs; a drop-in must match them
        self,
        feat_dim: int = 33,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        neighbour_um: float = 24.0,
        gate_um: float = 12.0,
        voxel_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    ) -> None:
        super().__init__()
        self.neighbour_um = neighbour_um
        self.gate_um = gate_um
        # Node coords arrive in RAW-grid voxels (z coarse, y/x pooled-fine); the gate, the line-to-line dline
        # and the neighbour locality are all PHYSICAL, so coords are scaled to micrometres before any distance.
        # Default = the corpus voxel size (pooled 1.625 µm ÷ POOLED_BY (1,4,4)); a run whose spacing differs
        # passes its own. A buffer so it follows `.to(device)`.
        self.register_buffer("voxel_um", torch.tensor(voxel_um, dtype=torch.float32).view(1, _AXES))
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.node_blocks = nn.ModuleList(
            self._NodeAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)
        )
        self.edge_in = nn.Linear(2 * hidden_dim, hidden_dim)
        self.edge_blocks = nn.ModuleList(
            self._EdgeAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.edge_logit = nn.Linear(hidden_dim, 1)

    @override
    def forward(
        self,
        feat_t: Float[Tensor, "*b n_t d"],
        feat_t1: Float[Tensor, "*b n_t1 d"],
        coords_t: Float[Tensor, "*b n_t 3"],
        coords_t1: Float[Tensor, "*b n_t1 3"],
        mask_t: Float[Tensor, "*b n_t"] | None = None,
        mask_t1: Float[Tensor, "*b n_t1"] | None = None,
    ) -> Float[Tensor, "*b n_t n_t1"]:
        """Edge logits between the two frames' detections; accepts unbatched ``(N, D)`` or batched ``(B, N, D)``."""
        unbatched = feat_t.ndim == _UNBATCHED_NDIM
        if unbatched:
            feat_t, feat_t1 = feat_t.unsqueeze(0), feat_t1.unsqueeze(0)
            coords_t, coords_t1 = coords_t.unsqueeze(0), coords_t1.unsqueeze(0)
        node_t = self._encode_nodes(feat_t, coords_t, mask_t)
        node_t1 = self._encode_nodes(feat_t1, coords_t1, mask_t1)
        logits = torch.stack(
            [
                self._edge_logits(
                    node_t[b],
                    node_t1[b],
                    coords_t[b],
                    coords_t1[b],
                    None if mask_t is None else mask_t[b],
                    None if mask_t1 is None else mask_t1[b],
                )
                for b in range(node_t.shape[0])
            ]
        )
        return logits.squeeze(0) if unbatched else logits

    def _encode_nodes(
        self, feat: Float[Tensor, "b n d"], coords: Float[Tensor, "b n 3"], mask: Float[Tensor, "b n"] | None
    ) -> Float[Tensor, "b n d"]:
        """Run the RoPE node self-attention stage over one frame's detections."""
        nodes = self.norm_in(self.proj(feat))
        for block in self.node_blocks:
            nodes = block(nodes, coords, mask)
        return self.norm_out(nodes)

    def _edge_logits(  # noqa: PLR0913 — two frames' (feature, coord, mask) triples; splitting them hides the pairing
        self,
        node_t: Float[Tensor, "n_t d"],
        node_t1: Float[Tensor, "n_t1 d"],
        coords_t: Float[Tensor, "n_t 3"],
        coords_t1: Float[Tensor, "n_t1 3"],
        mask_t: Float[Tensor, "n_t"] | None,
        mask_t1: Float[Tensor, "n_t1"] | None,
    ) -> Float[Tensor, "n_t n_t1"]:
        """Edge-to-edge stage for one gap over only the IN-GATE candidate edges — the rest stay non-links.

        A dense token per source→target pair is O((N_t·N_t1)²) in the edge-to-edge attention — intractable the
        moment a frame carries more than ~150 detections (the synthetic corpus carries ~700). The frontier runs
        the stage over the motion-gated candidate edges only, so this enumerates just the source→target pairs
        whose PHYSICAL separation is within ``gate_um`` (and whose endpoints are both real, never a pad node),
        attends over that small set, and scatters each edge's logit back into the dense ``(N_t, N_t1)`` block.
        Non-candidate cells read ``_NEG_INF`` — a link the gate excludes, exactly the ``softmax`` mass the loss
        and the inference scorer already deny an out-of-gate pair. The gate is in micrometres (coords scaled by
        ``voxel_um``) so the anisotropic raw grid gives a spherical physical gate, not a z-elongated voxel one
        that would sweep every z-layer into the candidate set. ``gate_um`` (12) exceeds the largest true step
        (max 9.96 µm) with margin, so a real successor is never dropped to a non-link.
        """
        n_t, n_t1 = node_t.shape[0], node_t1.shape[0]
        out = coords_t.new_full((n_t, n_t1), _NEG_INF)
        voxel_um = cast(Tensor, self.voxel_um)  # a registered buffer is Tensor | Module to the checker
        phys_t, phys_t1 = coords_t * voxel_um, coords_t1 * voxel_um  # raw voxels → micrometres
        gate = torch.cdist(phys_t, phys_t1) <= self.gate_um  # (n_t, n_t1) in-gate candidates
        if mask_t is not None:
            gate = gate & mask_t.bool().unsqueeze(1)
        if mask_t1 is not None:
            gate = gate & mask_t1.bool().unsqueeze(0)
        src_idx, tgt_idx = gate.nonzero(as_tuple=True)  # (e,), (e,) — the gated candidate edges
        if src_idx.numel() == 0:
            return out
        edges = self.edge_in(torch.cat([node_t[src_idx], node_t1[tgt_idx]], dim=-1))  # (e, hidden)
        p0, p1 = phys_t[src_idx], phys_t1[tgt_idx]  # segment start = source, end = target, in micrometres
        dline = self._segment_distance(p0, p1, p0, p1)  # (e, e)
        midpoint = 0.5 * (p0 + p1)
        neighbour = torch.where(torch.cdist(midpoint, midpoint) <= self.neighbour_um, 0.0, _NEG_INF)
        for block in self.edge_blocks:
            edges = block(edges, dline, neighbour)
        # out holds float32 _NEG_INF sentinels (from float32 coords); under AMP the head computes in bf16, so
        # align the scattered logits to the destination dtype — index_put requires source and dest dtypes match.
        out[src_idx, tgt_idx] = self.edge_logit(edges).squeeze(-1).to(out.dtype)
        return out
