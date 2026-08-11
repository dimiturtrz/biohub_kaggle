"""Relative spatial geometry inside the edge transformer's CROSS-ATTENTION — the pack's head, wrapped not edited.

The pack's head is not geometry-blind at the END: `SimpleNodeTransformer` concatenates the candidate
displacement `rel = (coords_t - coords_t1) / 100.0` onto every expanded pair before `pair_mlp`. The layers
BEFORE it are. A node's representation is built by attending over the other frame's nodes with no notion of
how far away any of them is, so "which of my neighbours should inform me" is decided on appearance alone —
and appearance is exactly what a field of near-identical nuclei does not disambiguate. Trackastra's verified
mechanism is the opposite: relative position injected into EVERY attention layer (RoPE over space and time)
plus a mask beyond a distance `d_max` (`research/deep_dives/2026-08-10_association-and-motion-sota-corrective.md`).

This adds the spatial half of that, in the one form a warm start survives: `DistanceAttentionBias` added to
every cross-attention score, its parameters starting at exact zero. RoPE proper is parameter-free but it is
not identity-preserving — rotating q and k changes every score the moment it is switched on, which is fine
for Trackastra (trained with it from scratch) and wrong for us (a published head we must not move at step 0).
The zero-initialised bias is the same trade `PriorVelocity.widen_linear_inputs` makes for the input
projection: the pretrained weights load unchanged, the output is identical on the first step, and the new
signal is learned from zero gain upward.

The TEMPORAL half of Trackastra's encoding is deliberately NOT built. Our window is T=2: every scored pair
spans exactly one frame gap, so a relative temporal offset is the same constant for every pair and any
encoding of it is a constant added to all scores — an identity under the softmax over sources. It becomes
live only if the window ever widens past two frames.
"""

from __future__ import annotations

from typing import cast, override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as grad_ckpt

from celltrack.models.prior_velocity import PriorVelocity
from celltrack.models.relative_position_bias import DistanceAttentionBias, RelativePositionConfig

_UNBATCHED_NDIM = 2
_MASKED_SCORE = float("-inf")


def _part(module: nn.Module, name: str) -> nn.Module:
    """A named child of a module, typed — `nn.Module.__getattr__` returns `Tensor | Module`, which is uncallable."""
    return cast(nn.Module, getattr(module, name))


class RelativePositionEdgeTransformer(nn.Module):
    """The pack's `SimpleNodeTransformer` with relative spatial position added to its cross-attention scores.

    A wrapper, never an edit: `external/kaggle-cell-tracking-competition/` is a pinned third-party checkout and
    the published weights were trained in ITS class, so the head is held as a child module and loaded unchanged
    (`load_pretrained` / `pretrained_state`). Disabled — the default — the forward is a straight delegation and
    the logits are the pack's own, bit for bit. Enabled, the attention stack is run here so each block's scores
    can take the additive geometry term, and the pair head that follows is the pack's own `norm_out` and
    `pair_mlp` under the pack's own chunking, so the only difference from the original forward is the bias.
    """

    def __init__(self, transformer: nn.Module, config: RelativePositionConfig) -> None:
        super().__init__()
        self.transformer = transformer
        self.config = config
        self.bias = DistanceAttentionBias(self.head_count(transformer), config) if config.enabled else None

    @staticmethod
    def head_count(transformer: nn.Module) -> int:
        """How many attention heads the wrapped head has — read off its own blocks, never re-declared."""
        blocks = cast(nn.ModuleList, transformer.blocks)
        return int(cast(nn.MultiheadAttention, cast(nn.Module, blocks[0]).cross_attn).num_heads)

    def load_pretrained(self, state: dict[str, Tensor]) -> None:
        """Load a published `SimpleNodeTransformer` state dict into the wrapped head, key for key, unchanged.

        The bias parameters are not in such a checkpoint and are not meant to be: they are the new, zero part.
        Loading strictly into the CHILD rather than into the wrapper is what keeps the pack's keys valid — a
        `transformer.`-prefixed rewrite would be a second spelling of the same weights and could drift.
        """
        self.transformer.load_state_dict(state)

    def pretrained_state(self) -> dict[str, Tensor]:
        """The wrapped head's own state dict, in the pack's key space — what a checkpoint of this should carry."""
        return self.transformer.state_dict()

    @override
    def forward(
        self,
        feat_t: Float[Tensor, "*b nt d"],
        feat_t1: Float[Tensor, "*b nu d"],
        coords_t: Float[Tensor, "*b nt 3"],
        coords_t1: Float[Tensor, "*b nu 3"],
        mask_t: Bool[Tensor, "b nt"] | None = None,
        mask_t1: Bool[Tensor, "b nu"] | None = None,
    ) -> Float[Tensor, "*b nt nu"]:
        """Edge logits for the frame pair, with the same signature and shapes the pack's head has."""
        if self.bias is None:
            return cast(Tensor, self.transformer(feat_t, feat_t1, coords_t, coords_t1, mask_t, mask_t1))
        unbatched = feat_t.ndim == _UNBATCHED_NDIM
        if unbatched:
            feat_t, feat_t1 = feat_t.unsqueeze(0), feat_t1.unsqueeze(0)
            coords_t, coords_t1 = coords_t.unsqueeze(0), coords_t1.unsqueeze(0)
        source, target = self._attend((feat_t, feat_t1), (coords_t, coords_t1), (mask_t, mask_t1))
        logits = self._pair_logits(source, target, coords_t, coords_t1)
        return logits.squeeze(0) if unbatched else logits

    def _attend(
        self,
        features: tuple[Float[Tensor, "b nt d"], Float[Tensor, "b nu d"]],
        coords: tuple[Float[Tensor, "b nt 3"], Float[Tensor, "b nu 3"]],
        masks: tuple[Bool[Tensor, "b nt"] | None, Bool[Tensor, "b nu"] | None],
    ) -> tuple[Float[Tensor, "b nt h"], Float[Tensor, "b nu h"]]:
        """The pack's bi-directional cross-attention stack, each block's scores taking the geometry bias.

        One distance matrix serves both directions: the separation of a (source, target) pair is the same
        number whichever of the two is querying, so the reverse direction takes the transpose of the same
        bias — one geometry, read twice, rather than two independently learned ones that could disagree.
        """
        bias = cast(DistanceAttentionBias, self.bias)
        mask_t, mask_t1 = masks
        project, norm_in = _part(self.transformer, "proj"), _part(self.transformer, "norm_in")
        source, target = (norm_in(project(feature)) for feature in features)
        forward_bias = bias(*coords)
        reverse_bias = forward_bias.transpose(1, 2)
        for block in cast(nn.ModuleList, self.transformer.blocks):
            source = self._block(block, source, target, mask_t1, forward_bias)
            target = self._block(block, target, source, mask_t, reverse_bias)
        norm_out = _part(self.transformer, "norm_out")
        return norm_out(source), norm_out(target)

    @staticmethod
    def _block(
        block: nn.Module,
        q: Float[Tensor, "b nq d"],
        kv: Float[Tensor, "b nkv d"],
        kv_mask: Bool[Tensor, "b nkv"] | None,
        bias: Float[Tensor, "bh nq nkv"],
    ) -> Float[Tensor, "b nq d"]:
        """One `CrossAttentionBlock`, called through its own submodules with the bias added to its scores.

        The pack's block forward takes no attention mask, so the block cannot be called as a unit; its parts
        are. Padding is folded into the same additive matrix instead of being passed separately, so there is
        one merged score offset rather than two that torch would have to reconcile.
        """
        attn_mask = RelativePositionEdgeTransformer._merge_padding(bias, kv_mask)

        norm1, attention = _part(block, "norm1"), _part(block, "cross_attn")

        def _attend(q: Tensor, kv: Tensor, attn_mask: Tensor) -> Tensor:
            normed_q, normed_kv = norm1(q), norm1(kv)
            attended, _ = attention(normed_q, normed_kv, normed_kv, attn_mask=attn_mask)
            return q + attended

        if torch.is_grad_enabled():
            q = cast(Tensor, grad_ckpt(_attend, q, kv, attn_mask, use_reentrant=False))
        else:
            q = _attend(q, kv, attn_mask)
        return q + _part(block, "mlp")(_part(block, "norm2")(q))

    @staticmethod
    def _merge_padding(
        bias: Float[Tensor, "bh nq nkv"], kv_mask: Bool[Tensor, "b nkv"] | None
    ) -> Float[Tensor, "bh nq nkv"]:
        """The geometry bias with padded keys driven to `-inf` — the pack's key-padding mask, added not passed."""
        if kv_mask is None:
            return bias
        padding = torch.where(kv_mask, 0.0, _MASKED_SCORE).to(bias.dtype)
        return bias + padding.repeat_interleave(bias.shape[0] // kv_mask.shape[0], dim=0).unsqueeze(1)

    def _pair_logits(
        self,
        source: Float[Tensor, "b nt h"],
        target: Float[Tensor, "b nu h"],
        coords_t: Float[Tensor, "b nt 3"],
        coords_t1: Float[Tensor, "b nu 3"],
    ) -> Float[Tensor, "b nt nu"]:
        """The pack's pair head over the attended nodes — its `pair_mlp`, its `rel` divisor, its chunking.

        Chunked and grad-checkpointed exactly as the original is: the full `(B, Nt, Nu, 2H+3)` tensor is tens
        of gigabytes at this problem's node counts, and the wrapper changing that would trade a memory
        property of the published head for nothing.
        """
        pair_mlp = _part(self.transformer, "pair_mlp")
        chunk = cast(int | None, self.transformer.pair_chunk_size) or source.shape[1]
        chunks: list[Tensor] = []
        for start in range(0, source.shape[1], chunk):
            piece, coords_piece = source[:, start : start + chunk, :], coords_t[:, start : start + chunk, :]

            def _score(q: Tensor, k: Tensor, cq: Tensor, ck: Tensor) -> Tensor:
                expanded_q = q.unsqueeze(2).expand(-1, -1, k.shape[1], -1)
                expanded_k = k.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
                rel = (cq.unsqueeze(2) - ck.unsqueeze(1)) / PriorVelocity.SCALE
                return cast(Tensor, pair_mlp(torch.cat([expanded_q, expanded_k, rel], dim=-1))).squeeze(-1)

            arguments = (piece, target, coords_piece, coords_t1)
            if torch.is_grad_enabled():
                chunks.append(cast(Tensor, grad_ckpt(_score, *arguments, use_reentrant=False)))
            else:
                chunks.append(_score(*arguments))
        return torch.cat(chunks, dim=1)
