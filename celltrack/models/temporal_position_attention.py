"""Temporal positional encoding for the backbone's per-voxel time attention — the motion sight the mount lacks.

`tracking_cellmot._TemporalAttention` attends each voxel across the `T` frames of a window with NO positional
signal, so the attention is permutation-invariant over time: `attn([t, t+1])` and `attn([t+1, t])` produce the
same mixture up to swapping. The backbone therefore cannot tell which frame came first, and every motion
quantity — direction, velocity, "this cell was already moving" — is invisible to it. Order enters the pipeline
only through the edge transformer's scalar time-fraction node feature, which is far too weak to carry a
trajectory. This is the one capability mounting the published weights cannot buy: the published backbone was
trained order-blind, so no configuration recovers it.

`TemporalPositionAttention` wraps one existing attention block, reusing its TRAINED `norm` and `attn` weights,
and adds a learned per-frame-index embedding to the tokens after the LayerNorm and before the attention — so a
voxel's token now says "I am frame k", and the attention can represent a direction. The embedding is a table
indexed by absolute frame position within the window, capped at `max_positions`; a window longer than the table
is refused rather than silently wrapped, because a wrong position is worse than none.

ZERO INIT is deliberate and load-bearing: the table starts at zeros, so at step 0 the wrapped block is
BYTE-IDENTICAL to the block it replaced and a warm start from the published weights is unchanged. The model
must then LEARN to use position — which is why this cannot be evaluated in the short warm-start fine-tune regime
(a converged model peaks before it learns a new input); it belongs to a long run where the table has time to
grow away from zero.
"""

from __future__ import annotations

import math
from typing import cast, override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class TemporalPositionAttention(nn.Module):
    """One `_TemporalAttention` block plus a learned frame-position embedding — order-aware, warm-start-safe.

    Holds the ORIGINAL block's `norm` and `attn` submodules (their trained weights are the whole point of a
    warm start), and re-runs its exact forward with one addition: a per-frame-index embedding added to the
    normalised tokens before self-attention. The residual, the reshape and the attention are otherwise identical,
    so at a zero-initialised embedding this module reproduces the wrapped block to the bit.
    """

    def __init__(self, block: nn.Module, max_positions: int = 8) -> None:
        super().__init__()
        self.norm = cast(nn.LayerNorm, block.norm)  # the wrapped block's trained LayerNorm, reused
        self.attn = cast(nn.MultiheadAttention, block.attn)  # the wrapped block's trained MultiheadAttention, reused
        channels = self.norm.normalized_shape[0]
        # Zero so the wrapped block is byte-identical at init: a warm start pays nothing until position is learned.
        # Matched to the reused weights' device/dtype so the block can be wrapped after the model is on-device.
        weight = self.norm.weight
        self.position = nn.Parameter(torch.zeros(max_positions, channels, device=weight.device, dtype=weight.dtype))

    @override
    def forward(self, x: Float[Tensor, "b t c z y x"]) -> Float[Tensor, "b t c z y x"]:
        """`x`: `(B, T, C, Z, Y, X)`. Attend across `T` with a frame-index embedding added before attention."""
        batch, frames, channels = x.shape[:3]
        spatial = x.shape[3:]
        voxels = math.prod(spatial)
        if frames > self.position.shape[0]:
            message = (
                f"window has {frames} frames but the position table holds {self.position.shape[0]}; "
                f"raise max_positions or shorten the window (a wrapped position is worse than none)"
            )
            raise ValueError(message)

        tokens = x.reshape(batch, frames, channels, voxels).permute(0, 3, 1, 2)
        tokens = tokens.reshape(batch * voxels, frames, channels)
        tokens = self.norm(tokens) + self.position[:frames]
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attended = (
            attended.reshape(batch, voxels, frames, channels)
            .permute(0, 2, 3, 1)
            .reshape(batch, frames, channels, *spatial)
        )
        return x + attended
