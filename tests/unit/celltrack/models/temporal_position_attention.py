import math
from typing import override

import pytest
import torch
from torch import nn

from celltrack.models.temporal_position_attention import TemporalPositionAttention


class _Block(nn.Module):
    """A minimal stand-in for the external `_TemporalAttention`: the `norm` + `attn` the wrapper reuses."""

    def __init__(self, channels: int, heads: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, channels = x.shape[:3]
        spatial = x.shape[3:]
        voxels = math.prod(spatial)
        tokens = x.reshape(batch, frames, channels, voxels).permute(0, 3, 1, 2)
        tokens = tokens.reshape(batch * voxels, frames, channels)
        tokens = self.norm(tokens)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attended = (
            attended.reshape(batch, voxels, frames, channels)
            .permute(0, 2, 3, 1)
            .reshape(batch, frames, channels, *spatial)
        )
        return x + attended


def _window(channels: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 2, channels, 4, 4, 4)


def test_forward():
    """The load-bearing property: at a zero-initialised table the wrapper's forward equals the block it replaced.

    This is what makes a warm start from the published weights free — the model is byte-identical until it
    LEARNS to use position, so wrapping the attention costs nothing at step 0.
    """
    channels = 8
    block = _Block(channels).eval()
    wrapped = TemporalPositionAttention(block, max_positions=8).eval()
    x = _window(channels)

    with torch.no_grad():
        assert torch.equal(block.forward(x), wrapped.forward(x))


def test_a_nonzero_position_changes_the_output():
    """Once the table is non-zero the attention sees which frame is which — the whole point of the module."""
    channels = 8
    block = _Block(channels).eval()
    wrapped = TemporalPositionAttention(block, max_positions=8).eval()
    x = _window(channels)

    with torch.no_grad():
        baseline = wrapped(x).clone()
        wrapped.position.add_(1.0)  # frames now carry distinct positional signal
        assert not torch.equal(wrapped(x), baseline)


def test_reuses_the_wrapped_blocks_trained_weights():
    """The wrapper shares the block's `norm`/`attn` modules — a warm start's trained weights are the point."""
    block = _Block(8)
    wrapped = TemporalPositionAttention(block, max_positions=8)

    assert wrapped.norm is block.norm
    assert wrapped.attn is block.attn


def test_a_window_longer_than_the_table_is_refused():
    """A wrapped position is worse than none, so a window past the table raises rather than indexing wrong."""
    wrapped = TemporalPositionAttention(_Block(8), max_positions=2).eval()
    too_long = torch.randn(1, 3, 8, 4, 4, 4)  # 3 frames, table holds 2

    with pytest.raises(ValueError, match="position table holds 2"):
        wrapped(too_long)
