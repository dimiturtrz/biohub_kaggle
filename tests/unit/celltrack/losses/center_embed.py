"""Unit test for the CELLECT center-point contrastive term over the dedicated embedding.

The term is the sum of two reused primitives (`InfoNCE`, `HardNegativeMargin`) on ONE 64-ch embedding, so
these tests cover the assembly's own contract: the defined zero on an unannotated pair, that aligning the
true pair below its decoy scores lower than inverting it, and that a gradient reaches the embedding.
"""

import torch

from celltrack.losses.center_embed import CenterEmbedContrastive

# source 0 -> target 0, source 1 -> target 1; target 2 is the shared confusor neither source parents.
_LINKS = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_DECOYS = torch.tensor([[0, 2], [1, 2]])  # each source's nearest wrong target is the confusor
_TEMPERATURE = 0.1


def _basis(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows)


def test_of():
    """A pair whose annotation links nothing yields a defined zero from both primitives, never a NaN."""
    embed = _basis([[1.0, 0.0], [0.0, 1.0]])
    value = CenterEmbedContrastive.of(
        embed, embed, torch.zeros(2, 2), torch.zeros(0, 2, dtype=torch.int64), _TEMPERATURE
    )
    assert value.item() == 0.0


def test_of_falls_as_the_true_pair_out_ranks_the_confusor():
    """Aligning each source's embedding with its true successor (not the confusor) scores below inverting it."""
    source = _basis([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    aligned_targets = _basis([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # true pair matches, confusor apart
    inverted_targets = _basis([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])  # true apart, confusor aligned

    aligned = CenterEmbedContrastive.of(source, aligned_targets, _LINKS, _DECOYS, _TEMPERATURE)
    inverted = CenterEmbedContrastive.of(source, inverted_targets, _LINKS, _DECOYS, _TEMPERATURE)

    assert inverted.item() > aligned.item() > 0.0


def test_of_backpropagates_to_the_embedding():
    """The term is differentiable into the embedding — the head can only be trained if a gradient reaches it."""
    source = _basis([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).requires_grad_(True)
    targets = _basis([[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.4, 0.4, 0.2]])

    value = CenterEmbedContrastive.of(source, targets, _LINKS, _DECOYS, _TEMPERATURE)
    value.backward()

    assert source.grad is not None and source.grad.abs().sum() > 0
