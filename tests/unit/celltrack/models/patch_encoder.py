"""Unit tests for the raw-box appearance encoder — shape contract and ragged-box tolerance.

The encoder's claims are structural: it maps a `(b, c, z, y, x)` box to a `(b, d)` embedding, accepts the odd,
per-axis box sizes `RawBoxes` hands it without a fixed input shape, and emits an UN-normalised vector in R^d (so
`SIGReg` can shape its distribution). The fixtures are small seeded random tensors, since only the shape and the
absence of normalisation are under test — content is the objective's job, not the module's.
"""

import pytest
import torch

from celltrack.models.patch_encoder import PatchEncoder, PatchEncoderConfig


def test_post_init():
    """An empty channel schedule is a mis-specified encoder and must fail at construction, not at forward."""
    with pytest.raises(ValueError, match="at least one convolution stage"):
        PatchEncoderConfig(channels=())


def test_embedding_dim():
    """The encoder reports the embedding width the objective and edge head must agree on."""
    encoder = PatchEncoder(PatchEncoderConfig(embedding_dim=48))
    assert encoder.embedding_dim == 48


def test_forward():
    """A batch of boxes maps to one embedding per box at the configured width, in R^d (not on the unit sphere)."""
    torch.manual_seed(0)
    encoder = PatchEncoder(PatchEncoderConfig(embedding_dim=32))
    boxes = torch.randn(4, 1, 5, 17, 17)

    embedding = encoder.forward(boxes)

    assert embedding.shape == (4, 32)
    assert not torch.allclose(embedding.norm(dim=-1), torch.ones(4), atol=1e-3)  # un-normalised: norms are not all 1


def test_forward_ragged():
    """The global pool accepts a different odd, per-axis box size without a fixed input shape."""
    torch.manual_seed(0)
    encoder = PatchEncoder(PatchEncoderConfig(embedding_dim=32))

    assert encoder.forward(torch.randn(2, 1, 7, 21, 21)).shape == (2, 32)  # larger, still odd per axis
