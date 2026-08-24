"""Unit tests for the learned-encoder confusor gate — the scorer wiring and the checkpoint round-trip.

The report opens the dense movie and its annotation, so its coverage is the gate run on the card, not a unit test.
What is unit-testable, CPU-only: `EncoderGate.scorer` returns one cosine per candidate and normalises at read time,
and `EncoderGate.load` rebuilds the exact encoder a checkpoint saved beside its config. The fixture is a tiny
encoder over random boxes — only the scorer contract and the save/load identity are under test.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from celltrack.analysis.appearance_encoder_gate import EncoderGate
from celltrack.models.patch_encoder import PatchEncoder, PatchEncoderConfig


def test_scorer():
    """The scorer embeds source and candidates and returns one cosine each; a box scores highest against itself."""
    torch.manual_seed(0)
    encoder = PatchEncoder(PatchEncoderConfig(embedding_dim=16)).eval()
    scorer = EncoderGate.scorer(encoder, "cpu")
    source = np.random.default_rng(0).standard_normal((5, 17, 17)).astype(np.float32)
    candidates = np.stack([source, np.random.default_rng(1).standard_normal((5, 17, 17)).astype(np.float32)])

    scores = scorer(source, candidates)

    assert scores.shape == (2,)
    assert scores[0] == max(scores)  # the identical box is the encoder's nearest candidate
    assert scores[0] <= 1.0 + 1e-5  # a cosine of unit vectors


def test_load(tmp_path: Path):
    """A saved checkpoint rebuilds an encoder with the same config and weights — no second width declaration."""
    config = PatchEncoderConfig(embedding_dim=16, channels=(8, 16))
    encoder = PatchEncoder(config)
    checkpoint = tmp_path / "enc.pt"
    torch.save({"config": asdict(config), "encoder": encoder.state_dict()}, checkpoint)

    loaded = EncoderGate.load(checkpoint, "cpu")

    boxes = torch.randn(2, 1, 5, 17, 17)
    with torch.no_grad():
        assert torch.allclose(loaded(boxes), encoder(boxes), atol=1e-6)  # identical shape and weights
