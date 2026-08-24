"""Unit tests for the SIGReg pretraining — the config both halves agree on and the loop's checkpoint write.

The default config pins the encoder width the checkpoint saves and the gate later rebuilds. The loop itself is
run on CPU over an injected fake corpus (the trainer takes its corpus, so no zarr is opened): one step must drive
the objective and leave a loadable checkpoint carrying the encoder config beside the weights. A seed is fixed so
the augmentation and SIGReg direction draw are deterministic.
"""

from pathlib import Path

import numpy as np
import torch

from celltrack.models.patch_encoder import PatchEncoderConfig
from celltrack.training.patch_ssl import PatchSslConfig, PatchSslTrainer


class _FakeCorpus:
    """A corpus stand-in that hands out a fixed-shape random box batch — no video, no zarr."""

    def sample(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        return rng.standard_normal((batch_size, 5, 17, 17)).astype(np.float32)


def test_config_defaults():
    """The default schedule carries an encoder and objective config, and the SIGReg weight is the one knob."""
    config = PatchSslConfig()

    assert config.encoder.embedding_dim == 64  # the width the gate rebuilds from the checkpoint
    assert config.sigreg.weight == 1.0  # the single hyperparameter
    assert config.steps_per_epoch > 0


def test_train(tmp_path: Path):
    """One SIGReg step runs on CPU and writes a checkpoint the gate can reload — config saved beside the weights."""
    torch.manual_seed(0)
    config = PatchSslConfig(
        encoder=PatchEncoderConfig(embedding_dim=16, channels=(8, 16)), batch_size=4, epochs=1, steps_per_epoch=1
    )
    trainer = PatchSslTrainer(config, _FakeCorpus(), device="cpu")  # type: ignore[arg-type]
    checkpoint = tmp_path / "encoder.pt"

    trainer.train(checkpoint)

    state = torch.load(checkpoint, weights_only=False)
    assert state["config"]["embedding_dim"] == 16  # the width the gate rebuilds
    assert "encoder" in state  # the weights are saved beside their shape
