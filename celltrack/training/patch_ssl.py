"""Pretrain the patch encoder with SIGReg — the card-gated run that produces the checkpoint the gate scores.

The loop is the whole of self-supervised identity: draw a batch of cell boxes, make two nuisance views of each,
encode both, and step on SIGReg (views alike, batch isotropic-Gaussian). No labels beyond the box centring, no
negatives, no teacher — the single hyperparameter is the SIGReg weight, carried in its config. The checkpoint
saves the encoder beside its shape so `appearance_encoder_gate` can rebuild and score it without a second
declaration of the width.

Success is read in the log, not assumed: invariance must fall while Gaussianity holds. Invariance falling with
Gaussianity climbing is representational collapse — the failure SIGReg exists to forbid — and seeing it here is
cheaper than seeing it as a dead gate.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from celltrack.data.patch_dataset import NuisanceViews, PatchCorpus
from celltrack.losses.sigreg import SigReg, SigRegConfig
from celltrack.models.patch_encoder import PatchEncoder, PatchEncoderConfig
from core.paths import DataRoot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchSslConfig:
    """Everything the pretraining run needs — the encoder shape, the objective, and the schedule."""

    encoder: PatchEncoderConfig = field(default_factory=PatchEncoderConfig)
    sigreg: SigRegConfig = field(default_factory=SigRegConfig)
    radius_um: float = 3.25
    batch_size: int = 128
    epochs: int = 20
    steps_per_epoch: int = 200
    learning_rate: float = 1e-3
    seed: int = 0
    split: str = "train"


class PatchSslTrainer:
    """Owns the encoder, the objective and the corpus for the length of a pretraining run."""

    def __init__(self, config: PatchSslConfig, corpus: PatchCorpus, device: str) -> None:
        self.config = config
        self.device = device
        self.encoder = PatchEncoder(config.encoder).to(device)
        self.sigreg = SigReg(config.sigreg).to(device)
        self.corpus = corpus
        self.rng = np.random.default_rng(config.seed)
        self.generator = torch.Generator(device=device).manual_seed(config.seed)

    def train(self, checkpoint: Path) -> None:
        """Run the SIGReg loop and write the encoder beside its config."""
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.config.learning_rate)
        self.encoder.train()
        for epoch in range(self.config.epochs):
            invariance, gaussianity = self._epoch(optimizer)
            logger.info("epoch %2d  invariance %.4f  gaussianity %.4f", epoch, invariance, gaussianity)
        self._save(checkpoint)

    def _epoch(self, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
        """One epoch of steps; returns the epoch-mean invariance and Gaussianity for the log."""
        invariance = 0.0
        gaussianity = 0.0
        for _ in range(self.config.steps_per_epoch):
            result = self._step(optimizer)
            invariance += result[0]
            gaussianity += result[1]
        return invariance / self.config.steps_per_epoch, gaussianity / self.config.steps_per_epoch

    def _step(self, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
        """Sample, embed two views, step on SIGReg; a fully-dropped batch is skipped as a no-op."""
        boxes = self.corpus.sample(self.rng, self.config.batch_size)
        if not len(boxes):
            return 0.0, 0.0
        tensor = torch.from_numpy(boxes.astype(np.float32)).to(self.device)
        view_one, view_two = NuisanceViews.two(tensor, self.generator)
        result = self.sigreg(self.encoder(view_one), self.encoder(view_two))
        optimizer.zero_grad()
        result.total.backward()
        optimizer.step()
        return float(result.invariance.detach()), float(result.gaussianity.detach())

    def _save(self, checkpoint: Path) -> None:
        """Write the encoder weights beside the config that rebuilds their shape."""
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config.encoder), "encoder": self.encoder.state_dict()}, checkpoint)
        logger.info("wrote encoder -> %s", checkpoint)


def main() -> None:
    """Pretrain the patch encoder self-supervised and write its checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="SIGReg self-supervised patch-encoder pretraining.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=PatchSslConfig.epochs)
    parser.add_argument("--sigreg-weight", type=float, default=SigRegConfig.weight)
    args = parser.parse_args()

    config = PatchSslConfig(sigreg=SigRegConfig(weight=args.sigreg_weight), epochs=args.epochs)
    root = DataRoot.from_config(args.config)
    corpus = PatchCorpus.build(root, config.split, config.radius_um)
    trainer = PatchSslTrainer(config, corpus, args.device)
    trainer.train(args.checkpoint)


if __name__ == "__main__":
    main()
