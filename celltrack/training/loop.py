"""The training loop: crops in, a detection U-Net out, masked loss and MLflow along the way.

Deliberately small and explicit — it exists to prove the harness runs end to end (sample, forward, masked
loss, step, log), not to train a competitive model. The production training schedule belongs with the
detection model; what this fixes is that the mask actually gates the gradient and the run is observable.
"""

from dataclasses import dataclass

import mlflow
import torch
from torch.utils.data import DataLoader

from celltrack.training.crops import CropSampler
from celltrack.training.dataset import AnnotatedVideo, CropDataset, _Item
from celltrack.training.loss import MaskedDetectionLoss
from celltrack.training.model import DetectionUNet


@dataclass(frozen=True)
class TrainingConfig:
    """Everything a smoke run needs, kept in one object rather than a long argument list."""

    window: int = 3
    size: tuple[int, int, int] = (16, 64, 64)
    scale_um: float = 4.0
    width: int = 16
    steps: int = 200
    batch: int = 2
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    experiment: str = "detection-smoke"


@dataclass(frozen=True)
class TrainingRun:
    """The trained model and the loss trajectory a run produced."""

    model: DetectionUNet
    losses: list[float]


@dataclass(frozen=True)
class DetectionTrainer:
    """Runs the masked-detection training loop under one configuration."""

    config: TrainingConfig

    def train(self, sources: list[AnnotatedVideo]) -> TrainingRun:
        """Sample crops, step the U-Net under the masked loss, and log each step to MLflow."""
        loader = self._loader(sources)
        model = DetectionUNet(window=self.config.window, width=self.config.width).to(self.config.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = MaskedDetectionLoss()

        losses: list[float] = []
        with mlflow.start_run(experiment_id=self._experiment()):
            mlflow.log_params({"steps": self.config.steps, "window": self.config.window, "width": self.config.width})
            for batch in loader:
                loss = self._step(model, optimiser, criterion, batch)
                losses.append(loss)
                mlflow.log_metric("masked_mse", loss, step=len(losses))
        return TrainingRun(model=model, losses=losses)

    def _loader(self, sources: list[AnnotatedVideo]) -> "DataLoader[_Item]":
        """A loader that streams `steps` batches of freshly sampled crops."""
        sampler = CropSampler(window=self.config.window, size=self.config.size, scale_um=self.config.scale_um)
        dataset = CropDataset(sources, sampler, steps=self.config.steps * self.config.batch, seed=self.config.seed)
        return DataLoader(dataset, batch_size=self.config.batch)

    def _step(
        self,
        model: DetectionUNet,
        optimiser: torch.optim.Optimizer,
        criterion: MaskedDetectionLoss,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> float:
        """One optimisation step over a (frames, heatmap, mask) batch; returns the scalar masked loss."""
        frames, heatmap, mask = (tensor.to(self.config.device) for tensor in batch)
        loss = criterion(model(frames), heatmap, mask)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return float(loss.detach())

    def _experiment(self) -> str:
        """The MLflow experiment id, created on first use."""
        existing = mlflow.get_experiment_by_name(self.config.experiment)
        return existing.experiment_id if existing else mlflow.create_experiment(self.config.experiment)
