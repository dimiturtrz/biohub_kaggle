"""The training loop: crops in, a detection U-Net out, masked loss and MLflow along the way.

The loop is built so the GPU never waits on the CPU. Crops are expensive to make — each decompresses whole
timepoint chunks from the zarr store — so they are sampled by a pool of DataLoader workers that prefetch the
next batches while the network trains on the current one. Host-to-device copies are pinned and non-blocking,
and the forward and loss run under autocast in bfloat16 so the RTX's tensor cores do the matmuls at half the
memory. bfloat16, not float16, on purpose: it keeps float32's exponent range, so a large activation cannot
overflow to inf and take the loss to NaN the way float16 does — a numerical fix, not a clip. A non-finite
gradient (a transient overflow or a stray NaN) still skips its optimiser step rather than corrupting the
weights. On CPU — the smoke and the unit tests — autocast disables itself and the worker pool collapses to
synchronous sampling, so the same loop runs unchanged.
"""

import logging
import time
from dataclasses import dataclass, field

import mlflow
import torch
from torch.utils.data import DataLoader

from celltrack.training.crops import CropSampler
from celltrack.training.dataset import AnnotatedVideo, CropDataset, _Item
from celltrack.training.loss import MaskedDetectionLoss
from celltrack.training.model import DetectionUNet

logger = logging.getLogger(__name__)

_PREFETCH_FACTOR = 2
# How often the loop reports throughput — often enough to see a data stall early, rarely enough not to spam.
_LOG_EVERY = 50
# Gradient-norm clip: a bigger model under fp16 autocast can let a gradient spike explode the weights into
# NaN (seen at width 64 around step 12k); clipping the global norm keeps the step bounded and stable.
_MAX_GRAD_NORM = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    """Everything a run needs, kept in one object rather than a long argument list."""

    window: int = 3
    size: tuple[int, int, int] = (16, 64, 64)
    scale_um: float = 4.0
    width: int = 16
    strides: tuple[tuple[int, int, int], ...] = ((2, 2, 2),)
    steps: int = 200
    batch: int = 2
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0
    amp: bool = True
    background_fraction: float = 0.0
    experiment: str = "detection-smoke"

    def device_type(self) -> str:
        """The autocast/scaler device family — `cuda` or `cpu`."""
        return "cuda" if self.device.startswith("cuda") else "cpu"

    def mixed_precision(self) -> bool:
        """Whether autocast and the gradient scaler are actually active (only ever on CUDA)."""
        return self.amp and self.device_type() == "cuda"


@dataclass(frozen=True)
class TrainingRun:
    """The trained model and the loss trajectory a run produced."""

    model: DetectionUNet
    losses: list[float]


@dataclass(frozen=True)
class DetectionTrainer:
    """Runs the masked-detection training loop under one configuration and objective."""

    config: TrainingConfig
    criterion: torch.nn.Module = field(default_factory=MaskedDetectionLoss)

    def train(self, sources: list[AnnotatedVideo]) -> TrainingRun:
        """Sample crops, step the U-Net under the masked objective, and log each step to MLflow."""
        loader = self._loader(sources)
        model = DetectionUNet(window=self.config.window, width=self.config.width, strides=self.config.strides).to(
            self.config.device
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scaler = torch.amp.GradScaler(self.config.device_type(), enabled=self.config.mixed_precision())

        losses: list[float] = []
        start = time.perf_counter()
        data_wait = 0.0
        with mlflow.start_run(experiment_id=self._experiment()):
            mlflow.log_params({"steps": self.config.steps, "window": self.config.window, "width": self.config.width})
            ready = time.perf_counter()
            for batch in loader:
                data_wait += time.perf_counter() - ready
                loss = self._step(model, optimiser, scaler, batch)
                losses.append(loss)
                mlflow.log_metric("loss", loss, step=len(losses))
                if len(losses) % _LOG_EVERY == 0:
                    self._log_throughput(len(losses), loss, start, data_wait)
                ready = time.perf_counter()
        return TrainingRun(model=model, losses=losses)

    def _log_throughput(self, step: int, loss: float, start: float, data_wait: float) -> None:
        """Report throughput and the data-wait share — the gauge that shows a data-starved loop as a number.

        `data_wait` is wall time the loop spent blocked on the loader (the `float(loss)` in `_step` syncs the
        step, so everything else is GPU compute). A low share means the GPU is compute-bound and the pipeline
        is not the bottleneck; a rising share means the workers stopped keeping up.
        """
        elapsed = time.perf_counter() - start
        steps_per_second = step / elapsed
        logger.info(
            "step %d/%d | loss %.4f | %.1f steps/s | %.0f crops/s | data-wait %.0f%% | %.0fs elapsed",
            step,
            self.config.steps,
            loss,
            steps_per_second,
            steps_per_second * self.config.batch,
            100.0 * data_wait / elapsed,
            elapsed,
        )

    def _loader(self, sources: list[AnnotatedVideo]) -> "DataLoader[_Item]":
        """A loader whose worker pool prefetches freshly sampled crops so the GPU never waits on the CPU."""
        sampler = CropSampler(
            window=self.config.window,
            size=self.config.size,
            scale_um=self.config.scale_um,
            background_fraction=self.config.background_fraction,
        )
        dataset = CropDataset(sources, sampler, steps=self.config.steps * self.config.batch, seed=self.config.seed)
        pin = self.config.device_type() == "cuda"
        if self.config.num_workers > 0:
            return DataLoader(
                dataset,
                batch_size=self.config.batch,
                num_workers=self.config.num_workers,
                pin_memory=pin,
                persistent_workers=True,
                prefetch_factor=_PREFETCH_FACTOR,
            )
        return DataLoader(dataset, batch_size=self.config.batch, pin_memory=pin)

    def _step(
        self,
        model: DetectionUNet,
        optimiser: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> float:
        """One optimisation step over a (frames, heatmap, mask) batch; returns the scalar masked loss."""
        frames, heatmap, mask = (tensor.to(self.config.device, non_blocking=True) for tensor in batch)
        with torch.autocast(
            device_type=self.config.device_type(),
            dtype=torch.bfloat16 if self.config.device_type() == "cuda" else torch.float32,
            enabled=self.config.mixed_precision(),
        ):
            loss: torch.Tensor = self.criterion(model(frames), heatmap, mask)
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_GRAD_NORM)
        if torch.isfinite(grad_norm):
            scaler.step(optimiser)
        scaler.update()
        return float(loss.detach())

    def _experiment(self) -> str:
        """The MLflow experiment id, created on first use."""
        existing = mlflow.get_experiment_by_name(self.config.experiment)
        return existing.experiment_id if existing else mlflow.create_experiment(self.config.experiment)
