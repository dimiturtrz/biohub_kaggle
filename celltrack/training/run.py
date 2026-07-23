"""CLI entrypoint for training the detection model on real videos.

Loads training videos, samples masked crops, and runs the loop — logging to MLflow and optionally saving
the trained weights for the recall evaluation and the inference kernel. MLflow is pointed at a SQLite
store under the processed data root, because the file backend is deprecated. `--anisotropic` selects the
deep anisotropy-aware detector; without it the shallow isotropic harness model trains (used by the smoke).
"""

import argparse
import logging
from pathlib import Path

import mlflow
import torch

from celltrack.training.dataset import AnnotatedVideo
from celltrack.training.loop import DetectionTrainer, TrainingConfig
from celltrack.training.model import ANISOTROPIC_STRIDES
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
_WEIGHTS = "detector.pt"

logger = logging.getLogger(__name__)


def main() -> None:
    """Train the detection model over real videos, logging to MLflow and optionally saving the weights."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Train the detection model on real videos.")
    parser.add_argument("--videos", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--anisotropic", action="store_true")
    parser.add_argument("--save", action="store_true")
    arguments = parser.parse_args()

    root = DataRoot.from_config(_CONFIG)
    mlflow.set_tracking_uri(f"sqlite:///{(root.processed(_DATASET) / 'mlflow.db').as_posix()}")

    sources = [
        AnnotatedVideo(
            video=(cell := CellVideo.from_ome_zarr(video)),
            tracks=AnnotatedTracks.from_geff(root.track_store(video)),
            spacing=cell.spacing,
        )
        for video in root.videos("train")[: arguments.videos]
    ]
    strides = ANISOTROPIC_STRIDES if arguments.anisotropic else ((2, 2, 2),)
    config = TrainingConfig(steps=arguments.steps, device=arguments.device, strides=strides)
    logger.info("train: %d videos, %d steps on %s, strides=%s", len(sources), arguments.steps, config.device, strides)

    run = DetectionTrainer(config).train(sources)
    logger.info("first loss %.4f -> last loss %.4f", run.losses[0], run.losses[-1])
    if arguments.save:
        destination = root.processed(_DATASET) / _WEIGHTS
        torch.save(
            {
                "state_dict": run.model.state_dict(),
                "strides": strides,
                "window": config.window,
                "width": config.width,
            },
            destination,
        )
        logger.info("saved weights to %s", destination)


if __name__ == "__main__":
    main()
