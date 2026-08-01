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

from celltrack.training.classification_loss import MaskedBCEClassificationLoss
from celltrack.training.dataset import AnnotatedVideo
from celltrack.training.loop import DetectionTrainer, TrainingConfig
from celltrack.training.loss import MaskedDetectionLoss
from celltrack.training.model import ANISOTROPIC_STRIDES
from core.data.split import AcquisitionFolds
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"
_WEIGHTS = "detector.pt"

_OBJECTIVES: dict[str, "type[MaskedDetectionLoss] | type[MaskedBCEClassificationLoss]"] = {
    "mse": MaskedDetectionLoss,
    "bce": MaskedBCEClassificationLoss,
}

logger = logging.getLogger(__name__)


def main() -> None:
    """Train the detection model over real videos, logging to MLflow and optionally saving the weights."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Train the detection model on real videos.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--videos", type=int, default=0, help="training-video cap; 0 = whole fold-train split")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--anisotropic", action="store_true")
    parser.add_argument("--objective", choices=sorted(_OBJECTIVES), default="mse")
    parser.add_argument("--weights", type=str, default=_WEIGHTS)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rechunked", action="store_true", help="read crop-sized re-chunked stores (tools/rechunk.py)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--background-fraction", type=float, default=0.0, help="share of crops from random background")
    parser.add_argument(
        "--density-stratified", action="store_true", help="draw cell-density octaves equally (balance sparse vs dense)"
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (regularization)")
    parser.add_argument("--save", action="store_true")
    arguments = parser.parse_args()

    root = DataRoot.from_config(_CONFIG)
    mlflow.set_tracking_uri(f"sqlite:///{(root.processed(_DATASET) / 'mlflow.db').as_posix()}")

    train_videos = AcquisitionFolds().split(root.videos("train"), arguments.fold).train
    limit = arguments.videos or len(train_videos)
    rechunked = root.processed(_DATASET) / "rechunked"
    sources = [
        AnnotatedVideo(
            video=(cell := CellVideo.from_ome_zarr(rechunked / video.name if arguments.rechunked else video)),
            tracks=AnnotatedTracks.from_geff(root.track_store(video)),
            spacing=cell.spacing,
        )
        for video in train_videos[:limit]
    ]
    strides = ANISOTROPIC_STRIDES if arguments.anisotropic else ((2, 2, 2),)
    config = TrainingConfig(
        steps=arguments.steps,
        device=arguments.device,
        strides=strides,
        width=arguments.width,
        num_workers=arguments.workers,
        batch=arguments.batch,
        amp=not arguments.no_amp,
        background_fraction=arguments.background_fraction,
        density_stratified=arguments.density_stratified,
        weight_decay=arguments.weight_decay,
    )
    criterion: torch.nn.Module = (
        MaskedBCEClassificationLoss(label_smoothing=arguments.label_smoothing)
        if arguments.objective == "bce"
        else MaskedDetectionLoss()
    )
    logger.info(
        "train: %d videos, %d steps on %s, strides=%s, objective=%s",
        len(sources),
        arguments.steps,
        config.device,
        strides,
        arguments.objective,
    )

    run = DetectionTrainer(config, criterion=criterion).train(sources)
    logger.info("first loss %.4f -> last loss %.4f", run.losses[0], run.losses[-1])
    if arguments.save:
        destination = root.processed(_DATASET) / arguments.weights
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
