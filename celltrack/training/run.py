"""CLI entrypoint for a detection-harness smoke run on real videos.

Loads a handful of training videos, samples masked crops from them, and runs the training loop — the
end-to-end check that the harness works on the actual data and logs to MLflow. MLflow is pointed at a
SQLite store under the processed data root, because the file backend is deprecated.
"""

import argparse
import logging
from pathlib import Path

import mlflow

from celltrack.training.dataset import AnnotatedVideo
from celltrack.training.loop import DetectionTrainer, TrainingConfig
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.paths import DataRoot

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"

logger = logging.getLogger(__name__)


def main() -> None:
    """Run a detection smoke over a handful of real videos, logging to MLflow."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Smoke-train the detection harness on real videos.")
    parser.add_argument("--videos", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
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
    logger.info("smoke: %d videos, %d steps on %s", len(sources), arguments.steps, arguments.device)
    run = DetectionTrainer(TrainingConfig(steps=arguments.steps, device=arguments.device)).train(sources)
    logger.info("first loss %.4f -> last loss %.4f", run.losses[0], run.losses[-1])


if __name__ == "__main__":
    main()
