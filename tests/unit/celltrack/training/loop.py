from pathlib import Path

import mlflow

from celltrack.training.dataset import AnnotatedVideo
from celltrack.training.loop import DetectionTrainer, TrainingConfig
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing

SMOKE = TrainingConfig(window=1, size=(2, 4, 4), width=4, steps=2, batch=1, seed=0, experiment="test-smoke")


def test_train(video_store: Path, in_bounds_tracks: AnnotatedTracks, tmp_path: Path):
    """The loop runs end to end — sample, forward, masked loss, step — and logs a loss per step to MLflow."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    source = AnnotatedVideo(
        video=CellVideo.from_ome_zarr(video_store),
        tracks=in_bounds_tracks,
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
    )
    run = DetectionTrainer(SMOKE).train([source])
    assert len(run.losses) == SMOKE.steps
    assert all(loss >= 0.0 for loss in run.losses)
