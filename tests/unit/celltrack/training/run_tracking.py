"""What a training run records — asserted against a recording mlflow backend, no real store touched."""

from pathlib import Path

import pytest

from celltrack.training.run_tracking import RunSetup, TrainingRun, TrainingSplit
from tests.unit.celltrack.conftest import RecordingMlflow

_CONFIG = {"steps": 10, "lr": 1e-4, "downsample": (1, 4, 4)}


def _open(save_to: Path, *, warm_start: bool = True, resume: bool = False) -> TrainingRun:
    setup = RunSetup(save_to, warm_start=warm_start, resume=resume, split=TrainingSplit(191, 4, 4))
    return TrainingRun.open(_CONFIG, setup)


def test_open(mlflow_backend: RecordingMlflow, tmp_path: Path):
    """Opening a run logs the config, the split and the warm-start provenance as params, once."""
    _open(tmp_path / "weights.pt")
    params = mlflow_backend.logged_params()
    assert params["steps"] == 10
    assert params["downsample"] == (1, 4, 4)
    assert params["split.train"] == 191
    assert params["warm_start"] is True
    assert params["weights"] == "weights.pt"


def test_open_tags_the_checkpoint_path_rather_than_uploading_it(mlflow_backend: RecordingMlflow, tmp_path: Path):
    """Weights are hundreds of MB — the run records WHERE they are, and never logs an artifact."""
    save_to = tmp_path / "weights.pt"
    _open(save_to)
    assert ("tag", "checkpoint", str(save_to)) in mlflow_backend.calls
    assert ("tag", "start", "warm") in mlflow_backend.calls
    assert not any(call[0] == "artifact" for call in mlflow_backend.calls)


def test_open_resumes_the_same_run(mlflow_backend: RecordingMlflow, tmp_path: Path):
    """A `--resume` continues the run id parked beside the checkpoint instead of opening a second run."""
    save_to = tmp_path / "weights.pt"
    _open(save_to)
    _open(save_to, resume=True)
    starts = [call for call in mlflow_backend.calls if call[0] == "start"]
    assert starts[0][1] == "weights-warm"  # first open: a fresh named run
    assert starts[1][2] == RecordingMlflow.RUN_ID  # second: resumed by id


def test_open_without_resume_starts_a_new_run(mlflow_backend: RecordingMlflow, tmp_path: Path):
    """A fresh run is a fresh mlflow run: the parked id of the last one is discarded, not resumed."""
    save_to = tmp_path / "weights.pt"
    _open(save_to)
    _open(save_to, warm_start=False)
    starts = [call for call in mlflow_backend.calls if call[0] == "start"]
    assert [start[1] for start in starts] == ["weights-warm", "weights-scratch"]
    assert all(start[2] is None for start in starts)


def test_window(mlflow_backend: RecordingMlflow, tmp_path: Path):
    """One eval window's numbers land at the optimiser step they were measured at."""
    _open(tmp_path / "weights.pt").window(500, {"proxy_score": 0.8, "node_recall": 0.9})
    assert ("metric", "proxy_score", 0.8, 500) in mlflow_backend.calls
    assert ("metric", "node_recall", 0.9, 500) in mlflow_backend.calls


def test_close(mlflow_backend: RecordingMlflow, tmp_path: Path):
    """Closing tags the selected checkpoint's score and ends the run."""
    _open(tmp_path / "weights.pt").close(0.7981)
    assert ("tag", "best_score", "0.7981") in mlflow_backend.calls
    assert ("end",) in mlflow_backend.calls


def test_open_is_inert_under_the_opt_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With `CELLTRACK_NO_MLFLOW` set (the suite default) every call is a no-op that must not raise."""
    monkeypatch.setenv("CELLTRACK_NO_MLFLOW", "1")
    run = TrainingRun.open(_CONFIG, RunSetup(tmp_path / "weights.pt"))
    run.window(1, {"proxy_score": 0.5})
    run.close(0.5)
    assert not (tmp_path / "weights.mlflow").exists()
