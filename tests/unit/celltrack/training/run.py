import sys
from pathlib import Path

import pytest

from celltrack.training import run


def test_main(dataset_root: Path, monkeypatch: pytest.MonkeyPatch):
    """The smoke CLI trains on the configured videos and leaves an MLflow store behind."""
    monkeypatch.setenv("CELLTRACK_DATA", str(dataset_root))
    monkeypatch.setattr(sys, "argv", ["run", "--videos", "2", "--steps", "1", "--device", "cpu"])
    run.main()
    assert (dataset_root / "processed" / "biohub_cell_tracking" / "mlflow.db").exists()
