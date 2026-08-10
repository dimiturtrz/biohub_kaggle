"""Unit tests for the sweep's mlflow record — that a grid cell lands as params + metrics.

The suite runs under `CELLTRACK_NO_MLFLOW` (autouse), so these assert the shape of what would be recorded
rather than reaching a backend: the resolved config flattens to params, and per-movie metrics carry the
stem so a sweep stays comparable months later.
"""

from celltrack.eval.sweep_tracking import SweepTracking
from celltrack.tracker import TrackerConfig


def test_params():
    """The resolved config becomes a plain nested dict — pydantic sub-configs dumped so the tracker flattens."""
    params = SweepTracking._params(TrackerConfig(threshold=0.97))

    assert params["threshold"] == 0.97
    assert isinstance(params["linker"], dict)  # sub-config dumped, not left as a model
    assert "gate_um" in params["linker"]


def test_cell(caplog):
    """Recording a cell is inert under the opt-out and never raises — tracking must not break a sweep."""
    SweepTracking("test-proxy").cell(TrackerConfig(threshold=0.97), score=0.9334, breakdown={})
