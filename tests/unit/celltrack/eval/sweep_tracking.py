"""Unit tests for the sweep's mlflow record — that a grid cell lands as params + metrics.

The suite runs under `CELLTRACK_NO_MLFLOW` (autouse), so these assert the shape of what would be recorded
rather than reaching a backend: the resolved config flattens to params, and per-movie metrics carry the
stem so a sweep stays comparable months later.
"""

from typing import cast

from celltrack.eval.sweep_tracking import SweepTracking
from celltrack.operating_point import TrackerConfig
from tests.unit.celltrack.conftest import RecordingMlflow


def test_params():
    """The resolved config becomes a plain nested dict — pydantic sub-configs dumped so the tracker flattens."""
    params = SweepTracking._params(TrackerConfig(threshold=0.97))

    assert params["threshold"] == 0.97
    assert isinstance(params["linker"], dict)  # sub-config dumped, not left as a model
    assert "gate_um" in params["linker"]


def test_cell(mlflow_backend: RecordingMlflow):
    """One grid cell lands as its resolved config in params and its score in metrics — that is the record.

    A sweep whose cells are not recorded is a sweep that has to be re-run to be compared, which is the whole
    reason this exists; asserting the call is inert would assert nothing.
    """
    SweepTracking("test-proxy").cell(TrackerConfig(threshold=0.97), score=0.9334, breakdown={})

    assert ("metric", "proxy_score", 0.9334, None) in mlflow_backend.calls
    params = cast(dict[str, object], next(call[1] for call in mlflow_backend.calls if call[0] == "params"))
    assert params["threshold"] == 0.97  # the RESOLVED config, so a cell is comparable months later
    assert params["linker.gate_um"] == 10.0  # nested sub-configs flattened to dotted keys, not dropped
