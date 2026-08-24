"""The warm/chained base-calibration gate: pass a calibrated base, abort a miscalibrated one, exempt from-scratch."""

import pytest

from celltrack.training.base_calibration import BaseCalibration


def test_gate():
    """A calibrated base sits near a zero node ratio — a warm start onto it is a no-op through the gate."""
    assert BaseCalibration.gate(0.1, expects_calibrated_base=True) is None
    assert BaseCalibration.gate(-0.4, expects_calibrated_base=True) is None


def test_gate_aborts_a_warm_start_onto_a_miscalibrated_base():
    """A warm/chained base that over- or under-detects past +-0.5 is barred, not trained onto and read as null."""
    with pytest.raises(ValueError, match="miscalibrated"):
        BaseCalibration.gate(2.0, expects_calibrated_base=True)
    with pytest.raises(ValueError, match="miscalibrated"):
        BaseCalibration.gate(-1.31, expects_calibrated_base=True)


def test_gate_exempts_a_from_scratch_run():
    """A from-scratch run legitimately starts uncalibrated — the gate never fires on it, at any ratio."""
    BaseCalibration.gate(2.0, expects_calibrated_base=False)


def test_gate_boundary_passes_at_the_threshold():
    """The threshold itself passes; a warm base is barred only when it strictly exceeds +-0.5."""
    BaseCalibration.gate(0.5, expects_calibrated_base=True)
    BaseCalibration.gate(-0.5, expects_calibrated_base=True)
    with pytest.raises(ValueError, match="miscalibrated"):
        BaseCalibration.gate(0.51, expects_calibrated_base=True)
