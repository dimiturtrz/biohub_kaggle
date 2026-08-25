import numpy as np

from celltrack.eval.fidelity_oracle import (
    _CAPS_LIFTED,
    FidelityOracle,
    FidelityReport,
)


def test_synthetic_state():
    graph = FidelityOracle.synthetic_state()
    assert graph.node_ids.dtype == np.int64
    assert graph.edges.shape[1] == 2
    assert len(graph.node_ids) == len(graph.coordinates)


def test_divsub_report():
    oracle = FidelityOracle()
    state = FidelityOracle.synthetic_state()
    # With caps lifted identically on both sides, the only divergence is the track-start decoy the kernel's
    # C1 gate rejects and our reimplementation admits — so ours over-emits exactly that one fork.
    gates = oracle.divsub_report(state, _CAPS_LIFTED, _CAPS_LIFTED)
    ours_only, kernel_only = gates.divergent()
    assert (30, 32) in ours_only
    assert kernel_only == set()
    assert (1, 4) in gates.ours and (1, 4) in gates.kernel
    # At the shipped caps the kernel floors its fractional cap at one fork; our budget rounds to zero on a
    # small graph, so the kernel emits the true division and our side emits nothing.
    shipped = oracle.divsub_report(state)
    assert shipped.kernel == {(1, 4)}
    assert shipped.ours == set()


def test_divsub_faithful_report():
    # The SHIPPED divsub arm (kernel_faithful=True) runs the kernel's own gate bracket + C1 + internal two-cap,
    # so it emits exactly the kernel's fork {1->4} (mid-track division admitted, track-start decoy 30->32
    # rejected by C1) — a byte-faithful match where the deprecated per-flag path diverged.
    report = FidelityOracle().divsub_faithful_report(FidelityOracle.synthetic_state())
    assert report.ours == {(1, 4)}
    assert report.kernel == {(1, 4)}
    assert report.agreement() == 1.0


def test_motion_relink_report():
    report = FidelityOracle().motion_relink_report(FidelityOracle.synthetic_state())
    ours_only, kernel_only = report.divergent()
    assert report.agreement() == 1.0
    assert ours_only == set()
    assert kernel_only == set()


def test_dispatch():
    dispatch = FidelityOracle().dispatch()
    assert set(dispatch) == {"divsub", "divsub_gates", "divsub_faithful", "motion_relink"}
    assert all(callable(builder) for builder in dispatch.values())


def test_run():
    reports = FidelityOracle().run(FidelityOracle.synthetic_state(), ["divsub_faithful", "motion_relink"])
    assert [report.name for report in reports] == ["divsub_faithful", "motion_relink"]
    assert all(report.agreement() == 1.0 for report in reports)


def test_agreement():
    assert FidelityReport(name="x", ours={(1, 2)}, kernel={(1, 2)}).agreement() == 1.0
    assert FidelityReport(name="x", ours=set(), kernel=set()).agreement() == 1.0
    assert FidelityReport(name="x", ours={(1, 2)}, kernel={(3, 4)}).agreement() == 0.0


def test_divergent():
    ours_only, kernel_only = FidelityReport(name="x", ours={(1, 2)}, kernel={(3, 4)}).divergent()
    assert ours_only == {(1, 2)}
    assert kernel_only == {(3, 4)}
