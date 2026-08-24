import torch

from celltrack.data.confusor_audit import ConfusorAudit
from celltrack.data.gpu_scene import GpuScenes
from celltrack.data.synthetic_scene import SceneConfig


def _audit(rate: float) -> ConfusorAudit:
    base = SceneConfig(n_cells=200, volume_shape=(48, 48, 48), confusor_rate=rate)
    return ConfusorAudit(base=base, gate_um=10.0, batch=2, seed=0)


def test_hard_fraction():
    """`hard_fraction` returns (confusor rate, median own-step um) for one PairSample, on the source-vs-target
    axis: a raised `confusor_rate` batch reads far more hard than a uniform one, while the own-step median stays
    faithful (the boosted sources are a minority). Run on 'cpu' (the device is a field)."""
    generator = torch.Generator("cpu").manual_seed(0)
    off = GpuScenes(SceneConfig(n_cells=200, volume_shape=(48, 48, 48), confusor_rate=0.0), "cpu").batch(1, generator)[
        0
    ]
    generator = torch.Generator("cpu").manual_seed(0)
    on = GpuScenes(SceneConfig(n_cells=200, volume_shape=(48, 48, 48), confusor_rate=0.4), "cpu").batch(1, generator)[0]

    off_rate, off_own = ConfusorAudit.hard_fraction(off, spacing_um=1.625, gate_um=10.0)
    on_rate, on_own = ConfusorAudit.hard_fraction(on, spacing_um=1.625, gate_um=10.0)

    assert off_rate < 0.15  # uniform placement barely poses it
    assert on_rate > 0.25  # construction poses it
    assert on_own < 2.0 * off_own  # step distribution stays faithful (median not blown up by the boosted minority)


def test_sweep():
    """`sweep` returns one (rate, confusor %, own-step um) row per requested `confusor_rate`, and the measured
    confusor % rises monotonically with the configured rate — the knob is a faithful dial."""
    rows = _audit(0.0).sweep([0.0, 0.3, 0.45])

    assert [r[0] for r in rows] == [0.0, 0.3, 0.45]  # one row per requested rate, in order
    percents = [r[1] for r in rows]
    assert percents[0] < 15.0  # uniform baseline
    assert percents[1] > 25.0 and percents[2] > percents[1]  # measured rate tracks the configured rate
