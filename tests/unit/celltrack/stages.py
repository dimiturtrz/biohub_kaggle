import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from celltrack.center_prior import CenterPriorRecipe, CenterPriorVeto, CenterVetoConfig
from celltrack.division_recovery import DivisionRecovery
from celltrack.gap_closer import GapCloser, SyntheticGap
from celltrack.linefit_smoother import LinefitSmoother
from celltrack.linking import NearestNeighbourLinker
from celltrack.short_track_filter import ShortTrackFilter
from celltrack.stages import (
    DivisionSpec,
    GapSpec,
    LinkerSpec,
    LinkerStage,
    ShortTrackSpec,
    SmoothSpec,
    StageSpec,
    TopologySpec,
)
from celltrack.topology_repair import TopologyRepair
from core.data.tracks import TrackGraph
from core.geometry import Spacing

SPACING = Spacing(z=2.0, y=1.0, x=1.0)


def test_linker_spec_build():
    """The linker slot delegates to the linker family and adapts it to the stage contract."""
    stage = LinkerSpec().build(SPACING)
    assert isinstance(stage, LinkerStage)
    assert stage.linker == NearestNeighbourLinker(spacing=SPACING, max_distance_um=10.0)


def test_gap_spec_build():
    assert GapSpec().build(SPACING) == GapCloser(SPACING, 5.8, 3.2, 0.05)


def test_division_spec_build():
    assert DivisionSpec().build(SPACING) == DivisionRecovery(SPACING, 4.7, 7.2, 0.004, 7.8, 0.008)


def _veto() -> CenterPriorVeto:
    """A centre-prior veto over a single trivial heatmap, enough to resolve a spec's gate to a confirmer."""
    return CenterPriorVeto(heatmaps=(np.ones((1, 1, 1), dtype=np.float32),), recipe=CenterPriorRecipe())


def test_division_spec_build_binds_the_veto_when_configured():
    """A division spec carrying a centre-veto config resolves to a confirmer once a video's heatmaps are in hand."""
    spec = DivisionSpec(center_veto=CenterVetoConfig(threshold=0.2))
    assert spec.build(SPACING, _veto()).confirmer is not None  # gate + heatmaps -> a bound confirmer
    assert spec.build(SPACING).confirmer is None  # a configured gate but no heatmaps stays veto-free
    assert DivisionSpec().build(SPACING, _veto()).confirmer is None  # no gate -> no veto even with heatmaps


def test_gap_spec_build_binds_the_veto_when_configured():
    """A gap spec carrying a centre-veto config resolves to a confirmer once a video's heatmaps are in hand."""
    assert GapSpec(center_veto=CenterVetoConfig()).build(SPACING, _veto()).confirmer is not None
    assert GapSpec(center_veto=CenterVetoConfig()).build(SPACING).confirmer is None


def test_gap_spec_build_carries_the_synthetic_policy():
    """A gap spec's synthetic-insertion policy is carried onto the built gap closer."""
    policy = SyntheticGap(min_span_um=8.5, max_added_fraction=0.05)
    assert GapSpec(synthetic=policy).build(SPACING, _veto()).synthetic == policy
    assert GapSpec().build(SPACING).synthetic is None


def test_short_track_spec_build():
    assert ShortTrackSpec().build(SPACING) == ShortTrackFilter(6)


def test_smooth_spec_build():
    assert SmoothSpec(strength=0.5).build(SPACING) == LinefitSmoother(0.5)


def test_topology_spec_build():
    assert TopologySpec().build(SPACING) == TopologyRepair(SPACING, 14.0)


def test_transform():
    """The linker adapter runs the wrapped linker, wiring two consecutive-frame nodes into one edge."""
    coordinates = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int64)
    nodes = TrackGraph(np.array([0, 1], dtype=np.int64), coordinates, np.empty((0, 2), dtype=np.int64))
    linked = LinkerStage(NearestNeighbourLinker(spacing=SPACING, max_distance_um=10.0)).transform(nodes)
    assert linked.edges.shape == (1, 2)


def test_specs_reject_out_of_bounds_gates_at_construction():
    """Every gate crosses the config trust boundary — a bad value is caught before a run."""
    with pytest.raises(ValidationError):
        GapSpec(gate_um=0.0)
    with pytest.raises(ValidationError):
        ShortTrackSpec(min_length=0)
    with pytest.raises(ValidationError):
        SmoothSpec(strength=1.5)


def test_stage_spec_union_dispatches_on_kind():
    """A serialised stage parses to its variant by the `kind` discriminator — how a config.json is loaded."""
    adapter = TypeAdapter(StageSpec)
    assert isinstance(adapter.validate_python({"kind": "gap", "gate_um": 6.0}), GapSpec)
    assert isinstance(adapter.validate_python({"kind": "division"}), DivisionSpec)


def test_stage_spec_union_rejects_an_unknown_kind():
    with pytest.raises(ValidationError):
        TypeAdapter(StageSpec).validate_python({"kind": "nope"})
