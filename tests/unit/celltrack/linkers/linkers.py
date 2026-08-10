from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import AssignmentLinker
from celltrack.linkers.division_linking import DivisionAwareLinker
from celltrack.linkers.ilp_linking import ILPLinker
from celltrack.linkers.linkers import LINKER_NAMES, LinkerConfig
from celltrack.linkers.linking import NearestNeighbourLinker
from celltrack.linkers.motion_linking import MotionHungarianLinker
from core.geometry import Spacing

SPACING = Spacing(z=2.0, y=1.0, x=1.0)


def test_build():
    """Each name resolves to its concrete linker, carrying the gate radii the config supplied."""
    nn = LinkerConfig(name="nn", gate_um=9.0)
    motion = LinkerConfig(name="motion", gate_um=9.0, tight_um=5.0)

    assert NearestNeighbourLinker(spacing=SPACING, max_distance_um=9.0) == nn.build(SPACING)
    assert MotionHungarianLinker(spacing=SPACING, tight_gate_um=5.0, loose_gate_um=9.0) == motion.build(SPACING)


def test_build_names_the_whole_family():
    """Every registered name builds a linker satisfying the contract (a `link` method)."""
    for name in LINKER_NAMES:
        linker = LinkerConfig(name=name).build(SPACING)
        assert callable(linker.link)


def test_build_rejects_an_unknown_name():
    """An unregistered name is rejected at construction (validated boundary), not deep inside build."""
    with pytest.raises(ValidationError):
        LinkerConfig(name="nope")


def test_config_rejects_a_non_positive_radius():
    """Radii cross the same trust boundary — a zero/negative gate is caught at construction."""
    with pytest.raises(ValidationError):
        LinkerConfig(name="nn", gate_um=0.0)


def test_build_wires_the_assignment_linker_and_passes_affinity():
    """The default `assignment` row builds AssignmentLinker with its bonus/disappearance and the given affinity."""
    affinity = cast(EdgeAffinity, object())
    linker = LinkerConfig(name="assignment", gate_um=9.0, affinity_bonus=18.0, disappearance_cost=2.0)
    built = linker.build(SPACING, affinity)

    assert isinstance(built, AssignmentLinker)
    assert built.affinity is affinity  # the per-video learned affinity is threaded through
    assert (built.max_distance_um, built.affinity_bonus, built.disappearance_cost) == (9.0, 18.0, 2.0)


def test_effective_bonus():
    """Unset, the bonus derives as 2·gate (so P-weighting holds when the gate moves); a pinned value wins."""
    assert LinkerConfig(gate_um=14.0).effective_bonus == 28.0
    assert LinkerConfig(gate_um=10.0).effective_bonus == 20.0  # the shipped default, unchanged
    assert LinkerConfig(gate_um=14.0, affinity_bonus=20.0).effective_bonus == 20.0
    assert LinkerConfig(name="assignment", gate_um=14.0).build(SPACING).affinity_bonus == 28.0


def test_build_wires_ilp_and_division_gates():
    """The ILP and division rows pass their radii through to the concrete linkers."""
    assert ILPLinker(spacing=SPACING, max_distance_um=10.0) == LinkerConfig(name="ilp").build(SPACING)
    assert DivisionAwareLinker(spacing=SPACING, max_distance_um=10.0, division_distance_um=8.0) == LinkerConfig(
        name="division"
    ).build(SPACING)


def test_bonus_is_readable():
    """A bonus on a linker that cannot read an affinity is refused — the knob could not have moved anything."""
    with pytest.raises(ValidationError, match="does not read an edge affinity"):
        LinkerConfig(name="motion", affinity_bonus=20.0)

    assert LinkerConfig(name="assignment", affinity_bonus=20.0).effective_bonus == 20.0  # readers still accept it


class _Mutual:
    """A stand-in for the bidirectionally fused affinity the agreement floor gates on."""

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return np.zeros((1, 1))


def test_agreement_floor_is_readable():
    """An agreement floor on a linker that cannot read an affinity is refused, exactly as a bonus is."""
    with pytest.raises(ValidationError, match="does not read an edge affinity"):
        LinkerConfig(name="motion", agreement_floor=0.5)

    assert LinkerConfig(name="flow", agreement_floor=0.5).agreement_floor == 0.5  # readers still accept it


def test_build_wires_the_agreement_gate_into_both_affinity_readers():
    """A floor plus the fused affinity builds the same `AgreementGate` onto the assignment and flow linkers."""
    mutual = cast(EdgeAffinity, _Mutual())
    for name in ("assignment", "flow"):
        built = LinkerConfig(name=name, agreement_floor=0.4).build(SPACING, cast(EdgeAffinity, object()), mutual)
        assert built.agreement == AgreementGate(mutual=mutual, floor=0.4)


def test_build_without_the_fused_affinity_refuses_a_floor():
    """A floor the pipeline could not honour is an error, not a silently dropped instruction."""
    with pytest.raises(ValueError, match="needs the bidirectionally fused affinity"):
        LinkerConfig(name="assignment", agreement_floor=0.4).build(SPACING, cast(EdgeAffinity, object()))


def test_build_leaves_the_gate_off_by_default():
    """No floor means no gate, even when a fused affinity is available — nothing is enabled implicitly."""
    built = LinkerConfig(name="flow").build(SPACING, cast(EdgeAffinity, object()), cast(EdgeAffinity, _Mutual()))
    assert built.agreement is None
