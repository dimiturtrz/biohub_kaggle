from typing import cast

import pytest
from pydantic import ValidationError

from celltrack.assignment_linking import AssignmentLinker
from celltrack.division_linking import DivisionAwareLinker
from celltrack.ilp_linking import ILPLinker
from celltrack.linkers import LINKER_NAMES, LinkerConfig
from celltrack.linking import NearestNeighbourLinker
from celltrack.motion_linking import EdgeAffinity, MotionHungarianLinker
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


def test_build_wires_ilp_and_division_gates():
    """The ILP and division rows pass their radii through to the concrete linkers."""
    assert ILPLinker(spacing=SPACING, max_distance_um=10.0) == LinkerConfig(name="ilp").build(SPACING)
    assert DivisionAwareLinker(spacing=SPACING, max_distance_um=10.0, division_distance_um=8.0) == LinkerConfig(
        name="division"
    ).build(SPACING)
