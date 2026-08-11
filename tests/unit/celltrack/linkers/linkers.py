from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import AssignmentLinker
from celltrack.linkers.boundary_prior import BoundaryPrior
from celltrack.linkers.division_linking import DivisionAwareLinker
from celltrack.linkers.ilp_linking import ILPLinker
from celltrack.linkers.linkers import LINKER_NAMES, LinkerConfig
from celltrack.linkers.linking import NearestNeighbourLinker
from celltrack.linkers.motion_linking import MotionHungarianLinker
from celltrack.linkers.mutual_bonus import MutualBonus
from celltrack.linkers.ranker_bonus import RankerBonus
from core.data.tracks import TrackGraph
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


class _Scored:
    """A stand-in learned quantity for one two-by-two gap — a real matrix, so a cost term can move the links."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._matrix


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


def test_mutual_bonus_is_readable():
    """A mutual bonus on a linker that cannot read an affinity is refused, exactly as a bonus and a floor are."""
    with pytest.raises(ValidationError, match="does not read an edge affinity"):
        LinkerConfig(name="motion", mutual_bonus=10.0)

    assert LinkerConfig(name="flow", mutual_bonus=10.0).mutual_bonus == 10.0  # readers still accept it


def test_build_wires_the_mutual_bonus_into_both_affinity_readers():
    """A bonus plus the fused affinity builds the same `MutualBonus` cost term onto the assignment and flow linkers."""
    mutual = cast(EdgeAffinity, _Mutual())
    for name in ("assignment", "flow"):
        built = LinkerConfig(name=name, mutual_bonus=10.0).build(SPACING, cast(EdgeAffinity, object()), mutual)
        assert built.mutual == MutualBonus(mutual=mutual, bonus=10.0)


def test_build_without_the_fused_affinity_refuses_a_mutual_bonus():
    """A weight on a quantity nothing scored is an error, not a silently dropped instruction — as with the floor."""
    with pytest.raises(ValueError, match="needs the bidirectionally fused affinity"):
        LinkerConfig(name="flow", mutual_bonus=10.0).build(SPACING, cast(EdgeAffinity, object()))


def test_build_leaves_the_mutual_bonus_off_by_default():
    """No bonus means no second cost term, even when a fused affinity is available — the shipped cost is one term."""
    for name in ("assignment", "flow"):
        built = LinkerConfig(name=name).build(SPACING, cast(EdgeAffinity, object()), cast(EdgeAffinity, _Mutual()))
        assert built.mutual is None


def test_needs_mutual():
    """The fused pass is paid for exactly when a knob consumes it — the floor, the cost term, or both."""
    assert not LinkerConfig().needs_mutual
    assert LinkerConfig(agreement_floor=0.5).needs_mutual
    assert LinkerConfig(mutual_bonus=10.0).needs_mutual
    assert LinkerConfig(agreement_floor=0.5, mutual_bonus=10.0).needs_mutual


def test_boundary_prior_is_readable():
    """The boundary prior on a linker that charges no track boundary is refused — it could not move anything."""
    with pytest.raises(ValidationError, match="charges no track-boundary cost"):
        LinkerConfig(name="assignment", boundary_prior=True)

    assert LinkerConfig(name="flow", boundary_prior=True).boundary_prior  # the flow linker prices boundaries


def test_build_wires_the_boundary_prior_with_the_volume_shape():
    """Enabled, the prior reaches the flow linker carrying the video's imaged extent — the fact it cannot infer."""
    built = LinkerConfig(name="flow", boundary_prior=True).build(SPACING, volume_shape=(20, 100, 100))
    assert built.boundary == BoundaryPrior(volume_shape=(20, 100, 100))


def test_build_without_the_volume_shape_refuses_the_prior():
    """A prior the caller gave no volume to is an error, not a silently flat cost — as with the agreement floor."""
    with pytest.raises(ValueError, match="needs the video's"):
        LinkerConfig(name="flow", boundary_prior=True).build(SPACING)


def test_build_leaves_the_boundary_prior_off_by_default():
    """No flag means no prior, even when the volume shape is available — the shipped cost stays flat."""
    assert LinkerConfig(name="flow").build(SPACING, volume_shape=(20, 100, 100)).boundary is None


def test_ranker_bonus_is_readable():
    """A ranker bonus on a linker that cannot read an affinity is refused, exactly as the other weights are."""
    with pytest.raises(ValidationError, match="does not read an edge affinity"):
        LinkerConfig(name="motion", ranker_bonus=17.0)

    assert LinkerConfig(name="flow", ranker_bonus=17.0).ranker_bonus == 17.0  # readers still accept it


def test_build_wires_the_ranker_bonus_into_both_affinity_readers():
    """Set, the re-ranker term reaches every linker whose cost can price it, carrying the scored affinity."""
    ranker = cast(EdgeAffinity, _Mutual())
    for name in ("assignment", "flow"):
        built = LinkerConfig(name=name, ranker_bonus=17.0).build(SPACING, cast(EdgeAffinity, object()), ranker=ranker)
        assert built.ranker == RankerBonus(ranker=ranker, bonus=17.0)


def test_build_without_the_scored_ranker_refuses_a_ranker_bonus():
    """A weight on a quantity nobody scored is refused, not silently dropped — as with the mutual bonus."""
    with pytest.raises(ValueError, match="needs the local-association re-ranker"):
        LinkerConfig(name="flow", ranker_bonus=17.0).build(SPACING, cast(EdgeAffinity, object()))


def test_build_leaves_the_ranker_bonus_off_by_default():
    """Unset, the term is absent even when the ranker's probabilities are mounted — the shipped cost is untouched."""
    for name in ("assignment", "flow"):
        built = LinkerConfig(name=name).build(
            SPACING, cast(EdgeAffinity, object()), ranker=cast(EdgeAffinity, _Mutual())
        )
        assert built.ranker is None


def test_the_default_cost_is_byte_identical_with_the_ranker_mounted():
    """Default-off means arithmetically absent: the selected links do not move when a ranker is merely available.

    The stand-in ranker prefers the crossing pairing over the near one, so a term at any positive weight WOULD
    move the links — which is what makes the byte-equality a claim about the default, not about a null input.
    """
    detections = TrackGraph(
        node_ids=np.array([1, 2, 3, 4]),
        coordinates=np.array([[0, 0, 0, 0], [0, 0, 0, 6], [1, 0, 0, 1], [1, 0, 0, 5]]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    scored = cast(EdgeAffinity, _Scored(np.array([[0.9, 0.1], [0.1, 0.9]])))  # the affinity prefers the near pairing
    ranker = cast(EdgeAffinity, _Scored(np.array([[0.0, 1.0], [1.0, 0.0]])))  # the ranker prefers the crossing one
    for name in ("assignment", "flow"):
        config = LinkerConfig(name=name, gate_um=10.0)
        without = config.build(SPACING, scored).link(detections)
        with_ranker = config.build(SPACING, scored, ranker=ranker).link(detections)
        assert with_ranker.edges.tobytes() == without.edges.tobytes()
        moved = config.model_copy(update={"ranker_bonus": 40.0}).build(SPACING, scored, ranker=ranker).link(detections)
        assert moved.edges.tobytes() != without.edges.tobytes()  # the term is inert by default, not by inability
