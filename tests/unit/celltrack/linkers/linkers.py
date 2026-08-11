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
from celltrack.linkers.motion_prediction import MotionPrediction
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
    assert MotionHungarianLinker(
        spacing=SPACING, tight_gate_um=5.0, loose_gate_um=9.0, affinity_bonus=18.0
    ) == motion.build(SPACING)


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
    """A bonus on a linker with no affinity term is refused — the knob could not have moved anything.

    `nn` is the exemplar because it is pure geometry: no affinity argument at all, so no wiring change could
    make the bonus mean something there. `motion` is NOT such a linker — its cost blends the affinity inside a
    geometric gate — which is why it appears below as an acceptor.
    """
    with pytest.raises(ValidationError, match="does not read affinity_bonus"):
        LinkerConfig(name="nn", affinity_bonus=20.0)

    for name in ("assignment", "flow", "motion"):  # every linker whose cost prices the forward probability
        assert LinkerConfig(name=name, affinity_bonus=20.0).effective_bonus == 20.0


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
    """An agreement floor on a linker with no admission gate is refused, exactly as an unreadable bonus is.

    The floor's readers are NARROWER than the bonus's: `motion` admits candidates by its raw tight/loose
    geometric gates alone, so a floor there would be accepted and silently dropped even though the same linker
    does price the bonus. One reader set for both knobs could only be wrong in one direction or the other.
    """
    for name in ("nn", "motion"):
        with pytest.raises(ValidationError, match="does not read agreement_floor"):
            LinkerConfig(name=name, agreement_floor=0.5)

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
    """A mutual bonus on a linker whose cost has no mutual term is refused, as an unreadable bonus and floor are.

    Same narrower set as the floor, for the same reason: `motion`'s cost is the forward probability (and, when
    set, the re-ranker) — there is no second term for the fused probability to enter, so it refuses.
    """
    for name in ("nn", "motion"):
        with pytest.raises(ValidationError, match="does not read mutual_bonus"):
            LinkerConfig(name=name, mutual_bonus=10.0)

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


def test_motion_distance_is_readable():
    """The flag on a linker whose cost it cannot rewrite is refused — including the one that already predicts."""
    for name in ("assignment", "motion"):
        with pytest.raises(ValidationError, match="does not price a motion-predicted distance"):
            LinkerConfig(name=name, motion_distance=True)

    assert LinkerConfig(name="flow", motion_distance=True).motion_distance


def test_build_wires_the_motion_predicted_distance_with_the_affinity_its_velocity_reads():
    """Enabled, the term reaches the flow linker carrying the affinity the previous gap's step is derived from."""
    affinity = cast(EdgeAffinity, _Mutual())
    built = LinkerConfig(name="flow", motion_distance=True).build(SPACING, affinity)
    assert built.prediction == MotionPrediction(affinity=affinity)


def test_build_without_the_affinity_refuses_the_motion_predicted_distance():
    """Without an affinity every velocity would be zero, so the flag is refused rather than silently inert."""
    with pytest.raises(ValueError, match="needs the edge affinity"):
        LinkerConfig(name="flow", motion_distance=True).build(SPACING)


def test_build_leaves_the_motion_predicted_distance_off_by_default():
    """No flag means the shipped raw-distance cost, even with an affinity mounted for the learned terms."""
    assert LinkerConfig(name="flow").build(SPACING, cast(EdgeAffinity, _Mutual())).prediction is None


def test_without_ranker_holds_the_cost_geometry():
    """The context pass links in the SAME geometry as the re-ranked one — only the evidence weights move."""
    assert LinkerConfig(name="flow", motion_distance=True, ranker_bonus=17.0).without_ranker().motion_distance


def test_ranker_bonus_is_readable():
    """A ranker bonus on a linker with no per-pair cost to discount is refused, exactly as the other weights are.

    Its readers match the bonus's, not the floor's: the term subtracts from whatever cost the linker already
    formed, so `motion` carries it — which is the whole point, since the re-ranker's features are computed off
    a motion-predicted distance.
    """
    with pytest.raises(ValidationError, match="does not read ranker_bonus"):
        LinkerConfig(name="nn", ranker_bonus=17.0)

    for name in ("assignment", "flow", "motion"):
        assert LinkerConfig(name=name, ranker_bonus=17.0).ranker_bonus == 17.0


def test_build_wires_the_ranker_bonus_into_every_ranker_reader():
    """Set, the re-ranker term reaches every linker whose cost can price it, carrying the scored affinity."""
    ranker = cast(EdgeAffinity, _Mutual())
    for name in ("assignment", "flow", "motion"):
        built = LinkerConfig(name=name, ranker_bonus=17.0).build(SPACING, cast(EdgeAffinity, object()), ranker=ranker)
        assert built.ranker == RankerBonus(ranker=ranker, bonus=17.0)


def test_build_wires_the_motion_linker_with_the_affinity_it_prices():
    """The motion row threads the affinity, its bonus and the re-ranker through — the path that was unreachable.

    `MotionHungarianLinker` has always accepted `affinity`/`affinity_bonus` and had a unit test exercising them,
    but its builder row dropped both, so no config could reach the blended cost. This is that wiring.
    """
    affinity = cast(EdgeAffinity, object())
    ranker = cast(EdgeAffinity, _Mutual())
    built = LinkerConfig(name="motion", gate_um=9.0, tight_um=5.0, affinity_bonus=3.0, ranker_bonus=17.0).build(
        SPACING, affinity, ranker=ranker
    )

    assert isinstance(built, MotionHungarianLinker)
    assert built.affinity is affinity
    assert (built.tight_gate_um, built.loose_gate_um, built.affinity_bonus) == (5.0, 9.0, 3.0)
    assert built.ranker == RankerBonus(ranker=ranker, bonus=17.0)


def test_the_motion_cost_reads_the_affinity_the_config_hands_it():
    """A strong affinity on the farther in-gate target overrides the geometric pick — THROUGH the config.

    The linker's own unit test proves the blended cost works when constructed by hand; this proves the config
    path reaches it, which is exactly what the dropped builder arguments broke.
    """
    detections = TrackGraph(
        node_ids=np.array([1, 2, 3, 4]),
        coordinates=np.array([[0, 0, 0, 0], [0, 0, 0, 6], [1, 0, 0, 1], [1, 0, 0, 5]]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    crossing = cast(EdgeAffinity, _Scored(np.array([[0.0, 1.0], [1.0, 0.0]])))
    config = LinkerConfig(name="motion", gate_um=10.0, tight_um=10.0)

    geometry = config.model_copy(update={"affinity_bonus": 0.0}).build(SPACING, crossing).link(detections)
    blended = config.model_copy(update={"affinity_bonus": 40.0}).build(SPACING, crossing).link(detections)
    assert geometry.edges.tolist() == [[1, 3], [2, 4]]  # the near pairing, on distance alone
    assert blended.edges.tolist() == [[1, 4], [2, 3]]  # the affinity flips it — the knob is live


def test_needs_ranker():
    """The context linking pass is paid for exactly when the cost prices its result."""
    assert not LinkerConfig().needs_ranker
    assert LinkerConfig(ranker_bonus=17.0).needs_ranker


def test_learned_budget():
    """The learned-evidence budget is the two probability weights summed, over the derived bonus as well."""
    assert LinkerConfig(gate_um=10.0).learned_budget == 20.0  # derived affinity bonus, no ranker
    assert LinkerConfig(affinity_bonus=3.0, ranker_bonus=17.0).learned_budget == 20.0
    assert LinkerConfig(gate_um=10.0, ranker_bonus=5.0).learned_budget == 25.0


def test_without_ranker():
    """The context pass spends the WHOLE learned-evidence budget on the affinity — the term it can price.

    Dropping the ranker's weight instead of folding it would link the context at the arm's leftover bonus, so
    the graph the 22 features describe would degrade in lockstep with the weight under test. Everything else
    is held: the context only describes "the current best link set" if it came from the cost being re-ranked.
    """
    config = LinkerConfig(gate_um=12.0, affinity_bonus=3.0, mutual_bonus=4.0, ranker_bonus=17.0)
    context = config.without_ranker()
    assert context.ranker_bonus is None
    assert context.affinity_bonus == 20.0
    assert (context.gate_um, context.mutual_bonus, context.name) == (12.0, 4.0, config.name)
    assert (config.affinity_bonus, config.ranker_bonus) == (3.0, 17.0)  # the priced config itself is unchanged


def test_the_context_pass_is_invariant_to_the_split_of_one_budget():
    """Two splits of the SAME learned-evidence budget link an IDENTICAL pass-1 context — the ic44 invariance.

    This is the property that makes a sweep over the split readable: the ranker's features are computed off the
    context graph, so if the context moved with the split, "monotone in ranker_bonus" would be indistinguishable
    from "monotone in how degraded the context is". The graph is compared, not only the config, because the
    invariance that matters is the one the features see.
    """
    detections = TrackGraph(
        node_ids=np.array([1, 2, 3, 4]),
        coordinates=np.array([[0, 0, 0, 0], [0, 0, 0, 6], [1, 0, 0, 1], [1, 0, 0, 5]]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    scored = cast(EdgeAffinity, _Scored(np.array([[0.9, 0.1], [0.1, 0.9]])))
    splits = ((20.0, 0.0), (17.0, 3.0), (3.0, 17.0), (0.0, 20.0))
    for name in ("assignment", "flow", "motion"):
        contexts = [
            LinkerConfig(
                name=name, gate_um=10.0, disappearance_cost=3.0, affinity_bonus=a, ranker_bonus=b
            ).without_ranker()
            for a, b in splits
        ]
        assert len({context.model_dump_json() for context in contexts}) == 1
        graphs = [context.build(SPACING, scored).link(detections) for context in contexts]
        assert len({graph.edges.tobytes() for graph in graphs}) == 1


def test_build_without_the_scored_ranker_refuses_a_ranker_bonus():
    """A weight on a quantity nobody scored is refused, not silently dropped — as with the mutual bonus."""
    with pytest.raises(ValueError, match="needs the local-association re-ranker"):
        LinkerConfig(name="flow", ranker_bonus=17.0).build(SPACING, cast(EdgeAffinity, object()))


def test_build_leaves_the_ranker_bonus_off_by_default():
    """Unset, the term is absent even when the ranker's probabilities are mounted — the shipped cost is untouched."""
    for name in ("assignment", "flow", "motion"):
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
    for name in ("assignment", "flow", "motion"):
        config = LinkerConfig(name=name, gate_um=10.0)
        without = config.build(SPACING, scored).link(detections)
        with_ranker = config.build(SPACING, scored, ranker=ranker).link(detections)
        assert with_ranker.edges.tobytes() == without.edges.tobytes()
        moved = config.model_copy(update={"ranker_bonus": 40.0}).build(SPACING, scored, ranker=ranker).link(detections)
        assert moved.edges.tobytes() != without.edges.tobytes()  # the term is inert by default, not by inability


def _one_source_two_targets() -> TrackGraph:
    """One cell at the origin with a near and a far candidate successor — the ramp's whole tradeoff, minimal."""
    return TrackGraph(
        node_ids=np.array([1, 2, 3]),
        coordinates=np.array([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 7]]),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_evidence_ramp_is_readable():
    """The ramp is refused on a linker whose cost it cannot reweight — the same rule as every other knob.

    `motion` is a refuser here even though it PRICES the forward affinity: it admits through a tight/loose gate
    pair, so there is no single admitted-candidate set whose mean separation the ramp normalises by.
    """
    for name in ("nn", "motion"):
        with pytest.raises(ValidationError, match="does not read evidence_ramp"):
            LinkerConfig(name=name, evidence_ramp=1.0)

    for name in ("assignment", "flow"):
        assert LinkerConfig(name=name, evidence_ramp=1.0).build(SPACING).ramp is not None


def test_build_leaves_the_evidence_ramp_off_by_default():
    """Unset means absent, not a zero-strength term — the default cost carries no ramp object at all."""
    for name in ("assignment", "flow"):
        assert LinkerConfig(name=name).build(SPACING, cast(EdgeAffinity, object())).ramp is None


def test_the_zero_strength_ramp_is_byte_identical_to_the_shipped_cost():
    """The control arm runs the ramp's OWN arithmetic and lands on the shipped links, byte for byte.

    This is the claim the whole sweep rests on: if `evidence_ramp=0` moved a single edge, every arm above it
    would be confounded by whatever else the ramp changed. It goes through the CONFIG path — `--set
    linker.evidence_ramp=…` is how the sweep reaches it, so that is what must be identity at zero.
    """
    detections = _one_source_two_targets()
    scored = cast(EdgeAffinity, _Scored(np.array([[0.8, 0.55]])))  # the near target is preferred at a flat weight
    for name in ("assignment", "flow"):
        config = LinkerConfig(name=name, gate_um=10.0, affinity_bonus=20.0)
        shipped = config.build(SPACING, scored).link(detections)
        control = config.model_copy(update={"evidence_ramp": 0.0}).build(SPACING, scored).link(detections)
        assert control.edges.tobytes() == shipped.edges.tobytes()


def test_a_dosed_ramp_moves_the_bonus_onto_the_far_candidate():
    """At a positive strength the far candidate's evidence outweighs the near one's — the mechanism, end to end.

    Flat, the near target wins (`1 - 20*0.8` beats `7 - 20*0.55`). The ramp holds the same budget and
    redistributes it by separation (mean 4um: weights 0.25 and 1.75), which flips the choice to the far one —
    so the zero-strength equality above is a claim about the DOSE, not about a term that cannot act.
    """
    detections = _one_source_two_targets()
    scored = cast(EdgeAffinity, _Scored(np.array([[0.8, 0.55]])))
    for name in ("assignment", "flow"):
        config = LinkerConfig(name=name, gate_um=10.0, affinity_bonus=20.0)
        assert config.build(SPACING, scored).link(detections).edges.tolist() == [[1, 2]]
        dosed = config.model_copy(update={"evidence_ramp": 1.0}).build(SPACING, scored).link(detections)
        assert dosed.edges.tolist() == [[1, 3]]


def test_appearance_cost_defaults_to_the_boundary_charge():
    """Unset, a track's start costs exactly what its end costs — the shipped symmetric behaviour, byte-identical.

    Asserted on the LINKED GRAPH rather than the config, because the claim is about what the solver decides:
    an asymmetry that changed no edge would pass a config comparison and still be wrong.

    An AFFINITY is mandatory here and the assertion is worthless without one: every arc weight in the flow
    network is positive unless `affinity_bonus * P` drives a transition below zero, and the cheapest
    circulation over all-positive weights is NO FLOW. Built without one this linker emits an empty graph, so
    both sides of the comparison would be empty and the test would pass while checking nothing.
    """
    detections = TrackGraph(
        node_ids=np.array([1, 2, 3, 4]),
        coordinates=np.array([[0, 0, 0, 0], [0, 0, 0, 6], [1, 0, 0, 1], [1, 0, 0, 5]]),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    scored = cast(EdgeAffinity, _Scored(np.array([[0.9, 0.1], [0.1, 0.9]])))
    symmetric = LinkerConfig(name="flow", gate_um=10.0, disappearance_cost=3.0).build(SPACING, scored)
    spelled = LinkerConfig(name="flow", gate_um=10.0, disappearance_cost=3.0, appearance_cost=3.0).build(
        SPACING, scored
    )
    linked = symmetric.link(detections).edge_rows()
    assert len(linked) > 0  # the comparison below is vacuous on an empty graph — see the docstring
    assert np.array_equal(linked, spelled.link(detections).edge_rows())


def test_pricing_the_two_boundaries_apart_reaches_the_network():
    """A cheap start against an expensive end really is charged on the two arcs — the knob is not inert.

    Asserted on the charges the graph builder receives rather than on linked edges, because a flow whose arc
    weights are all POSITIVE has zero flow as its cheapest circulation: without an affinity to push a transition
    below zero this linker emits no edges at all, so an edge-level assertion here would test nothing. Without
    this test the identity test above would also pass for an `appearance_cost` the network never reads, which is
    exactly how three other mechanisms in this repo stayed unreachable while their tests were green.
    """
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])
    timepoints = np.array([0, 0, 1, 1])
    symmetric = LinkerConfig(name="flow", gate_um=4.0, disappearance_cost=0.4).build(SPACING)
    asymmetric = LinkerConfig(name="flow", gate_um=4.0, disappearance_cost=0.4, appearance_cost=8.0).build(SPACING)
    assert symmetric._boundary_costs(positions, timepoints) == ([400] * 4, [400] * 4)
    assert asymmetric._boundary_costs(positions, timepoints) == ([8000] * 4, [400] * 4)


def test_appearance_cost_is_readable():
    """Only the flow network charges the two boundaries on separate arcs, so only it may take the knob."""
    for name in ("assignment", "motion", "nn"):
        with pytest.raises(ValidationError, match="one price for a track boundary"):
            LinkerConfig(name=name, gate_um=10.0, appearance_cost=1.0)
    assert LinkerConfig(name="flow", gate_um=10.0, appearance_cost=1.0).appearance_cost == 1.0
