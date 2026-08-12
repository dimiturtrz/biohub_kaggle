from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import AssignmentLinker
from celltrack.linkers.mutual_bonus import MutualBonus
from core.data.tracks import TrackGraph
from core.geometry import Spacing

LINKER = AssignmentLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0)


@dataclass(frozen=True)
class _StubAffinity:
    """A fixed per-gap probability matrix, so a test can steer the assignment away from the nearest target."""

    matrix: Float[np.ndarray, "s t"]

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:
        return self.matrix if timepoint == 0 else None


def detections(coordinates: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_link():
    """A single chain links exactly as nearest-neighbour would — one edge."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 1, 0]]))
    assert graph.edges.tolist() == [[0, 10]]


def test_link_is_one_to_one_under_a_fork():
    """One source facing two next-frame candidates keeps a single child — MaxChildren(1), the nearer wins."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 3, 0]]))
    assert graph.edges.tolist() == [[0, 10]]


def test_link_ignores_a_target_beyond_the_gate():
    """A successor further than the gate is left unlinked rather than forced into an edge."""
    linker = AssignmentLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=5.0)
    graph = linker.link(detections([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 50, 0]]))
    assert graph.edges.tolist() == [[0, 10]]


def test_link_with_a_high_affinity_overrides_the_nearer_target():
    """A confident learned association wins the assignment against a geometrically nearer competitor.

    Mirrors the ILP test: source 0 at y=0, a near candidate at y=1 (row 0) and a far one at y=4 (row 1); an
    affinity scoring the far pair 0.99 vs the near 0.01, with a bonus large enough to overcome the 3-unit
    distance gap, flips the assignment to the learned partner.
    """
    affinity = _StubAffinity(np.array([[0.01, 0.99]], dtype=np.float64))
    linker = AssignmentLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        max_distance_um=10.0,
        affinity=affinity,
        affinity_bonus=20.0,
    )
    graph = linker.link(detections([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 4, 0]]))
    assert graph.edges.tolist() == [[0, 20]]


def test_disappearance_cost_recovers_a_track_break():
    """A positive disappearance cost links a source whose only successor costs more than a free skip would.

    With a learned term pricing the cost the link reward is zero, so a pair the affinity REJECTS (probability 0)
    five units apart costs 5 — above a free skip, so the default linker ends the track. Charging 6 to disappear
    makes continuing the cheaper option, recovering the edge; this is the frontier's disappearance weight
    preferring an unbroken lineage.

    The rejection has to come from the PROBABILITY, not from a zero weight: a zero `affinity_bonus` means no
    learned term is in play at all, which is the distance-only mode and links this pair on the link reward.
    """
    affinity = _StubAffinity(np.array([[0.0]], dtype=np.float64))  # priced and rejected, not merely unweighted
    coordinates = detections([[0, 0, 0, 0], [1, 0, 5, 0]])
    ends = AssignmentLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0
    )
    assert ends.link(coordinates).edges.tolist() == []
    continues = AssignmentLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        max_distance_um=10.0,
        affinity=affinity,
        affinity_bonus=20.0,
        disappearance_cost=6.0,
    )
    assert continues.link(coordinates).edges.tolist() == [[0, 10]]


def _inverted_scene() -> tuple[TrackGraph, _StubAffinity]:
    """The measured dense failure: a wrong NEAR neighbour outscoring the true, farther successor.

    Source at y=0; the near candidate (row 1, y=1) takes the inverted forward probability 0.686, the true
    successor (row 2, y=5) only 0.4. At a bonus of 20 the near pair costs 1 - 13.72 = -12.72 against the true
    pair's 5 - 8 = -3, so the cost alone links the wrong one.
    """
    return detections([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 5, 0]]), _StubAffinity(
        np.array([[0.686, 0.4]], dtype=np.float64)
    )


def test_link_agreement_gate_excludes_the_inverted_near_neighbour():
    """A high-forward / low-mutual candidate is refused admission, and the true high-mutual successor is linked.

    The gate reads the fused probabilities, where the near neighbour collapses (0.1 — it has its own better
    parent backwards) and the true successor holds (0.8). The cost is untouched: it still ranks by the sharp
    forward probability, but now over the admitted candidates only.
    """
    graph, affinity = _inverted_scene()
    ungated = AssignmentLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0
    )
    gated = AssignmentLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        max_distance_um=10.0,
        affinity=affinity,
        affinity_bonus=20.0,
        agreement=AgreementGate(mutual=_StubAffinity(np.array([[0.1, 0.8]], dtype=np.float64)), floor=0.5),
    )

    assert ungated.link(graph).edges.tolist() == [[0, 10]]  # the near mislink the cost cannot resist
    assert gated.link(graph).edges.tolist() == [[0, 20]]  # the true successor, admitted where the near one is not


def test_link_agreement_gate_off_is_byte_identical():
    """Unset, the gate changes nothing: the default linker's edges are byte-for-byte the explicit-None linker's."""
    graph, affinity = _inverted_scene()
    # A non-zero bonus is what makes this test say anything: at bonus 0 no learned term prices the cost, both
    # linkers fall into the distance-only mode the gate cannot touch, and the byte-identity holds vacuously.
    gates = (
        AssignmentLinker(
            spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0
        ),
        AssignmentLinker(
            spacing=Spacing(z=1.0, y=1.0, x=1.0),
            max_distance_um=10.0,
            affinity=affinity,
            affinity_bonus=20.0,
            agreement=None,
        ),
    )

    assert gates[0].agreement is None  # the shipped default is OFF
    assert gates[0].link(graph).edges.tobytes() == gates[1].link(graph).edges.tobytes()


# Source at y=0 facing a near candidate (row 1, y=1) and a far one (row 2, y=4). Forward probability prefers the
# near pair (0.9 vs 0.5), mutual agreement prefers the far one (0.1 vs 0.9) — preference and agreement disagree.
_SPLIT_SCENE = [[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 4, 0]]
_FORWARD = _StubAffinity(np.array([[0.9, 0.5]], dtype=np.float64))
_MUTUAL = _StubAffinity(np.array([[0.1, 0.9]], dtype=np.float64))
_SPLIT_BONUS = 10.0  # a + b = 20, the derived total P-weight at gate 10, split evenly between the two terms


def _split_linker(mutual: MutualBonus | None) -> AssignmentLinker:
    return AssignmentLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        max_distance_um=10.0,
        affinity=_FORWARD,
        affinity_bonus=_SPLIT_BONUS,
        mutual=mutual,
    )


def test_link_mutual_bonus_wins_a_pair_the_forward_term_ranks_second():
    """A lower-preference, higher-agreement pair takes the link — the arithmetic of the two-term cost.

    Forward only (`a=10`): near = 1 - 10*0.9 = -8, far = 4 - 10*0.5 = -1, so the near pair links. Adding the
    agreement term at `b=10`: near = -8 - 10*0.1 = -9, far = -1 - 10*0.9 = -10, so the far pair now links. The
    fused probability moved the answer as EVIDENCE the objective weighed, without filtering anything out.
    """
    graph = detections(_SPLIT_SCENE)

    assert _split_linker(None).link(graph).edges.tolist() == [[0, 10]]
    assert _split_linker(MutualBonus(mutual=_MUTUAL, bonus=_SPLIT_BONUS)).link(graph).edges.tolist() == [[0, 20]]


def test_link_mutual_bonus_off_is_byte_identical():
    """Unset, the second term is inert: the default linker's edges are byte-for-byte the explicit-None linker's.

    Asserted on the scene where the term CHANGES the answer, so a bonus leaking on by default fails here.
    """
    graph = detections(_SPLIT_SCENE)
    default = AssignmentLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, affinity=_FORWARD, affinity_bonus=_SPLIT_BONUS
    )

    assert default.mutual is None  # the shipped default is OFF
    assert default.link(graph).edges.tobytes() == _split_linker(None).link(graph).edges.tobytes()


def test_priced_affinity():
    """A zero weight is the knob's LIMIT, not a cliff — it must behave exactly like no affinity at all.

    This is the defect the predicate exists to close. The link reward used to switch on `affinity is None`, so
    a mounted affinity at `affinity_bonus = 0` fell between the two modes: prob-driven selection with nothing
    to drive it. Every arc then cost a positive distance, nothing linked, and the short-track filter deleted the
    whole graph — a sweep to zero read 0.0000 and looked like "the affinity is everything" when it was an empty
    graph. Pinned on both linkers' shared accessor.
    """
    affinity = _StubAffinity(np.array([[0.9]], dtype=np.float64))
    unweighted = AssignmentLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, affinity=affinity)
    unmounted = AssignmentLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0)
    coordinates = detections([[0, 0, 0, 0], [1, 0, 5, 0]])

    assert unweighted._priced_affinity() is None
    assert unweighted.link(coordinates).edges.tolist() == unmounted.link(coordinates).edges.tolist() == [[0, 10]]
