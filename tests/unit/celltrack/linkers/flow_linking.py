import numpy as np

from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import AssignmentLinker
from celltrack.linkers.boundary_prior import BoundaryPrior
from celltrack.linkers.flow_linking import FlowLinker
from celltrack.linkers.motion_prediction import MotionPrediction
from celltrack.linkers.mutual_bonus import MutualBonus
from core.data.tracks import TrackGraph
from core.geometry import Spacing

ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)


class _Affinity:
    """A fixed per-gap probability matrix, indexed by timepoint — the edge scorer's `EdgeAffinity` contract."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


def _graph(coordinates: list[list[int]]) -> TrackGraph:
    """An edge-free detection graph whose node ids are its row indices."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.array(coordinates, dtype=np.int64),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def _sorted_edges(graph: TrackGraph) -> list[tuple[int, int]]:
    return sorted((int(a), int(b)) for a, b in graph.edges)


def test_link():
    """With a free track boundary the flow reduces to the per-frame assignment — identical links, edge for edge.

    Two cells per frame with a crossing affinity (each source prefers the far target): the global flow at
    `boundary_cost = 0` must select exactly the edges `AssignmentLinker` does, since free boundaries make the
    per-frame matching globally optimal. This pins the reduction that the boundary cost is then swept against.
    """
    detections = _graph([[0, 0, 0, 0], [0, 0, 0, 9], [1, 0, 0, 0], [1, 0, 0, 9]])
    # gap 0->1: sources (rows 0,1), targets (rows 2,3). Each source's true successor is the *far* one.
    affinity = _Affinity({0: np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64)})

    flow = FlowLinker(ISOTROPIC, max_distance_um=20.0, affinity=affinity, affinity_bonus=20.0, boundary_cost=0.0)
    assignment = AssignmentLinker(ISOTROPIC, max_distance_um=20.0, affinity=affinity, affinity_bonus=20.0)

    linked = _sorted_edges(flow.link(detections))
    assert linked == _sorted_edges(assignment.link(detections))
    assert linked == [(0, 3), (1, 2)]  # each source took its far, high-probability successor


def test_link_boundary_cost_raises_the_bar():
    """A positive track-boundary cost withholds a weakly-negative link the free-boundary flow would take.

    One source, one in-gate successor, transition cost `distance - bonus*P = 0 - 20*0.2 = -4`. At
    `boundary_cost = 0` the negative cost links them; at `boundary_cost = 3` the track costs `2*3` in
    boundaries, so the link (saving only 4) is no longer worth starting a track for — it is dropped.
    """
    detections = _graph([[0, 0, 0, 0], [1, 0, 0, 0]])
    affinity = _Affinity({0: np.array([[0.2]], dtype=np.float64)})

    free = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0, boundary_cost=0.0)
    priced = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0, boundary_cost=3.0)

    assert _sorted_edges(free.link(detections)) == [(0, 1)]
    assert _sorted_edges(priced.link(detections)) == []


def test_link_agreement_gate_composes_with_the_flow():
    """The mutual-agreement floor filters the flow's transitions too — an excluded pair is absent from the network.

    Same inverted scene as the assignment linker's: the near candidate (row 1) wins on forward probability and
    loses on the fused one, so gating reroutes the track through the true successor (row 2). The global
    optimiser and the per-frame one therefore share one admission rule, not two.
    """
    detections = _graph([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 5, 0]])
    affinity = _Affinity({0: np.array([[0.686, 0.4]], dtype=np.float64)})
    gate = AgreementGate(mutual=_Affinity({0: np.array([[0.1, 0.8]], dtype=np.float64)}), floor=0.5)

    ungated = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0)
    gated = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0, agreement=gate)

    assert _sorted_edges(ungated.link(detections)) == [(0, 1)]
    assert _sorted_edges(gated.link(detections)) == [(0, 2)]


def test_link_agreement_gate_off_is_byte_identical():
    """Unset, the gate is inert: the default flow linker's edges are byte-for-byte the explicit-None linker's."""
    detections = _graph([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 5, 0]])
    affinity = _Affinity({0: np.array([[0.686, 0.4]], dtype=np.float64)})
    default = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0)
    explicit = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=affinity, affinity_bonus=20.0, agreement=None)

    assert default.agreement is None  # the shipped default is OFF
    assert default.link(detections).edges.tobytes() == explicit.link(detections).edges.tobytes()


# The assignment linker's split scene: forward probability prefers the near candidate (row 1), mutual agreement
# the far one (row 2), and the two terms each carry half of the derived total P-weight at gate 10.
_SPLIT_SCENE = [[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 4, 0]]
_SPLIT_FORWARD = _Affinity({0: np.array([[0.9, 0.5]], dtype=np.float64)})
_SPLIT_MUTUAL = _Affinity({0: np.array([[0.1, 0.9]], dtype=np.float64)})
_SPLIT_BONUS = 10.0


def _split_flow(mutual: MutualBonus | None) -> FlowLinker:
    return FlowLinker(
        ISOTROPIC, max_distance_um=10.0, affinity=_SPLIT_FORWARD, affinity_bonus=_SPLIT_BONUS, mutual=mutual
    )


def test_link_mutual_bonus_composes_with_the_flow():
    """The second cost term prices the flow's transitions too — near = -8 vs far = -1 flips to -9 vs -10.

    Same arithmetic as the assignment linker's, so the global optimiser and the per-frame one weigh preference
    and agreement by one objective rather than two.
    """
    graph = _graph(_SPLIT_SCENE)

    assert _sorted_edges(_split_flow(None).link(graph)) == [(0, 1)]
    assert _sorted_edges(_split_flow(MutualBonus(mutual=_SPLIT_MUTUAL, bonus=_SPLIT_BONUS)).link(graph)) == [(0, 2)]


def test_link_mutual_bonus_off_is_byte_identical():
    """Unset, the second term is inert on the scene where it CHANGES the answer — the shipped flow is untouched."""
    graph = _graph(_SPLIT_SCENE)
    default = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=_SPLIT_FORWARD, affinity_bonus=_SPLIT_BONUS)

    assert default.mutual is None  # the shipped default is OFF
    assert default.link(graph).edges.tobytes() == _split_flow(None).link(graph).edges.tobytes()


# An 81-voxel cube at 1 um: with a 10 um gate the boundary band is voxels 0..10 and 70..80, so the centre (40)
# is deep interior. Anchors at t=0 and t=3 own the clip's ends, 38 um from the probe pair and thus out of gate.
_CUBE = (81, 81, 81)
_EDGE_SCENE = [[0, 40, 40, 78], [1, 40, 40, 2], [2, 40, 40, 2], [3, 40, 40, 78]]
_INTERIOR_SCENE = [[0, 40, 40, 78], [1, 40, 40, 40], [2, 40, 40, 40], [3, 40, 40, 78]]
# One in-gate transition per scene (rows 1->2), worth `0 - 20*0.2 = -4` against a boundary cost of 3 per arc.
_PAIR_AFFINITY = _Affinity({1: np.array([[0.2]], dtype=np.float64)})


def _priced(scene: list[list[int]], boundary: BoundaryPrior | None) -> list[tuple[int, int]]:
    linker = FlowLinker(
        ISOTROPIC,
        max_distance_um=10.0,
        affinity=_PAIR_AFFINITY,
        affinity_bonus=20.0,
        boundary_cost=3.0,
        boundary=boundary,
    )
    return _sorted_edges(linker.link(_graph(scene)))


def test_link_boundary_prior_admits_a_track_born_at_the_volume_edge():
    """A mid-movie track the flat cost refuses is admitted when it starts against a face — same track, same cost.

    The two scenes are identical but for WHERE the pair sits: 2 um from the x face, or 40 um deep. The link saves
    4 and two flat boundaries cost 6, so the flat linker drops both. Under the prior the edge pair's appearance
    and disappearance are explained by the field of view (a cell walked in through that face), so the link is all
    that is left to pay and it is taken — while the interior pair, with no such excuse, still pays 6 and is
    dropped. That geometric difference is exactly what a flat cost cannot express.
    """
    prior = BoundaryPrior(_CUBE)
    assert _priced(_EDGE_SCENE, prior) == [(1, 2)]
    assert _priced(_INTERIOR_SCENE, prior) == []
    assert _priced(_EDGE_SCENE, None) == []  # flat: the edge track is refused with the interior one


def test_link_boundary_prior_off_is_byte_identical():
    """Unset, the prior is inert: the default linker's graph is byte-for-byte the explicit-None linker's.

    Asserted on the scene where the prior CHANGES the answer, so a prior that leaked on by default would fail
    here rather than pass as a coincidence.
    """
    detections = _graph(_EDGE_SCENE)
    default = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=_PAIR_AFFINITY, affinity_bonus=20.0)
    explicit = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=_PAIR_AFFINITY, affinity_bonus=20.0, boundary=None)

    assert default.boundary is None  # the shipped default is OFF
    linked, reference = default.link(detections), explicit.link(detections)
    assert linked.edges.tobytes() == reference.edges.tobytes()
    assert linked.node_ids.tobytes() == reference.node_ids.tobytes()
    assert linked.coordinates.tobytes() == reference.coordinates.tobytes()


# A mover carried +6 um into row 1, then two equally-probable candidates: the true far successor (row 2, 6 um
# away) and a near neighbour (row 3, 3 um). Raw distance prefers the near one; measured from the damped
# prediction at x=9 the preference inverts (3 um vs 6 um) with nothing else changed.
_MOVER_SCENE = [[0, 0, 0, 0], [1, 0, 0, 6], [2, 0, 0, 12], [2, 0, 0, 3]]
_MOVER_AFFINITY = _Affinity({0: np.array([[1.0]]), 1: np.array([[0.5, 0.5]])})


def _mover_flow(prediction: MotionPrediction | None) -> FlowLinker:
    return FlowLinker(
        ISOTROPIC, max_distance_um=10.0, affinity=_MOVER_AFFINITY, affinity_bonus=20.0, prediction=prediction
    )


def test_link_motion_predicted_distance_prices_the_far_true_successor_nearer():
    """The cost's geometry decides between two equally-probable candidates — and predicting inverts the choice."""
    graph = _graph(_MOVER_SCENE)

    assert _sorted_edges(_mover_flow(None).link(graph)) == [(0, 1), (1, 3)]
    assert _sorted_edges(_mover_flow(MotionPrediction(affinity=_MOVER_AFFINITY)).link(graph)) == [(0, 1), (1, 2)]


def test_link_motion_predicted_distance_still_gates_on_the_raw_separation():
    """A target 11 um away is out of a 10 um gate however near the prediction puts it — admissibility is physics.

    The same +6 um mover, and a single candidate at raw 11 um whose PREDICTED distance is 8 um. Were the gate
    reading the predicted distance the pair would be admitted (and, at this bonus, linked); it is refused,
    because how far a cell can travel in one gap is a statement about travel, not about our estimate of it.
    """
    graph = _graph([[0, 0, 0, 0], [1, 0, 0, 6], [2, 0, 0, 17]])
    affinity = _Affinity({0: np.array([[1.0]]), 1: np.array([[0.9]])})
    linker = FlowLinker(
        ISOTROPIC,
        max_distance_um=10.0,
        affinity=affinity,
        affinity_bonus=20.0,
        prediction=MotionPrediction(affinity=affinity),
    )

    assert _sorted_edges(linker.link(graph)) == [(0, 1)]


def test_link_motion_predicted_distance_off_is_byte_identical():
    """Unset, the term is inert on the scene where it CHANGES the answer — the shipped raw-distance cost stands."""
    graph = _graph(_MOVER_SCENE)
    default = FlowLinker(ISOTROPIC, max_distance_um=10.0, affinity=_MOVER_AFFINITY, affinity_bonus=20.0)

    assert default.prediction is None  # the shipped default is OFF
    assert default.link(graph).edges.tobytes() == _mover_flow(None).link(graph).edges.tobytes()
