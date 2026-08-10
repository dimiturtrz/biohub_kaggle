import numpy as np

from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import AssignmentLinker
from celltrack.linkers.flow_linking import FlowLinker
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
