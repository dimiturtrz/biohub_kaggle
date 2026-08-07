import numpy as np

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
