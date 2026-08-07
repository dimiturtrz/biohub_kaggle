from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from celltrack.linkers.assignment_linking import AssignmentLinker
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

    With an affinity wired the link reward is zero, so a probability-0 pair five units apart costs 5 — above a
    free skip, so the default linker ends the track. Charging 6 to disappear makes continuing the cheaper
    option, recovering the edge; this is the frontier's disappearance weight preferring an unbroken lineage.
    """
    affinity = _StubAffinity(np.array([[0.0]], dtype=np.float64))  # affinity present → no link reward, cost = distance
    coordinates = detections([[0, 0, 0, 0], [1, 0, 5, 0]])
    ends = AssignmentLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, affinity=affinity)
    assert ends.link(coordinates).edges.tolist() == []
    continues = AssignmentLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, affinity=affinity, disappearance_cost=6.0
    )
    assert continues.link(coordinates).edges.tolist() == [[0, 10]]
