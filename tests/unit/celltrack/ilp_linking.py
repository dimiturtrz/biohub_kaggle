from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from celltrack.ilp_linking import ILPLinker
from core.data.tracks import TrackGraph
from core.geometry import Spacing

LINKER = ILPLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=10.0, division=True)


@dataclass(frozen=True)
class _StubAffinity:
    """A fixed per-gap probability matrix, so a test can steer the ILP away from the nearest target."""

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
    """A single chain links exactly as nearest-neighbour would — one edge, no spurious division."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 1, 0]]))
    assert graph.edges.tolist() == [[0, 10]]


def test_link_recovers_a_division():
    """One parent with two daughters gets both edges — the topology a one-to-one assignment cannot express."""
    graph = LINKER.link(detections([[0, 0, 0, 0], [1, 0, 3, 0], [1, 0, -3, 0]]))
    assert sorted(graph.edges.tolist()) == [[0, 10], [0, 20]]


def test_link_ignores_a_target_beyond_the_gate():
    """A successor further than the gate is left unlinked rather than forced into an edge."""
    linker = ILPLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), max_distance_um=5.0, division=True)
    graph = linker.link(detections([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 50, 0]]))
    assert graph.edges.tolist() == [[0, 10]]


def test_link_with_a_high_affinity_overrides_the_nearer_target():
    """A confident learned association wins the assignment against a geometrically nearer competitor.

    Source 0 sits at y=0 with two candidates one frame on: a near one at y=1 (row 0) and a far one at y=4
    (row 1). Distance alone links the near one; an affinity that scores the far pair 0.99 vs the near 0.01,
    with a bonus large enough to overcome the 3-unit distance gap, flips the ILP to the learned partner.
    """
    affinity = _StubAffinity(np.array([[0.01, 0.99]], dtype=np.float64))
    linker = ILPLinker(
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
        max_distance_um=10.0,
        division=False,
        affinity=affinity,
        affinity_bonus=20.0,
    )
    graph = linker.link(detections([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 4, 0]]))
    assert graph.edges.tolist() == [[0, 20]]
