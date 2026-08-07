from dataclasses import dataclass

import numpy as np

from celltrack.linkers.motion_linking import MotionHungarianLinker
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_SPACING = Spacing(z=1.0, y=1.0, x=1.0)
LINKER = MotionHungarianLinker(spacing=_SPACING, tight_gate_um=6.0, loose_gate_um=10.0)


@dataclass(frozen=True)
class _FixedAffinity:
    """A stub `EdgeAffinity` returning one hand-built probability matrix for the single frame gap under test."""

    matrix: np.ndarray

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self.matrix if timepoint == 0 else None


def a_graph(coordinates: list[tuple[int, int, int, int]]) -> TrackGraph:
    coords = np.array(coordinates, dtype=np.int64)
    return TrackGraph(
        node_ids=np.arange(len(coords), dtype=np.int64), coordinates=coords, edges=np.empty((0, 2), dtype=np.int64)
    )


def edges_of(graph: TrackGraph) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in graph.edges}


def test_link():
    """A cell within the tight gate joins to its successor in the next frame."""
    linked = LINKER.link(a_graph([(0, 0, 0, 0), (1, 0, 0, 4)]))
    assert edges_of(linked) == {(0, 1)}


def test_link_recovers_a_fast_mover_in_the_loose_pass():
    """A displacement beyond the tight gate but within the loose one is still linked, in the second pass."""
    linked = LINKER.link(a_graph([(0, 0, 0, 0), (1, 0, 0, 8)]))  # 8 um: past tight 6, within loose 10
    assert edges_of(linked) == {(0, 1)}


def test_link_uses_motion_to_pick_the_moving_successor_over_a_nearer_decoy():
    """Prediction from velocity links a mover to its extrapolated position, not the raw-nearest stationary decoy."""
    graph = a_graph([(0, 0, 0, 0), (1, 0, 0, 4), (2, 0, 0, 8), (2, 0, 0, 2)])  # node 2 = mover, node 3 = decoy
    present = edges_of(LINKER.link(graph))
    assert (1, 2) in present  # motion picks the extrapolated successor
    assert (1, 3) not in present  # not the nearer-by-raw-distance decoy


def test_link_learned_bonus_flips_the_assignment_to_the_favoured_but_farther_candidate():
    """A strong affinity for the farther in-gate target overrides the raw-nearest geometry pick."""
    graph = a_graph([(0, 0, 0, 0), (1, 0, 0, 5), (1, 0, 0, 3)])  # target 1 = far, target 2 = near decoy
    geometry = edges_of(LINKER.link(graph))
    assert geometry == {(0, 2)}  # geometry alone links the source to its nearer target
    affinity = _FixedAffinity(np.array([[1.0, 0.0]]))  # reward the farther target (column 0)
    blended = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, affinity_bonus=5.0)
    assert edges_of(blended.link(graph)) == {(0, 1)}  # bonus pulls the assignment onto the favoured target
