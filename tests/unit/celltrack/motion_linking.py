import numpy as np

from celltrack.motion_linking import MotionHungarianLinker
from core.data.tracks import TrackGraph
from core.geometry import Spacing

LINKER = MotionHungarianLinker(spacing=Spacing(z=1.0, y=1.0, x=1.0), tight_gate_um=6.0, loose_gate_um=10.0)


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
