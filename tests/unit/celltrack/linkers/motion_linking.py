from dataclasses import dataclass

import numpy as np

from celltrack.linkers.motion_linking import _KERNEL_MAX_FRAME_NODES, MotionHungarianLinker
from celltrack.linkers.product_of_experts import MotionDiffusionPrior
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


def test_blended_cost_fuses_as_a_product_of_experts_when_a_prior_is_set():
    """With a prior the cost is `distance^2 / 2sigma^2 - log P`, not the linear blend — the fusion actually lands."""
    affinity = _FixedAffinity(np.array([[0.8, 0.4]]))
    linker = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, prior=MotionDiffusionPrior(sigma_um=2.0))
    distance = np.array([[3.0, 5.0]])

    cost = linker._blended_cost(0, distance)

    expected = distance**2 / (2.0 * 2.0**2) - np.log(np.array([[0.8, 0.4]]))
    np.testing.assert_allclose(cost, expected)


def test_product_of_experts_flips_the_assignment_on_appearance_like_the_linear_bonus():
    """A confident appearance overrides the raw-nearest geometry through the fused cost, with no fitted bonus."""
    graph = a_graph([(0, 0, 0, 0), (1, 0, 0, 5), (1, 0, 0, 3)])  # target 1 = far, target 2 = near decoy
    # A wide prior makes motion near-flat, so appearance decides; strong P for the farther target (column 0).
    affinity = _FixedAffinity(np.array([[1.0, 0.0]]))
    fused = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, prior=MotionDiffusionPrior(sigma_um=100.0))
    assert edges_of(fused.link(graph)) == {(0, 1)}  # the product of experts pulls onto the favoured target


# Three confident movers (nodes 0-2) each step +6 in x within the tight gate; node 3 is a track start whose true
# successor drifted out of the tight gate, with a decoy nearer its undrifted position. Voters commit in the tight
# pass and their common-mode +6 is the drift estimate; the track start rides it in the loose pass.
_DRIFT_GRAPH = [
    (0, 0, 0, 0),
    (0, 0, 50, 0),
    (0, 0, 100, 0),
    (0, 0, 200, 20),  # sources: 3 voters + a track start
    (1, 0, 0, 6),
    (1, 0, 50, 6),
    (1, 0, 100, 6),  # the voters' +6 successors
    (1, 0, 200, 13),
    (1, 0, 200, 28),  # node 7 = near decoy, node 8 = drifted true
]


def test_self_estimated_drift_pulls_a_track_start_onto_the_drifted_successor():
    """The tight pass's common-mode step, estimated per gap, shifts a still-unmoved source onto its drifted target."""
    graph = a_graph(_DRIFT_GRAPH)
    assert (3, 7) in edges_of(LINKER.link(graph))  # driftless: the track start takes its raw-nearer decoy
    drifted = MotionHungarianLinker(_SPACING, 6.0, 10.0, use_drift=True)
    present = edges_of(drifted.link(graph))
    assert (3, 8) in present  # predicted x=26 (20 + estimated +6) now sits nearest the drifted successor
    assert (3, 7) not in present


def test_drift_needs_a_quorum_of_tight_matches_to_be_trusted():
    """Below the sample floor the per-gap estimate is dominated by own-motion, so no drift is applied."""
    two_voters = [
        (0, 0, 0, 0),
        (0, 0, 50, 0),
        (0, 0, 200, 20),  # only 2 voters — under the quorum
        (1, 0, 0, 6),
        (1, 0, 50, 6),
        (1, 0, 200, 13),
        (1, 0, 200, 28),
    ]
    graph = a_graph(two_voters)
    drifted = MotionHungarianLinker(_SPACING, 6.0, 10.0, use_drift=True)
    assert (2, 5) in edges_of(drifted.link(graph))  # no quorum → behaves as driftless, takes the decoy
    assert (2, 6) not in edges_of(drifted.link(graph))


def test_kernel_faithful_cost_is_motion_plus_raw_minus_bonus_prob():
    """Faithful mode prices a pair as the published kernel does: `motion + 0.05*raw - learned_bonus*prob`."""
    affinity = _FixedAffinity(np.array([[0.8, 0.4]]))
    linker = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, kernel_faithful=True, learned_bonus=1.0)
    motion = np.array([[3.0, 5.0]])
    raw = np.array([[4.0, 6.0]])

    cost = linker._cost(0, motion, raw)

    np.testing.assert_allclose(cost, motion + 0.05 * raw - 1.0 * np.array([[0.8, 0.4]]))


def test_kernel_faithful_honours_the_configured_bonus():
    """The learned-probability weight is the config's, not a hard-wired constant — 0.75 discounts less than 1.0."""
    affinity = _FixedAffinity(np.array([[0.8, 0.4]]))
    linker = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, kernel_faithful=True, learned_bonus=0.75)
    motion, raw = np.array([[3.0, 5.0]]), np.array([[4.0, 6.0]])

    cost = linker._cost(0, motion, raw)

    np.testing.assert_allclose(cost, motion + 0.05 * raw - 0.75 * np.array([[0.8, 0.4]]))


def test_default_mode_cost_is_the_shipped_blend_unchanged():
    """With the toggle off the cost is byte-identical to the shipped blended cost — the raw term never appears."""
    affinity = _FixedAffinity(np.array([[0.8, 0.4]]))
    linker = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, affinity_bonus=5.0)
    motion, raw = np.array([[3.0, 5.0]]), np.array([[4.0, 6.0]])

    np.testing.assert_allclose(linker._cost(0, motion, raw), linker._blended_cost(0, motion))


def test_kernel_faithful_sigmoids_an_out_of_range_probability():
    """A leaked logit (outside [0,1]) is squashed through the kernel's sigmoid guard; an in-range value is kept."""
    affinity = _FixedAffinity(np.array([[2.0, 0.4]]))  # column 0 = a stray logit, column 1 = a real probability
    linker = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, kernel_faithful=True, learned_bonus=1.0)
    zeros = np.zeros((1, 2))

    cost = linker._cost(0, zeros, zeros)

    guarded = np.array([[1.0 / (1.0 + np.exp(-2.0)), 0.4]])
    np.testing.assert_allclose(cost, -guarded)


def test_edge_admission_floor_rejects_a_sub_threshold_pair_only_when_engaged():
    """The kernel's 0.48 edge-candidate floor drops a pair the geometry would link but whose probability is low."""
    graph = a_graph([(0, 0, 0, 0), (1, 0, 0, 4)])  # one in-gate pair
    affinity = _FixedAffinity(np.array([[0.4]]))  # its learned edge probability is below the 0.48 floor
    open_gate = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity)
    assert edges_of(open_gate.link(graph)) == {(0, 1)}  # geometry alone admits it
    floored = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, edge_admission_prob=0.48)
    assert edges_of(floored.link(graph)) == set()  # the floor excludes the low-probability candidate


def test_edge_admission_floor_admits_a_pair_that_clears_it():
    """A pair whose learned probability clears the 0.48 floor is still linked under the admission gate."""
    graph = a_graph([(0, 0, 0, 0), (1, 0, 0, 4)])
    affinity = _FixedAffinity(np.array([[0.5]]))  # clears the floor
    floored = MotionHungarianLinker(_SPACING, 6.0, 10.0, affinity=affinity, edge_admission_prob=0.48)
    assert edges_of(floored.link(graph)) == {(0, 1)}


def test_edge_admission_floor_without_an_affinity_admits_everything():
    """With no scorer to read a probability from, the floor excludes nothing — the pure-geometry gate is unchanged."""
    graph = a_graph([(0, 0, 0, 0), (1, 0, 0, 4)])
    floored = MotionHungarianLinker(_SPACING, 6.0, 10.0, edge_admission_prob=0.48)
    assert edges_of(floored.link(graph)) == {(0, 1)}


def test_kernel_faithful_bails_on_a_frame_over_the_node_cap():
    """A frame exceeding the >2600-node cap skips the relink under faithful mode; our shipped path has no bail."""
    sources = [(0, 0, i, 0) for i in range(_KERNEL_MAX_FRAME_NODES + 1)]  # 2601 sources at t=0, over the cap
    graph = a_graph([*sources, (1, 0, 0, 1)])  # one target next to source 0
    target_id = _KERNEL_MAX_FRAME_NODES + 1

    faithful = MotionHungarianLinker(_SPACING, 6.0, 10.0, kernel_faithful=True)
    assert edges_of(faithful.link(graph)) == set()  # oversized frame → relink skipped entirely

    ours = MotionHungarianLinker(_SPACING, 6.0, 10.0)  # shipped path has no bail: it still links the near pair
    assert (0, target_id) in edges_of(ours.link(graph))
