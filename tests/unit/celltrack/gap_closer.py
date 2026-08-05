import numpy as np

from celltrack.center_prior import CenterConfirmer
from celltrack.gap_closer import BridgeConfig, DensityGapBridge, GapCloser, ReuseConfig, SyntheticGap
from core.data.tracks import TrackGraph
from core.geometry import Spacing


def bridger(reach_um: float = 10.0, max_added_fraction: float = 1.0) -> DensityGapBridge:
    return DensityGapBridge(
        spacing=Spacing(z=1.0, y=1.0, x=1.0), reach_um=reach_um, max_added_fraction=max_added_fraction
    )


def confirmer(value: float) -> CenterConfirmer:
    """A centre-prior confirmer answering `value` everywhere — 1.0 confirms every midpoint, 0.0 vetoes them."""
    heatmaps = tuple(np.full((1, 1, 1), value, dtype=np.float32) for _ in range(6))
    return CenterConfirmer(heatmaps=heatmaps, pool_factor=4, threshold=0.5, window_z=1, window_yx=2)


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2) if edges else np.empty((0, 2), dtype=np.int64),
    )


def closer(
    confirmer: CenterConfirmer | None = None, synthetic: SyntheticGap | None = None, **overrides: float
) -> GapCloser:
    settings: dict[str, float] = {"gate_um": 10.0, "reuse_um": 5.0, "max_added_fraction": 1.0}
    return GapCloser(
        spacing=Spacing(z=1.0, y=1.0, x=1.0), confirmer=confirmer, synthetic=synthetic, **(settings | overrides)
    )


def gappy() -> TrackGraph:
    """An end at t1 and a start at t3 six voxels apart in x, with no isolated node near their midpoint."""
    # P->E (end at t1) at x0; S (start at t3)->S2 at x6; a wide gap with nothing reusable between them.
    return graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 6], [4, 0, 0, 6]], [[0, 1], [2, 3]])


def test_gap_closer_transform():
    """A track ending at t, an isolated node at t+1 near the midpoint, and a start at t+2 are bridged."""
    # nodes: P->E (end at t1), M isolated at t2, S start at t3; all colinear at the origin.
    gapped = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 0]], [[0, 1]])
    bridged = closer().transform(gapped).edges.tolist()
    assert [10, 30] in bridged and [30, 20] in bridged


def test_transform_leaves_a_gap_open_without_a_reusable_node():
    """No isolated node near the midpoint means the gap is left open, not filled with an invention."""
    far_middle = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 50]], [[0, 1]])
    assert closer().transform(far_middle).edges.tolist() == [[0, 10]]


def test_transform_rejects_an_end_and_start_beyond_the_gate():
    """Two track fragments too far apart to be the same cell are not joined."""
    far_apart = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 50], [2, 0, 0, 25]], [[0, 1]])
    assert closer().transform(far_apart).edges.tolist() == [[0, 10]]


def test_transform_bridges_only_a_midpoint_the_centre_prior_confirms():
    """With a veto, a reused midpoint bridges the gap only when the centre prior confirms a cell at its voxel."""
    gapped = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 0]], [[0, 1]])
    confirmed = closer(confirmer(1.0)).transform(gapped).edges.tolist()
    assert [10, 30] in confirmed and [30, 20] in confirmed
    assert closer(confirmer(0.0)).transform(gapped).edges.tolist() == [[0, 10]]


def test_transform_honours_the_cap():
    """A zero budget adds no bridges however reconnectable the gap is."""
    gapped = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [2, 0, 0, 0]], [[0, 1]])
    assert closer(max_added_fraction=0.0).transform(gapped).edges.tolist() == [[0, 10]]


def test_transform_inserts_a_confirmed_synthetic_node_when_no_reusable_node():
    """A wide gap with nothing to reuse gets a new node at the midpoint once the centre prior confirms it."""
    synthetic = SyntheticGap(min_span_um=3.0, max_added_fraction=1.0)
    closed = closer(confirmer(1.0), synthetic).transform(gappy())
    assert closed.node_ids.tolist() == [0, 10, 20, 30, 31]
    assert closed.coordinates[-1].tolist() == [2, 0, 0, 3]
    assert [10, 31] in closed.edges.tolist() and [31, 20] in closed.edges.tolist()


def test_transform_rejects_an_unconfirmed_wide_synthetic_gap():
    """A wide gap the centre prior does not confirm inserts no node — no unconfirmed invention."""
    synthetic = SyntheticGap(min_span_um=3.0, max_added_fraction=1.0)
    closed = closer(confirmer(0.0), synthetic).transform(gappy())
    assert closed.node_ids.tolist() == [0, 10, 20, 30]
    assert closed.edges.tolist() == [[0, 10], [20, 30]]


def test_transform_inserts_a_short_synthetic_gap_without_confirmation():
    """A gap below the span threshold is close-motion continuation — inserted even against a vetoing prior."""
    synthetic = SyntheticGap(min_span_um=100.0, max_added_fraction=1.0)
    closed = closer(confirmer(0.0), synthetic).transform(gappy())
    assert closed.node_ids.tolist() == [0, 10, 20, 30, 31]


def test_transform_without_a_confirmer_stays_reuse_only():
    """Synthetic policy without a mounted confirmer keeps the reuse-only behaviour — no node is invented."""
    synthetic = SyntheticGap(min_span_um=3.0, max_added_fraction=1.0)
    closed = closer(synthetic=synthetic).transform(gappy())
    assert closed.node_ids.tolist() == [0, 10, 20, 30]
    assert closed.edges.tolist() == [[0, 10], [20, 30]]


def test_transform_honours_the_synthetic_node_cap():
    """A zero synthetic budget inserts no node however confirmable the gap is."""
    synthetic = SyntheticGap(min_span_um=3.0, max_added_fraction=0.0)
    closed = closer(confirmer(1.0), synthetic).transform(gappy())
    assert closed.node_ids.tolist() == [0, 10, 20, 30]


def test_density_gap_bridge_transform():
    """An end at t1 and a lone start at t3 become one track: a synthetic t2 node and its two edges."""
    gapped = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]], [[0, 1], [2, 3]])
    closed = bridger().transform(gapped)
    assert closed.node_ids.tolist() == [0, 10, 20, 30, 31]
    assert closed.coordinates[-1].tolist() == [2, 0, 0, 0]
    assert [10, 31] in closed.edges.tolist() and [31, 20] in closed.edges.tolist()


def test_density_bridge_predicts_along_the_ends_velocity():
    """A moving end is bridged to the start at its predicted position, not the one it stood still over."""
    moving = graph([[0, 0, 0, 0], [1, 0, 0, 2], [3, 0, 0, 6], [4, 0, 0, 6]], [[0, 1], [2, 3]])
    closed = bridger().transform(moving)
    assert closed.coordinates[-1].tolist() == [2, 0, 0, 4]
    assert [10, 31] in closed.edges.tolist() and [31, 20] in closed.edges.tolist()


def test_density_bridge_leaves_an_ambiguous_gap_open():
    """Two starts within the radius mean the successor is not unique, so nothing is bridged."""
    ambiguous = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [3, 0, 0, 2]], [[0, 1]])
    assert bridger().transform(ambiguous).node_ids.tolist() == [0, 10, 20, 30]


def test_density_bridge_tightens_the_radius_in_a_crowded_region():
    """A near same-frame neighbour shrinks the Voronoi radius below the gap, suppressing a bridge a sparse
    region would take."""
    crowded = graph(
        [[0, 0, 0, 2], [1, 0, 0, 2], [2, 0, 0, 2], [0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 3]],
        [[0, 1], [1, 2], [3, 4]],
    )
    assert bridger().transform(crowded).node_ids.tolist() == [0, 10, 20, 30, 40, 50]
    sparse = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 3], [4, 0, 0, 3]], [[0, 1], [2, 3]])
    assert len(bridger().transform(sparse).node_ids) == 5


def test_density_bridge_caps_the_radius_at_the_motion_reach():
    """An isolated end has no neighbour spacing, so a start past the motion reach stays unbridged."""
    far = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 20], [4, 0, 0, 20]], [[0, 1], [2, 3]])
    assert bridger(reach_um=5.0).transform(far).node_ids.tolist() == [0, 10, 20, 30]


def test_density_bridge_honours_the_cap():
    """A zero added-node budget bridges nothing however clean the gap."""
    gapped = graph([[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]], [[0, 1], [2, 3]])
    assert bridger(max_added_fraction=0.0).transform(gapped).node_ids.tolist() == [0, 10, 20, 30]


def test_density_bridge_leaves_a_gapless_graph_unchanged():
    """No qualifying end means the graph is returned as-is."""
    linked = graph([[0, 0, 0, 0], [1, 0, 0, 0]], [[0, 1]])
    assert bridger().transform(linked).edges.tolist() == [[0, 10]]


def test_bridge_config_build():
    """`BridgeConfig.build` carries its knobs into a `DensityGapBridge` at the given spacing."""
    stage = BridgeConfig(reach_um=8.0, max_added_fraction=0.03).build(Spacing(1.0, 1.0, 1.0))
    assert isinstance(stage, DensityGapBridge)
    assert (stage.reach_um, stage.max_added_fraction) == (8.0, 0.03)


def test_reuse_config_build():
    """`ReuseConfig.build` maps `radius_um` onto the `GapCloser`'s `reuse_um` and carries the gate + cap."""
    stage = ReuseConfig(gate_um=5.0, radius_um=2.0, max_added_fraction=0.04).build(Spacing(1.0, 1.0, 1.0))
    assert isinstance(stage, GapCloser)
    assert (stage.gate_um, stage.reuse_um, stage.max_added_fraction) == (5.0, 2.0, 0.04)
