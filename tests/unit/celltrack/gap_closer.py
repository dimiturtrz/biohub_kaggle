import numpy as np

from celltrack.center_prior import CenterConfirmer
from celltrack.gap_closer import GapCloser, SyntheticGap
from core.data.tracks import TrackGraph
from core.geometry import Spacing


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


def test_transform():
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
