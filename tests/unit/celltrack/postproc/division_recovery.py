import numpy as np

from celltrack.detectors.center_prior import CenterConfirmer
from celltrack.postproc.division_recovery import DivisionRecovery, DivisionRecoveryConfig
from core.data.tracks import TrackGraph
from core.geometry import Spacing


def confirmer(value: float) -> CenterConfirmer:
    """A centre-prior confirmer that answers `value` everywhere — 1.0 confirms every fork, 0.0 vetoes them all."""
    heatmaps = tuple(np.full((1, 1, 1), value, dtype=np.float32) for _ in range(4))
    return CenterConfirmer(heatmaps=heatmaps, pool_factor=4, threshold=0.5, window_z=1, window_yx=2)


def graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)) * 10,
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=(np.array(edges, dtype=np.int64) * 10).reshape(-1, 2) if edges else np.empty((0, 2), dtype=np.int64),
    )


def recovery(confirmer: CenterConfirmer | None = None, **overrides: float) -> DivisionRecovery:
    settings: dict[str, float] = {"parent_gate_um": 10.5, "sister_gate_um": 8.0, "max_added_fraction": 1.0}
    return DivisionRecovery(spacing=Spacing(z=1.0, y=1.0, x=1.0), confirmer=confirmer, **(settings | overrides))


def test_transform():
    """An unparented cell beside a single-child parent becomes that parent's second daughter."""
    mitosis = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -1]], [[0, 1]])
    forked = recovery().transform(mitosis)
    assert forked.edges.tolist() == [[0, 10], [0, 20]]
    assert forked.division_parents().tolist() == [0]


def test_transform_rejects_a_daughter_whose_sister_is_far_away():
    """Proximity to the mother is not enough — a real mitosis leaves its two products adjacent."""
    lost_link = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -9]], [[0, 1]])
    assert recovery().transform(lost_link).edges.tolist() == [[0, 10]]


def test_transform_rejects_a_daughter_beyond_the_parent_gate():
    """A cell too far from any mother is an appearance, not a division."""
    distant = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 20]], [[0, 1]])
    assert recovery().transform(distant).edges.tolist() == [[0, 10]]


def test_transform_rejects_a_mother_whose_existing_link_is_long():
    """A mother whose surviving link is itself a long mis-link is not a divider — the existing-child gate."""
    mislinked = graph([[0, 0, 0, 0], [1, 0, 0, 9], [1, 0, 0, 1]], [[0, 1]])
    gated = recovery(existing_child_gate_um=7.8, sister_gate_um=10.0)
    assert gated.transform(mislinked).edges.tolist() == [[0, 10]]
    ungated = recovery(existing_child_gate_um=float("inf"), sister_gate_um=10.0)
    assert ungated.transform(mislinked).edges.tolist() == [[0, 10], [0, 20]]


def test_transform_never_gives_a_parent_a_third_child():
    """A node that already forked is not a candidate — the metric's divisions are binary."""
    already_forked = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -1], [1, 0, 1, 0]], [[0, 1], [0, 2]])
    assert recovery().transform(already_forked).edges.tolist() == [[0, 10], [0, 20]]


def test_transform_honours_the_global_cap():
    """The cap is what stops a permissive gate from inventing forks across the whole video."""
    mitosis = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -1]], [[0, 1]])
    assert recovery(max_added_fraction=0.0).transform(mitosis).edges.tolist() == [[0, 10]]


def test_transform_keeps_a_fork_the_centre_prior_confirms():
    """With a veto, a recovered daughter survives only when the centre prior confirms a cell at its voxel."""
    mitosis = graph([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, -1]], [[0, 1]])
    assert recovery(confirmer(1.0)).transform(mitosis).edges.tolist() == [[0, 10], [0, 20]]
    assert recovery(confirmer(0.0)).transform(mitosis).edges.tolist() == [[0, 10]]


def test_transform_prefers_the_nearer_mother_when_two_compete():
    """With one orphan and two eligible mothers the closer one gets the daughter."""
    contested = graph(
        [[0, 0, 0, 0], [0, 0, 0, 6], [1, 0, 0, 1], [1, 0, 0, 5], [1, 0, 0, 4]],
        [[0, 2], [1, 3]],
    )
    forked = recovery().transform(contested)
    assert forked.edges.tolist() == [[0, 20], [10, 30], [10, 40]]


def test_build():
    """The config builds a DivisionRecovery carrying its gates and this video's spacing, no centre-prior veto."""
    spacing = Spacing(z=1.0, y=1.0, x=1.0)
    stage = DivisionRecoveryConfig(parent_gate_um=4.7, sister_gate_um=7.2, existing_child_gate_um=7.8).build(spacing)
    assert isinstance(stage, DivisionRecovery)
    assert stage.parent_gate_um == 4.7
    assert stage.sister_gate_um == 7.2
    assert stage.existing_child_gate_um == 7.8
    assert stage.spacing == spacing
    assert stage.confirmer is None
