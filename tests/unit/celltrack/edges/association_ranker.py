import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from celltrack.edges.association_ranker import (
    AssociationContext,
    AssociationFeatures,
    AssociationRanker,
    _RankerMLP,
)
from core.data.tracks import TrackGraph
from core.geometry import Spacing

SPACING = Spacing(z=2.0, y=1.0, x=1.0)

FEATURES = (
    "edge_prob",
    "has_learned_edge",
    "source_out_degree",
    "target_in_degree",
    "source_frame_count",
    "target_frame_count",
    "source_density_7um",
    "target_density_7um",
    "candidate_rank_dist",
    "candidate_count",
    "edge_dz_um",
    "edge_dy_um",
    "edge_dx_um",
    "edge_dist_um",
    "edge_xy_um",
    "edge_abs_z_um",
    "motion_dist_um",
    "motion_gain_um",
    "source_has_prev",
    "target_has_next",
    "target_best_next_prob",
    "t_norm",
)

# Five detections over three frames, chosen so every feature has a hand-checkable value under SPACING:
#   t=0: 10 at (0,0,0) um and 11 at (0,0,5) um  — 5 um apart, so each is the other's 7 um neighbour
#   t=1: 12 at (0,3,0) um and 13 at (0,0,6) um  — 6.708 um apart, likewise
#   t=2: 14 at (0,6,0) um
# The context links are 10->12 and 12->14, a straight track through the crowd.
COORDINATES = np.array([[0, 0, 0, 0], [0, 0, 0, 5], [1, 0, 3, 0], [1, 0, 0, 6], [2, 0, 6, 0]], dtype=np.int64)
NODE_IDS = np.array([10, 11, 12, 13, 14], dtype=np.int64)
LINKS = np.array([[10, 12], [12, 14]], dtype=np.int64)


class _Affinity:
    """The per-gap learned probabilities, in the ascending node order the gap matrices are aligned to."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


def _detections() -> TrackGraph:
    return TrackGraph(node_ids=NODE_IDS, coordinates=COORDINATES, edges=np.empty((0, 2), dtype=np.int64))


def _links() -> TrackGraph:
    return TrackGraph(node_ids=NODE_IDS, coordinates=COORDINATES, edges=LINKS)


def _affinity() -> _Affinity:
    return _Affinity({0: np.array([[0.8, 0.1], [0.05, 0.7]]), 1: np.array([[0.9], [0.2]])})


def _context() -> AssociationContext:
    return AssociationContext.of(SPACING, _detections(), _links(), _affinity())


def _artifact(tmp_path: Path, module: _RankerMLP, mean: list[float], std: list[float]) -> Path:
    """A minimal artifact directory with the real manifest layout — the shape `from_artifact` reads."""
    (tmp_path / "model").mkdir()
    torch.save(
        {"state_dict": module.state_dict(), "config": {"dropout": 0.05}, "mean": mean, "std": std},
        tmp_path / "model/ranker.pt",
    )
    manifest = {"model": {"feature_columns": list(FEATURES), "path": "model/ranker.pt"}}
    (tmp_path / "ASSOCIATION_RANKER_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_association_context_of():
    """Degrees, best-next probability and the motion prediction all come from the links being re-ranked."""
    context = _context()

    assert context.in_degree.tolist() == [0, 0, 1, 0, 1]
    assert context.out_degree.tolist() == [1, 0, 1, 0, 0]
    assert context.best_next_prob.tolist() == [0.8, 0.0, 0.9, 0.0, 0.0]
    # 12 has parent 10, so it predicts a half-step further along that displacement; the rest have no parent.
    assert context.predicted_um.tolist() == [[0, 0, 0], [0, 0, 5], [0, 4.5, 0], [0, 0, 6], [0, 7.5, 0]]


def test_context_counts_frame_size_and_seven_micrometre_crowding():
    """`*_frame_count` is the frame's detection count; `*_density_7um` its neighbours within 7 um, minus self."""
    context = _context()

    assert context.frame_count.tolist() == [2, 2, 2, 2, 1]
    assert context.density.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert context.max_timepoint == 2


def test_planes():
    """Every one of the 22 columns of one gap, computed by hand and compared element by element.

    The `1 -> 2` gap is the informative one: its sources carry a parent (so the motion prediction is a real
    extrapolation, not the degenerate own-position case), a selected primary link, and a non-zero `t_norm`.
    """
    features = AssociationFeatures.of(_context(), np.array([2, 3]), np.array([4]), 1, 12.0)

    planes = features.planes()
    row_12, row_13 = 0, 1
    diagonal = math.sqrt(72.0)  # 13 -> 14 spans (0, 6, -6) um
    expected_12 = {
        "edge_prob": 0.9,  # 12 -> 14 IS the primary link, and the affinity scored it 0.9
        "has_learned_edge": 1.0,
        "source_out_degree": 1.0,
        "target_in_degree": 1.0,
        "source_frame_count": 2.0,
        "target_frame_count": 1.0,
        "source_density_7um": 1.0,
        "target_density_7um": 0.0,
        "candidate_rank_dist": 1.0,  # the only candidate, so first of one
        "candidate_count": 1.0,
        "edge_dz_um": 0.0,
        "edge_dy_um": 3.0,
        "edge_dx_um": 0.0,
        "edge_dist_um": 3.0,
        "edge_xy_um": 3.0,
        "edge_abs_z_um": 0.0,
        "motion_dist_um": 1.5,  # predicted (0, 4.5, 0) is 1.5 um short of the target
        "motion_gain_um": 1.5,  # 3.0 raw - 1.5 motion: the velocity model explains half the gap
        "source_has_prev": 1.0,
        "target_has_next": 0.0,
        "target_best_next_prob": 0.0,
        "t_norm": 0.5,
    }
    expected_13 = {
        "edge_prob": 0.0,  # 13 -> 14 is not a primary link, so it carries no learned probability
        "has_learned_edge": 0.0,
        "source_out_degree": 0.0,
        "target_in_degree": 1.0,
        "source_frame_count": 2.0,
        "target_frame_count": 1.0,
        "source_density_7um": 1.0,
        "target_density_7um": 0.0,
        "candidate_rank_dist": 1.0,
        "candidate_count": 1.0,
        "edge_dz_um": 0.0,
        "edge_dy_um": 6.0,
        "edge_dx_um": -6.0,
        "edge_dist_um": diagonal,
        "edge_xy_um": diagonal,
        "edge_abs_z_um": 0.0,
        "motion_dist_um": diagonal,  # no parent, so the prediction is its own position
        "motion_gain_um": 0.0,
        "source_has_prev": 0.0,
        "target_has_next": 0.0,
        "target_best_next_prob": 0.0,
        "t_norm": 0.5,
    }
    for name in FEATURES:
        assert planes[name][row_12, 0] == pytest.approx(expected_12[name]), name
        assert planes[name][row_13, 0] == pytest.approx(expected_13[name]), name


def test_candidate_list_features_rank_a_known_candidate_set():
    """`candidate_rank_dist` is the 1-based rank by physical distance within the source's gated candidates."""
    features = AssociationFeatures.of(_context(), np.array([0, 1]), np.array([2, 3]), 0, 12.0)

    planes = features.planes()
    # 10 is 3 um from 12 and 6 um from 13; 11 is 5.83 um from 12 and 1 um from 13 — so the orders differ.
    assert planes["candidate_rank_dist"].tolist() == [[1.0, 2.0], [2.0, 1.0]]
    assert planes["candidate_count"].tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_gated():
    """A tighter gate shortens the list the rank and count are taken over — the features follow the gate."""
    features = AssociationFeatures.of(_context(), np.array([0, 1]), np.array([2, 3]), 0, 5.0)

    planes = features.planes()
    assert features.gated().tolist() == [[True, False], [False, True]]
    assert planes["candidate_count"].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    # A pair outside the gate is in nobody's candidate list, so it holds no rank.
    assert planes["candidate_rank_dist"].tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_matrix():
    """The scored rows are exactly the gated pairs, and the columns are the order the manifest asked for."""
    features = AssociationFeatures.of(_context(), np.array([0, 1]), np.array([2, 3]), 0, 5.0)

    matrix = features.matrix(FEATURES)
    assert matrix.shape == (2, len(FEATURES))
    # Row-major over the gap: 10 -> 12 (3 um) then 11 -> 13 (1 um).
    assert matrix[:, FEATURES.index("edge_dist_um")].tolist() == [3.0, 1.0]
    assert matrix[:, FEATURES.index("edge_prob")].tolist() == [0.8, 0.0]


def test_matrix_refuses_a_feature_it_cannot_compute():
    """An unknown column is an error, never a zero fill — a silently wrong vector reads as 'the ranker fails'."""
    features = AssociationFeatures.of(_context(), np.array([0, 1]), np.array([2, 3]), 0, 12.0)

    with pytest.raises(ValueError, match="does not compute"):
        features.matrix(("edge_prob", "source_in_degree"))


def test_from_artifact(tmp_path: Path):
    """The feature contract comes from the artifact — not from the stale list every public notebook carries."""
    root = _artifact(tmp_path, _RankerMLP(22, 64, 32, 0.05), [0.0] * 22, [1.0] * 22)

    assert AssociationRanker.from_artifact(root).feature_names == FEATURES


def test_from_artifact_rejects_a_manifest_that_disagrees_with_the_weights(tmp_path: Path):
    """A feature list that does not match the first layer's width is a mismatch, not a truncation."""
    root = _artifact(tmp_path, _RankerMLP(21, 64, 32, 0.05), [0.0] * 21, [1.0] * 21)

    with pytest.raises(ValueError, match="21 inputs"):
        AssociationRanker.from_artifact(root)


def test_probabilities(tmp_path: Path):
    """Each column is centred and scaled by the artifact's own mean/std before the net sees it."""
    torch.manual_seed(0)
    module = _RankerMLP(22, 64, 32, 0.05).eval()  # the mounted ranker scores in eval mode; dropout must be off
    mean = list(np.arange(22, dtype=np.float64))
    std = [2.0] * 22
    ranker = AssociationRanker.from_artifact(_artifact(tmp_path, module, mean, std))

    row = np.arange(22, dtype=np.float64)[None, :] + 4.0
    with torch.no_grad():
        expected = torch.sigmoid(module(torch.full((1, 22), 2.0))).numpy()  # (row - mean) / std == 2 everywhere
    assert ranker.probabilities(row) == pytest.approx(expected)


def test_probabilities_floor_a_zero_variance_column(tmp_path: Path):
    """A zero-std column would divide by zero; it is floored, as the public loader floors it."""
    root = _artifact(tmp_path, _RankerMLP(22, 64, 32, 0.05), [0.0] * 22, [0.0] * 22)

    assert np.isfinite(AssociationRanker.from_artifact(root).probabilities(np.zeros((1, 22)))).all()


def test_probabilities_reject_a_feature_count_the_artifact_does_not_take(tmp_path: Path):
    """A short vector is refused rather than broadcast into a wrong-but-plausible score."""
    ranker = AssociationRanker.from_artifact(_artifact(tmp_path, _RankerMLP(22, 64, 32, 0.05), [0.0] * 22, [1.0] * 22))

    with pytest.raises(ValueError, match="the artifact takes 22"):
        ranker.probabilities(np.zeros((1, 21)))


def test_affinities(tmp_path: Path):
    """The video's gaps come back as `s x t` matrices keyed by source timepoint — the `EdgeAffinity` contract."""
    ranker = AssociationRanker.from_artifact(_artifact(tmp_path, _RankerMLP(22, 64, 32, 0.05), [0.0] * 22, [1.0] * 22))

    affinity = ranker.affinities(SPACING, _detections(), _links(), _affinity(), 5.0)

    first, second = affinity.probabilities(0), affinity.probabilities(1)
    assert first is not None and second is not None
    assert first.shape == (2, 2)
    assert second.shape == (2, 1)
    assert affinity.probabilities(2) is None  # the last frame opens no gap
    # A pair outside the gate is never in a candidate list, so it is scored 0 rather than guessed at.
    scored = first
    assert scored[0, 1] == 0.0
    assert scored[1, 0] == 0.0
    assert ((scored >= 0.0) & (scored <= 1.0)).all()


def test_rows_at():
    """A frame's detection rows come back ascending — the order every gap matrix is aligned to."""
    assert _context().rows_at(1).tolist() == [2, 3]


def test_association_features_of():
    """The gap carries the source→target displacement every geometric column is derived from."""
    features = AssociationFeatures.of(_context(), np.array([0, 1]), np.array([2, 3]), 0, 12.0)

    assert features.delta_um.shape == (2, 2, 3)
    assert features.delta_um[0, 0].tolist() == [0.0, 3.0, 0.0]  # 10 at the origin to 12 three um along y


def test_distance_um():
    """The physical pair distance — what the gate cuts on and what `edge_dist_um` reports."""
    features = AssociationFeatures.of(_context(), np.array([0, 1]), np.array([2, 3]), 0, 12.0)

    assert features.distance_um()[0].tolist() == [3.0, 6.0]


def test_forward():
    """The net emits one logit per candidate row — the shape `probabilities` turns into a probability."""
    assert _RankerMLP(22, 64, 32, 0.0).eval().forward(torch.zeros((5, 22))).shape == (5,)


def test_association_context_of_without_links_or_affinity():
    """With nothing linked yet, every context feature degenerates to 'no evidence' rather than erroring."""
    context = AssociationContext.of(SPACING, _detections(), _detections(), None)

    assert context.in_degree.tolist() == [0] * 5
    assert context.best_next_prob.tolist() == [0.0] * 5
    assert context.predicted_um.tolist() == context.positions_um.tolist()  # no parent, so no velocity
    features = AssociationFeatures.of(context, np.array([0, 1]), np.array([2, 3]), 0, 12.0)
    assert features.planes()["has_learned_edge"].tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_affinities_skip_a_gap_with_no_gated_candidate(tmp_path: Path):
    """A gap where the gate admits nothing is scored all-zero — no candidate list means nothing to rank."""
    ranker = AssociationRanker.from_artifact(_artifact(tmp_path, _RankerMLP(22, 64, 32, 0.05), [0.0] * 22, [1.0] * 22))

    affinity = ranker.affinities(SPACING, _detections(), _links(), _affinity(), 0.5)

    scored = affinity.probabilities(0)
    assert scored is not None and not scored.any()
