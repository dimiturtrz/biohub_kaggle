from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from celltrack.training.detected_windows import (
    CandidateFrame,
    CandidateSelection,
    DetectedWindows,
    RatioMatchedSelection,
    SparseEdgeTargets,
)


def _frame(coords: npt.ArrayLike, matched: npt.ArrayLike) -> CandidateFrame:
    return CandidateFrame(
        coords=np.asarray(coords, dtype=np.float32),
        matched_gt=np.asarray(matched, dtype=np.int64),
    )


def _selection(keep: int = 10, gate_um: float = 5.0) -> CandidateSelection:
    return CandidateSelection(keep=keep, gate_um=gate_um)


def test_select():
    score = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    matched = np.array([5, -1, 7, -1], dtype=np.int64)
    # matched (0, 2) are kept regardless of score; the single fill slot takes the best unmatched (1).
    assert _selection(keep=3).select(score, matched).tolist() == [0, 1, 2]


def test_select_never_drops_a_matched_detection_for_a_higher_scoring_negative():
    score = np.array([0.01, 0.99], dtype=np.float32)
    matched = np.array([3, -1], dtype=np.int64)
    assert _selection(keep=1).select(score, matched).tolist() == [0]


def test_select_falls_back_to_score_when_matched_alone_overflows():
    score = np.array([0.2, 0.7, 0.5], dtype=np.float32)
    matched = np.array([1, 2, 3], dtype=np.int64)
    assert _selection(keep=2).select(score, matched).tolist() == [1, 2]


def test_select_keeps_everything_when_the_cap_exceeds_the_crowd():
    score = np.array([0.3, 0.4], dtype=np.float32)
    matched = np.array([-1, 8], dtype=np.int64)
    assert _selection(keep=10).select(score, matched).tolist() == [0, 1]


def test_match_to_ground_truth():
    detections = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    gt = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    ids = np.array([42], dtype=np.int64)
    # inside the gate -> the GT id; far outside -> -1, never the nearest-anyway id.
    assert _selection(gate_um=1.0).match_to_ground_truth(detections, gt, ids).tolist() == [42, -1]


def test_match_to_ground_truth_gate_is_inclusive_at_the_boundary():
    detections = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    gt = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ids = np.array([7], dtype=np.int64)
    assert _selection(gate_um=2.0).match_to_ground_truth(detections, gt, ids).tolist() == [7]


def test_frame():
    coords = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32)
    score = np.array([0.1, 0.9], dtype=np.float32)
    gt = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ids = np.array([3], dtype=np.int64)
    frame = _selection(keep=1, gate_um=1.0).frame(coords, score, coords, gt, ids)
    assert len(frame) == 1
    assert frame.matched_gt.tolist() == [3]


def test_pairs():
    source = np.array([10, -1], dtype=np.int64)
    target = np.array([-1, 20], dtype=np.int64)
    rows, cols = SparseEdgeTargets.pairs(source, target, {(10, 20)})
    assert rows.tolist() == [0]
    assert cols.tolist() == [1]


def test_pairs_is_empty_when_no_kept_pair_is_a_ground_truth_edge():
    source = np.array([10], dtype=np.int64)
    target = np.array([20], dtype=np.int64)
    rows, cols = SparseEdgeTargets.pairs(source, target, {(10, 99)})
    assert rows.tolist() == []
    assert cols.tolist() == []


def test_pairs_ignores_unmatched_detections_on_both_sides():
    source = np.array([-1, -1], dtype=np.int64)
    target = np.array([-1], dtype=np.int64)
    rows, _ = SparseEdgeTargets.pairs(source, target, {(-1, -1)})
    assert rows.tolist() == []


def test_sparse_edge_targets():
    targets = SparseEdgeTargets(
        pairs=[(np.array([0], dtype=np.int32), np.array([1], dtype=np.int32))],
        shapes=[(2, 2)],
    )
    assert torch.equal(targets[0], torch.tensor([[0.0, 1.0], [0.0, 0.0]]))


def test_sparse_edge_targets_densifies_an_all_zero_window():
    targets = SparseEdgeTargets(
        pairs=[(np.array([], dtype=np.int32), np.array([], dtype=np.int32))],
        shapes=[(3, 2)],
    )
    assert targets[0].shape == (3, 2)
    assert targets[0].sum() == 0.0


def test_sparse_edge_targets_len():
    targets = SparseEdgeTargets(pairs=[], shapes=[])
    assert len(targets) == 0


def test_sparse_edge_targets_iter():
    pair = (np.array([0], dtype=np.int32), np.array([0], dtype=np.int32))
    targets = SparseEdgeTargets(pairs=[pair, pair], shapes=[(1, 1), (1, 1)])
    assert [t.item() for t in targets] == [1.0, 1.0]


def test_candidate_frame_len():
    assert len(_frame([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [-1, 4])) == 2


@dataclass(frozen=True)
class _Window:
    t_start: int
    n_frames: int
    stamped: list[np.ndarray]
    coords: list[torch.Tensor]
    node_counts: list[int]
    targets: SparseEdgeTargets


def _windows(window_size: int = 2) -> DetectedWindows[_Window]:
    return DetectedWindows(window_size=window_size, window_factory=_Window)


def test_build():
    frames = [_frame([[0.0, 0.0, 0.0]], [1]), _frame([[1.0, 0.0, 0.0]], [2])]
    windows = _windows().build(frames, {(1, 2)})
    assert len(windows) == 1
    assert windows[0].node_counts == [1, 1]
    assert windows[0].targets[0].item() == 1.0


def test_build_skips_a_span_with_an_empty_frame():
    frames = [_frame([[0.0, 0.0, 0.0]], [1]), _frame(np.zeros((0, 3)), [])]
    assert _windows().build(frames, set()) == []


def test_build_skips_a_span_with_a_missing_frame():
    frames = [_frame([[0.0, 0.0, 0.0]], [1]), None]
    assert _windows().build(frames, set()) == []


def test_build_stamps_each_frame_with_its_own_timepoint():
    frames = [_frame([[0.0, 0.0, 0.0]], [1]), _frame([[1.0, 0.0, 0.0]], [2])]
    windows = _windows().build(frames, set())
    assert windows[0].stamped[0][0, 0] == 0.0
    assert windows[0].stamped[1][0, 0] == 1.0


def test_carrying():
    base = RatioMatchedSelection(keep=4, gate_um=7.0)
    edges = {(1, 2), (2, 3), (4, 5), (5, 6)}
    ids = np.array([1, 2, 3, 4, 5, 6])
    carried = base.carrying(ids, edges, tracks_carried=1, seed=0).carried_gt_ids
    assert len(carried) == 3
    assert carried in ({1, 2, 3}, {4, 5, 6})
    assert base.carrying(ids, edges, tracks_carried=1, seed=0).carried_gt_ids == carried
    assert base.carrying(ids, edges, tracks_carried=2, seed=0).carried_gt_ids == {1, 2, 3, 4, 5, 6}


def test_track_components():
    labels = RatioMatchedSelection(keep=4, gate_um=7.0).track_components(np.array([1, 2, 3, 4]), {(1, 2), (2, 3)})
    assert labels[1] == labels[2] == labels[3]
    assert labels[4] != labels[1]


def test_ratio_matched_select_drops_uncarried_cells_instead_of_demoting_them():
    selection = RatioMatchedSelection(keep=4, gate_um=7.0, carried_gt_ids=frozenset({7}))
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    matched = np.array([3, 7, -1, -1, -1])
    kept = selection.select(score, matched)
    assert kept.tolist() == [1, 2, 3, 4]
