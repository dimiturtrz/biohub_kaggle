"""Equivalence classes for candidate coverage: inside the gate, outside it, and the wrong frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from celltrack.eval.candidate_coverage import CandidateCoverage, CoverageRow


def _point(frame: float, z: float, y: float, x: float) -> list[float]:
    return [frame, z, y, x]


def test_count() -> None:
    measure = CandidateCoverage(match_distance_um=7.0)
    on_cell = np.array([_point(0, 10.0, 10.0, 10.5)])
    beyond = np.array([_point(0, 10.0, 10.0, 25.0)])
    truth = np.array([_point(0, 10.0, 10.0, 10.0)])
    assert measure.count(truth, on_cell) == 1
    assert measure.count(truth, beyond) == 0
    assert measure.count(np.array([_point(0, 0.0, 0.0, 0.0)]), np.array([_point(0, 0.0, 0.0, 7.0)])) == 1


def test_count_ignores_candidates_in_another_frame() -> None:
    measure = CandidateCoverage(match_distance_um=7.0)
    truth = np.array([_point(3, 10.0, 10.0, 10.0)])
    next_frame = np.array([_point(4, 10.0, 10.0, 10.0)])
    assert measure.count(truth, next_frame) == 0


def test_count_over_a_frame_without_candidates() -> None:
    measure = CandidateCoverage(match_distance_um=7.0)
    truth = np.array([_point(0, 1.0, 1.0, 1.0), _point(1, 1.0, 1.0, 1.0)])
    found = np.array([_point(0, 1.0, 1.0, 1.0)])
    assert measure.count(truth, found) == 1


def test_count_gives_one_candidate_to_one_cell() -> None:
    measure = CandidateCoverage(match_distance_um=7.0)
    truth = np.array([_point(0, 0.0, 0.0, 0.0), _point(0, 0.0, 0.0, 100.0)])
    found = np.array([_point(0, 0.0, 0.0, 0.0)])
    assert measure.count(truth, found) == 1


def test_candidates_um(tmp_path: Path) -> None:
    npz_path = tmp_path / "movie.npz"
    np.savez(
        npz_path,
        coords=np.array([[0, 2, 3, 4]], dtype=np.int64),
        voxel_size=np.array([1.625, 1.625, 1.625]),
        scale=np.array([1.625, 0.40625, 0.40625]),
    )
    microns = CandidateCoverage().candidates_um(npz_path)
    assert microns.shape == (1, 4)
    assert microns[0, 0] == 0.0
    np.testing.assert_allclose(microns[0, 1:], [2 * 1.625, 3 * 1.625, 4 * 1.625])


def test_row(tmp_path: Path) -> None:
    npz_path = tmp_path / "movie.npz"
    np.savez(
        npz_path,
        coords=np.array([[0, 0, 0, 0], [0, 0, 0, 40]], dtype=np.int64),
        voxel_size=np.array([1.0, 1.0, 1.0]),
    )
    truth = np.array([_point(0, 0.0, 0.0, 0.0), _point(0, 0.0, 0.0, 200.0)])
    row = CandidateCoverage(match_distance_um=7.0).row("movie", truth, npz_path)
    assert (row.stem, row.gt_cells, row.covered, row.candidates) == ("movie", 2, 1, 2)


def test_recall() -> None:
    assert CoverageRow(stem="m", gt_cells=200, covered=150, candidates=4000).recall == 0.75
    assert np.isnan(CoverageRow(stem="m", gt_cells=0, covered=0, candidates=10).recall)


def test_candidates_per_gt_cell() -> None:
    row = CoverageRow(stem="m", gt_cells=200, covered=150, candidates=4000)
    assert row.candidates_per_gt_cell == 20.0
    assert np.isnan(CoverageRow(stem="m", gt_cells=0, covered=0, candidates=10).candidates_per_gt_cell)
