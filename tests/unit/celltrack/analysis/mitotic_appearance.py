import logging
from typing import cast

import numpy as np

from celltrack.analysis.mitotic_appearance import (
    FEATURE_NAMES,
    AppearanceFeatures,
    DivisionCandidates,
    DivisionPopulations,
    FeatureTable,
    MitoticAppearanceGate,
    PatchBox,
    TrueMother,
)
from core.data.tracks import Adjacency, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, NodeMatching

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)
# A coarse isotropic spacing keeps a test patch small enough to reason about voxel by voxel.
_UNIT = Spacing(z=1.0, y=1.0, x=1.0)


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph over consecutively-numbered node ids, `t, z, y, x` per node."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def _box(radius_um: float = 4.0) -> PatchBox:
    """A small isotropic box, so a patch is a handful of voxels rather than thousands."""
    return PatchBox.of(_UNIT, patch_radius_um=radius_um)


def _table(values: list[list[float]], rows: list[int]) -> FeatureTable:
    """A feature table with the given rows and a full feature vector each."""
    return FeatureTable(rows=np.asarray(rows, dtype=np.int64), values=np.asarray(values, dtype=np.float64))


def _constant_table(feature: str, values: list[float]) -> FeatureTable:
    """A table where only `feature` varies and everything else is zero."""
    matrix = np.zeros((len(values), len(FEATURE_NAMES)))
    matrix[:, FEATURE_NAMES.index(feature)] = values
    return FeatureTable(rows=np.arange(len(values), dtype=np.int64), values=matrix)


class _Video:
    """A stand-in for `CellVideo` that serves prepared volumes by timepoint."""

    def __init__(self, volumes: dict[int, np.ndarray]) -> None:
        self._volumes = volumes

    def frame(self, timepoint: int) -> np.ndarray:
        return self._volumes[timepoint]


def test_patch_box_of() -> None:
    box = PatchBox.of(_SPACING, patch_radius_um=6.5)

    assert box.half_voxels.tolist() == [4, 16, 16]
    assert box.core.shape == (9, 33, 33)
    assert box.core.sum() < box.background.size


def test_patch_box_of_masks_are_physical_not_voxel_shaped() -> None:
    box = PatchBox.of(_SPACING, patch_radius_um=6.5)
    centre = tuple(int(half) for half in box.half_voxels)

    assert box.core[centre]
    assert not box.background[centre]
    # 4 voxels of z is 6.5 um; 4 voxels of y is 1.625 um — an anisotropic box, an isotropic mask.
    assert not box.core[0, centre[1], centre[2]]
    assert box.core[centre[0], centre[1] - 4, centre[2]]


def test_voxel_volume_um3() -> None:
    assert np.isclose(PatchBox.of(_SPACING).voxel_volume_um3(), 1.625 * 0.40625 * 0.40625)


def test_cut() -> None:
    box = _box(radius_um=1.0)
    volume = np.arange(5 * 5 * 5, dtype=np.uint16).reshape(5, 5, 5)

    patch = box.cut(volume, np.array([2, 2, 2]))

    assert patch.shape == (3, 3, 3)
    assert patch[1, 1, 1] == volume[2, 2, 2]


def test_cut_pads_at_the_volume_border() -> None:
    box = _box(radius_um=1.0)
    volume = np.full((3, 3, 3), 7, dtype=np.uint16)

    patch = box.cut(volume, np.array([0, 0, 0]))

    assert patch.shape == (3, 3, 3)
    assert np.all(patch == 7.0)


def test_appearance_features_of() -> None:
    box = _box()
    patch = np.zeros(box.core.shape)
    patch[box.core] = 10.0

    features = AppearanceFeatures.of(patch, box)
    values = dict(zip(FEATURE_NAMES, features.values, strict=True))

    assert values["intensity_mean"] == 10.0
    assert values["intensity_max"] == 10.0
    assert values["local_contrast"] == 10.0
    assert np.isclose(values["centroid_offset_um"], 0.0)


def test_appearance_features_of_reports_an_offset_centroid() -> None:
    box = _box()
    patch = np.zeros(box.core.shape)
    centre = tuple(int(half) for half in box.half_voxels)
    patch[centre[0], centre[1], centre[2] + 2] = 100.0

    values = AppearanceFeatures.of(patch, box).values

    assert values[FEATURE_NAMES.index("centroid_offset_um")] > 1.0


def test_appearance_features_of_on_an_empty_patch() -> None:
    box = _box()

    values = AppearanceFeatures.of(np.zeros(box.core.shape), box).values

    assert values[FEATURE_NAMES.index("centroid_offset_um")] == 0.0
    assert values[FEATURE_NAMES.index("elongation")] == 1.0


def test_feature_table_measure() -> None:
    box = _box(radius_um=1.0)
    graph = _graph([[0, 2, 2, 2], [1, 2, 2, 2]], [])
    volumes = {0: np.zeros((5, 5, 5), dtype=np.uint16), 1: np.full((5, 5, 5), 9, dtype=np.uint16)}

    table = FeatureTable.measure(cast(CellVideo, _Video(volumes)), graph, np.array([0, 1]), box)

    assert table.rows.tolist() == [0, 1]
    assert table.column("intensity_mean").tolist() == [0.0, 9.0]


def test_column() -> None:
    table = _constant_table("volume_um3", [1.0, 2.0])

    assert table.column("volume_um3").tolist() == [1.0, 2.0]


def test_quartiles() -> None:
    table = _constant_table("intensity_mean", [0.0, 1.0, 2.0, 3.0, 4.0])

    assert table.quartiles("intensity_mean") == (1.0, 2.0, 3.0)


def test_quartiles_on_an_empty_population() -> None:
    assert all(np.isnan(value) for value in _constant_table("intensity_mean", []).quartiles("intensity_mean"))


def test_percentile_of() -> None:
    table = _constant_table("intensity_max", [0.0, 1.0, 2.0, 3.0])

    assert table.percentile_of("intensity_max", 2.5) == 75.0
    assert table.percentile_of("intensity_max", -1.0) == 0.0


def test_percentile_of_splits_ties() -> None:
    table = _constant_table("intensity_max", [0.0, 1.0, 1.0, 2.0])

    assert table.percentile_of("intensity_max", 1.0) == 50.0


def test_of_row() -> None:
    table = _table([[0.0] * len(FEATURE_NAMES), [5.0] * len(FEATURE_NAMES)], rows=[3, 9])

    assert table.of_row(9).tolist() == [5.0] * len(FEATURE_NAMES)


def test_true_mother_of() -> None:
    coordinates = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 1], [3, 0, 0, 2]]
    truth = _graph(coordinates, [[0, 1], [1, 2], [2, 3], [2, 4]])
    inverse = np.array([10, 11, 12, 13, 14], dtype=np.int64)

    mother = TrueMother.of(truth, Adjacency.of(truth), inverse, truth_row=2)

    assert mother.truth_id == 2
    assert mother.prediction_row == 12
    assert mother.run_up_rows == (11, 10)


def test_true_mother_of_skips_unmatched_run_up_frames() -> None:
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 1], [2, 0, 0, 2]], [[0, 1], [1, 2], [1, 3]])
    inverse = np.array([7, UNMATCHED, 8, 9], dtype=np.int64)

    mother = TrueMother.of(truth, Adjacency.of(truth), inverse, truth_row=1)

    assert mother.run_up_rows == (7,)


def test_measured_rows() -> None:
    mother = TrueMother(truth_id=1, prediction_row=4, run_up_rows=(3, 2))

    assert mother.measured_rows() == (4, 3, 2)


def test_division_populations_of() -> None:
    prediction = _graph(
        [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 9], [1, 0, 0, 9], [2, 0, 0, 9]],
        [[0, 1], [1, 2], [3, 4], [4, 5]],
    )
    truth = _graph([[1, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 1]], [[0, 1], [0, 2]])
    matching = NodeMatching(gt_rows=np.array([UNMATCHED, 0, 1, UNMATCHED, UNMATCHED, 2], dtype=np.int64))

    populations = DivisionPopulations.of(prediction, truth, matching, np.array([4], dtype=np.int64))

    assert [mother.prediction_row for mother in populations.mothers] == [1]
    assert populations.false_candidates.tolist() == [4]
    assert populations.annotated_others.tolist() == [2, 5]


def test_division_populations_of_excludes_mothers_from_the_candidates() -> None:
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 4]], [[0, 1], [0, 2]])
    matching = NodeMatching(gt_rows=np.array([0, 1, 2], dtype=np.int64))

    populations = DivisionPopulations.of(prediction, truth, matching, np.array([0], dtype=np.int64))

    assert populations.false_candidates.tolist() == []


class _Affinity:
    """A fixed one-gap probability matrix, in the linker's ascending row order."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._matrix if timepoint == 0 else None


def test_division_candidates_of() -> None:
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])
    affinity = _Affinity(np.array([[0.9, 0.6]]))

    candidates = DivisionCandidates.of(prediction, _SPACING, affinity, floor=0.5)

    assert candidates.rows.tolist() == [0]


def test_division_candidates_of_below_the_floor() -> None:
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 4]], [[0, 1]])
    affinity = _Affinity(np.array([[0.9, 0.1]]))

    assert DivisionCandidates.of(prediction, _SPACING, affinity, floor=0.5).rows.tolist() == []


def _gate(mother_value: float, candidate_values: list[float], annotated_values: list[float]) -> MitoticAppearanceGate:
    """A gate whose tables are prepared directly, so the report is exercised without mounting a tracker."""
    mothers = (TrueMother(truth_id=1, prediction_row=0, run_up_rows=()),)
    return MitoticAppearanceGate(
        populations=DivisionPopulations(
            mothers=mothers,
            false_candidates=np.arange(len(candidate_values), dtype=np.int64),
            linked_baseline=np.empty(0, dtype=np.int64),
            annotated_others=np.arange(len(annotated_values), dtype=np.int64),
        ),
        tables={
            "mothers+run-up": _constant_table("intensity_max", [mother_value]),
            "false candidates": _constant_table("intensity_max", candidate_values),
            "linked baseline": _constant_table("intensity_max", candidate_values),
            "annotated others": _constant_table("intensity_max", annotated_values),
        },
    )


def test_report(caplog: object) -> None:
    gate = _gate(mother_value=10.0, candidate_values=[0.0, 1.0, 2.0], annotated_values=[0.0, 1.0])

    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        gate.report()

    messages = caplog.text  # type: ignore[attr-defined]
    assert "intensity_max" in messages
    assert "mother id=1" in messages


def test_report_names_a_separating_feature(caplog: object) -> None:
    gate = _gate(mother_value=10.0, candidate_values=[0.0, 1.0, 2.0], annotated_values=[0.0, 1.0])

    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        gate.report()

    assert "'intensity_max'" in caplog.text  # type: ignore[attr-defined]


def test_report_needs_every_mother_in_the_tail_not_just_the_last(caplog: object) -> None:
    mothers = (
        TrueMother(truth_id=1, prediction_row=0, run_up_rows=()),
        TrueMother(truth_id=2, prediction_row=1, run_up_rows=()),
    )
    controls = _constant_table("intensity_max", [0.0, 1.0, 2.0])
    gate = MitoticAppearanceGate(
        populations=DivisionPopulations(
            mothers=mothers,
            false_candidates=np.arange(3, dtype=np.int64),
            linked_baseline=np.empty(0, dtype=np.int64),
            annotated_others=np.arange(3, dtype=np.int64),
        ),
        tables={
            "mothers+run-up": _constant_table("intensity_max", [1.0, 10.0]),
            "false candidates": controls,
            "linked baseline": controls,
            "annotated others": controls,
        },
    )

    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        gate.report()

    assert "BOTH controls: NONE" in caplog.text  # type: ignore[attr-defined]


def test_report_reports_no_separator_when_the_mother_is_typical(caplog: object) -> None:
    gate = _gate(mother_value=1.0, candidate_values=[0.0, 1.0, 2.0], annotated_values=[0.0, 1.0, 2.0])

    with caplog.at_level(logging.INFO):  # type: ignore[attr-defined]
        gate.report()

    assert "BOTH controls: NONE" in caplog.text  # type: ignore[attr-defined]
