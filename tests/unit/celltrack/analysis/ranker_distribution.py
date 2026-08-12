import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from celltrack.affinity import EdgeAffinity
from celltrack.analysis.ranker_distribution import (
    DriftReport,
    FeatureDrift,
    FeaturePopulation,
    RankerDriftDiagnosis,
    RankerOutput,
    ScoredMovie,
)
from celltrack.edges.association_ranker import AssociationContext, AssociationRanker, _RankerMLP
from celltrack.linkers.linkers import LinkerConfig
from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing

_UNIT = Spacing(1.0, 1.0, 1.0)
_GATE_UM = 10.0
_NAMES = ("edge_dist_um", "candidate_count", "edge_prob")


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph whose node ids are its row indices, so edge ids and rows coincide for the fixture."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.array(coordinates, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )


def _detections() -> TrackGraph:
    """Two t0 detections and three t1 ones, spread along x so the 10 um gate admits a known candidate list."""
    return _graph([[0, 0, 0, 0], [0, 0, 0, 20], [1, 0, 0, 2], [1, 0, 0, 8], [1, 0, 0, 21]], [])


def _ranker(mean: list[float] | None = None, std: list[float] | None = None) -> AssociationRanker:
    """A ranker over a three-feature subset of the manifest — `matrix` is manifest-ORDERED, not manifest-complete."""
    return AssociationRanker(
        module=_RankerMLP(len(_NAMES), 4, 4, 0.0),
        feature_names=_NAMES,
        mean=np.array(mean if mean is not None else [0.0] * len(_NAMES), dtype=np.float64),
        std=np.array(std if std is not None else [1.0] * len(_NAMES), dtype=np.float64),
    )


def _population(values: list[list[float]], groups: int) -> FeaturePopulation:
    """A population stated directly, so a statistic is checked against hand arithmetic."""
    return FeaturePopulation(names=_NAMES, values=np.array(values, dtype=np.float64), groups=groups)


def _uniform(rows: int, groups: int) -> FeaturePopulation:
    """A population of a chosen size and source count — for the guard, whose only inputs are those two."""
    return FeaturePopulation(names=_NAMES, values=np.zeros((rows, len(_NAMES)), dtype=np.float64), groups=groups)


def _drift(ours_mean: float, ours_std: float, training_mean: float, training_std: float) -> FeatureDrift:
    """One feature's drift stated directly."""
    return FeatureDrift(
        name="edge_dist_um",
        ours_mean=ours_mean,
        ours_std=ours_std,
        training_mean=training_mean,
        training_std=training_std,
    )


class _Affinity:
    """A fixed per-gap probability matrix indexed by timepoint — the edge scorer's `EdgeAffinity` contract."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


@dataclass(frozen=True)
class _ScoredRun:
    """A tracker pass's detections and affinity — what `run_scored` hands the diagnosis."""

    detections: TrackGraph
    affinity: _Affinity


@dataclass(frozen=True)
class _Video:
    """The two per-video facts the diagnosis reads off a `CellVideo`."""

    spacing: Spacing
    volume_shape: tuple[int, int, int]


class _FakeTracker:
    """A tracker that only has to fuse the affinity — the one thing `ScoredMovie.of` asks a mounted tracker for."""

    def __init__(self, fused: EdgeAffinity | None = None) -> None:
        self.fused = fused

    def fused_affinities(self, path: Path, nodes: TrackGraph, affinity: EdgeAffinity | None) -> EdgeAffinity | None:
        return self.fused


def _movie(detections: TrackGraph | None = None) -> ScoredMovie:
    """A tracker pass over the fixture detections, already unpacked."""
    return ScoredMovie(
        detections=detections if detections is not None else _detections(),
        affinity=None,
        mutual=None,
        spacing=_UNIT,
        volume_shape=(4, 32, 32),
    )


def _diagnosis() -> RankerDriftDiagnosis:
    """The diagnosis under a geometry-only linker, so a context graph exists without a mounted edge head."""
    return RankerDriftDiagnosis(linker=LinkerConfig(name="nn", gate_um=_GATE_UM))


def test_feature_population_of():
    """The gated rows of every gap, built through the SHIPPED feature code, with each source's list counted.

    Detections 0 (t0, x=0) and 1 (t0, x=20) face t1 rows 2 (x=2), 3 (x=8) and 4 (x=21). At the 10 um gate
    source 0 chooses between 2 and 3, source 1 sees only 4 — three rows over two sources, row-major over the
    gap grid, and `candidate_count` reads 2, 2, 1 accordingly.
    """
    context = AssociationContext.of(_UNIT, _detections(), _detections().without_edges(), None)

    population = FeaturePopulation.of(_ranker(), context, _GATE_UM)

    assert population.names == _NAMES
    assert population.size() == 3
    assert population.groups == 2
    assert population.values[:, 0].tolist() == [2.0, 8.0, 1.0]  # edge_dist_um
    assert population.values[:, 1].tolist() == [2.0, 2.0, 1.0]  # candidate_count
    assert population.values[:, 2].tolist() == [0.0, 0.0, 0.0]  # edge_prob: no primary links, no affinity


def test_feature_population_of_truncates_to_the_requested_gaps():
    """`max_gaps` reads only the first N frame gaps — a cheap read on a long movie, not a different quantity."""
    detections = _graph([[0, 0, 0, 0], [1, 0, 0, 1], [2, 0, 0, 2]], [])
    context = AssociationContext.of(_UNIT, detections, detections.without_edges(), None)

    both = FeaturePopulation.of(_ranker(), context, _GATE_UM)
    first = FeaturePopulation.of(_ranker(), context, _GATE_UM, max_gaps=1)

    assert both.size() == 2
    assert first.size() == 1


def test_pooled():
    """`pooled` concatenates rows and source counts; a differing feature order is refused, not silently permuted."""
    pooled = FeaturePopulation.pooled([_population([[1.0, 2.0, 3.0]], groups=1), _population([[4.0, 5.0, 6.0]], 1)])

    assert pooled.values.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert pooled.groups == 2

    permuted = FeaturePopulation(names=("candidate_count",), values=np.zeros((1, 1)), groups=1)
    with pytest.raises(ValueError, match="different feature orders"):
        FeaturePopulation.pooled([_population([[1.0, 2.0, 3.0]], 1), permuted])
    with pytest.raises(ValueError, match="at least one movie"):
        FeaturePopulation.pooled([])


def test_size():
    """`size` is the number of gated candidate rows — the artifact's "rows"."""
    assert _population([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], groups=1).size() == 2
    assert _uniform(rows=0, groups=0).size() == 0


def test_rows_per_source():
    """Candidates per source list — the artifact's rows/groups, and NaN when no source contributed at all."""
    assert _uniform(rows=6, groups=2).rows_per_source() == 3.0
    assert math.isnan(_uniform(rows=0, groups=0).rows_per_source())


def test_is_reportable():
    """The guard: too few rows, or lists of one candidate, earn no drift table."""
    assert _uniform(rows=1000, groups=400).is_reportable()  # 2.5 per source, plenty of rows
    assert not _uniform(rows=1000, groups=1000).is_reportable()  # one candidate per source: no choice posed
    assert not _uniform(rows=100, groups=20).is_reportable()  # a mean over 100 rows is mostly sampling error


def test_moments():
    """Our per-feature mean and std, and NaN (not zero) on an empty population — the pair the table compares."""
    mean, std = _population([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]], groups=1).moments()

    assert mean.tolist() == [2.0, 4.0, 6.0]
    assert std.tolist() == [1.0, 2.0, 2.0]
    assert all(math.isnan(value) for value in _uniform(rows=0, groups=0).moments()[0])


def test_table():
    """Each feature paired with its own training moments by manifest index, worst standardised gap first."""
    population = _population([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], groups=1)

    table = FeatureDrift.table(population, _ranker(mean=[1.0, 0.0, 3.0], std=[1.0, 1.0, 1.0]))

    assert [drift.name for drift in table] == ["candidate_count", "edge_dist_um", "edge_prob"]
    assert table[0].z() == 2.0  # candidate_count: ours 2.0 against a training mean of 0.0, one std wide
    assert table[0].training_std == 1.0
    assert table[1].z() == 0.0


def test_z():
    """The gap between the means in TRAINING stds — the net's own input scale, and floored against a zero std."""
    assert _drift(ours_mean=7.0, ours_std=1.0, training_mean=3.0, training_std=2.0).z() == 2.0
    assert _drift(ours_mean=3.0, ours_std=1.0, training_mean=3.0, training_std=0.0).z() == 0.0


def test_spread():
    """Our std as a multiple of theirs — 1 is the same random variable, 0 a column that does not vary."""
    assert _drift(ours_mean=0.0, ours_std=1.0, training_mean=0.0, training_std=4.0).spread() == 0.25


def test_is_off_distribution():
    """A column is off-distribution by LOCATION or by SPREAD — a matching mean does not excuse a dead column."""
    assert not _drift(ours_mean=3.2, ours_std=1.9, training_mean=3.0, training_std=2.0).is_off_distribution()
    assert _drift(ours_mean=6.0, ours_std=2.0, training_mean=3.0, training_std=2.0).is_off_distribution()
    assert _drift(ours_mean=3.0, ours_std=0.0, training_mean=3.0, training_std=2.0).is_off_distribution()
    assert _drift(ours_mean=3.0, ours_std=9.0, training_mean=3.0, training_std=2.0).is_off_distribution()


def test_feature_drift_line():
    """The row carries both distributions, the standardised gap, and a flag only when the column is off."""
    off = _drift(ours_mean=6.0, ours_std=2.0, training_mean=3.0, training_std=2.0).line()

    assert "edge_dist_um" in off
    assert "z=" in off and "spread=" in off
    assert off.strip().startswith("!!")
    assert not _drift(3.0, 2.0, 3.0, 2.0).line().strip().startswith("!!")


def test_ranker_output_of():
    """Scoring the population's rows exactly as the linker's cost term does — one probability per row."""
    population = _population([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], groups=1)

    output = RankerOutput.of(_ranker(), population)

    assert output.probability.shape == (2,)
    assert bool(np.all((output.probability > 0.0) & (output.probability < 1.0)))


def test_decisive_fraction():
    """The share the net calls a successor — NaN, not zero, when nothing was scored."""
    assert RankerOutput(probability=np.array([0.1, 0.6, 0.9])).decisive_fraction() == pytest.approx(2 / 3)
    assert math.isnan(RankerOutput(probability=np.empty(0)).decisive_fraction())


def test_ranker_output_line():
    """The output distribution as quantiles plus the decisive share; an empty population says so instead."""
    line = RankerOutput(probability=np.linspace(0.0, 1.0, 11)).line()

    assert "p00=0.0000" in line and "p100=1.0000" in line and "p50=0.5000" in line
    assert "frac>0.5=" in line
    assert "no rows scored" in RankerOutput(probability=np.empty(0)).line()


def test_lines():
    """The counts print first, then the table — and a degenerate population is refused with the defect named."""
    ranker = _ranker(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    healthy = np.tile(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]), (500, 1))

    lines = DriftReport(label="movie", population=FeaturePopulation(_NAMES, healthy, 400), ranker=ranker).lines()

    assert lines[0].startswith("movie: 1000 gated candidate rows over 400 sources")
    assert "the artifact trained on" in lines[0]
    assert any("edge_dist_um" in line for line in lines)
    assert any(line.startswith("  off-distribution (") for line in lines)
    assert any("P(ranker):" in line for line in lines)

    thin = DriftReport(label="movie", population=_uniform(rows=100, groups=50), ranker=ranker).lines()
    assert "REFUSED" in thin[1] and "sampling error" in thin[1]
    assert not any("z=" in line for line in thin)

    choiceless = DriftReport(label="movie", population=_uniform(rows=1000, groups=1000), ranker=ranker).lines()
    assert "REFUSED" in choiceless[1] and "candidates per source" in choiceless[1]


def test_pass_one():
    """The diagnosis reads the arm's real context: the tracker's own `without_ranker`, budget and all.

    The population this module characterises is defined by the cost pass 1 links under, so it must be the
    tracker's rule and not a local paraphrase of it — the whole learned-evidence budget on the affinity, which
    is what makes the population invariant to the split under test (bead ic44).
    """
    diagnosis = RankerDriftDiagnosis(linker=LinkerConfig(ranker_bonus=17.0, affinity_bonus=3.0))

    pass_one = diagnosis.pass_one()

    assert pass_one.ranker_bonus is None
    assert pass_one.affinity_bonus == 20.0  # the split's own budget, not the arm's leftover affinity share
    assert not pass_one.needs_ranker


def test_scored_movie_of(monkeypatch: pytest.MonkeyPatch):
    """Unpacking a tracker pass reads the video's geometry and fuses the affinity by the TRACKER's own rule."""
    monkeypatch.setattr(
        CellVideo, "from_ome_zarr", classmethod(lambda cls, path: _Video(spacing=_UNIT, volume_shape=(4, 32, 32)))
    )
    forward = _Affinity({0: np.zeros((2, 3), dtype=np.float64)})
    fused = _Affinity({0: np.ones((2, 3), dtype=np.float64)})

    movie = ScoredMovie.of(_FakeTracker(fused), Path("m.zarr"), _detections(), forward)

    assert movie.affinity is forward
    assert movie.mutual is fused  # never re-derived here; whatever the tracker's rule returns
    assert movie.spacing == _UNIT
    assert movie.volume_shape == (4, 32, 32)
    assert ScoredMovie.of(_FakeTracker(), Path("m.zarr"), _detections(), forward).mutual is None


def test_of_movie():
    """One movie's rows, with the context linked by the arm's own pass-1 linker over the pass's detections."""
    population = _diagnosis().of_movie(_ranker(), _movie())

    assert population.size() == 3
    assert population.groups == 2
    assert population.values[:, 0].tolist() == [2.0, 8.0, 1.0]


def test_by_movie():
    """Every movie's population keyed by stem — one linking pass per movie, one mounted ranker."""
    by_movie = _diagnosis().by_movie(_ranker(), {"m": _movie(), "n": _movie()})

    assert list(by_movie) == ["m", "n"]
    assert by_movie["m"].size() == 3


def test_feature_population_of_skips_a_gap_with_no_partner_frame_and_one_with_no_in_gate_pair():
    """A gap missing its target frame, and one whose every pair is outside the gate, contribute no rows.

    Both are silent in the shipped ranker too (it emits no matrix for either), so the population must not
    invent a row — or a source — for a gap the pipeline never scores.
    """
    orphan = _graph([[0, 0, 0, 0], [2, 0, 0, 0]], [])  # t0 and t2: the t0 -> t1 gap has no targets
    far = _graph([[0, 0, 0, 0], [1, 0, 0, 99]], [])  # a single pair, 99 um apart at a 10 um gate

    for detections in (orphan, far):
        context = AssociationContext.of(_UNIT, detections, detections.without_edges(), None)
        population = FeaturePopulation.of(_ranker(), context, _GATE_UM)
        assert population.size() == 0
        assert population.groups == 0
