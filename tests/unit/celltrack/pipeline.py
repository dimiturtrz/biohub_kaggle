from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from celltrack.gap_closer import GapCloser
from celltrack.pipeline import DetectorScorer, DetectorSpec, Pipeline, PipelineConfig
from celltrack.response_cache import ResponseCache
from celltrack.short_track_filter import ShortTrackFilter
from celltrack.stages import GapSpec, ShortTrackSpec
from celltrack.tunet import DetectorRecipe, TemporalUNetDetector
from core.data.tracks import TrackGraph
from core.geometry import Spacing

SPACING = Spacing(z=2.0, y=1.0, x=1.0)


def _nodes() -> TrackGraph:
    coordinates = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int64)
    return TrackGraph(np.array([0, 1], dtype=np.int64), coordinates, np.empty((0, 2), dtype=np.int64))


@dataclass
class _RecordingScorer:
    """A stand-in scorer: returns canned nodes and remembers the threshold it was asked for."""

    graph: TrackGraph
    seen_threshold: float | None = None

    def nodes(self, video_key: str, path: Path, threshold: float) -> TrackGraph:
        self.seen_threshold = threshold
        return self.graph


@dataclass(frozen=True)
class _RecordingStage:
    """A stand-in stage that logs the order it ran in and passes the graph through untouched."""

    name: str
    log: list[str]

    def transform(self, graph: TrackGraph) -> TrackGraph:
        self.log.append(self.name)
        return graph


@dataclass
class _Order:
    log: list[str] = field(default_factory=list)


@dataclass
class _FakeDetector:
    """Duck-typed detector: counts forwards so the cache's forward-once contract can be checked off the GPU."""

    forwards: list[int] = field(default_factory=list)

    def probability_volumes(self, path: Path, recipe: DetectorRecipe, device: str) -> list[np.ndarray]:
        self.forwards.append(1)
        return [np.zeros((2, 2, 2), dtype=np.float32)]

    def voxel_scale(self, path: Path) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0)

    def graph_from_volumes(
        self,
        volumes: list[np.ndarray],
        scale: tuple[float, float, float],
        threshold: float,
        recipe: DetectorRecipe,
        device: str,
    ) -> TrackGraph:
        return _nodes()


def test_nodes(tmp_path: Path):
    """The detector scorer forwards a video once and replays the read-out from cache on later calls."""
    fake = _FakeDetector()
    scorer = DetectorScorer(
        detector=cast(TemporalUNetDetector, fake),
        recipe=DetectorRecipe(),
        cache=ResponseCache(tmp_path, "weights"),
        device="cpu",
    )
    first = scorer.nodes("vid", Path("unused.zarr"), 0.99)
    scorer.nodes("vid", Path("unused.zarr"), 0.99)
    assert fake.forwards == [1]  # forwarded once, second call hit the cache
    assert np.array_equal(first.node_ids, _nodes().node_ids)


def test_build_stages():
    """The config's ordered specs become the concrete impls in the same order — order is the pipeline."""
    config = PipelineConfig(stages=(GapSpec(), ShortTrackSpec(min_length=4)))
    assert config.build_stages(SPACING) == (GapCloser(SPACING, 5.8, 3.2, 0.05), ShortTrackFilter(4))


def test_finish():
    """`finish` folds the stages left to right — post-proc order is exactly the list order."""
    order = _Order()
    pipeline = Pipeline(
        scorer=_RecordingScorer(_nodes()),
        threshold=0.99,
        stages=(_RecordingStage("first", order.log), _RecordingStage("second", order.log)),
    )
    pipeline.finish(_nodes())
    assert order.log == ["first", "second"]


def test_detect():
    """`detect` asks the scorer for nodes at the pipeline's threshold."""
    scorer = _RecordingScorer(_nodes())
    pipeline = Pipeline(scorer=scorer, threshold=0.9, stages=())
    detected = pipeline.detect("vid", Path("unused.zarr"))
    assert scorer.seen_threshold == pytest.approx(0.9)
    assert np.array_equal(detected.node_ids, _nodes().node_ids)


def test_run():
    """`run` detects then finishes — the scorer sees the threshold and the stage runs."""
    scorer = _RecordingScorer(_nodes())
    order = _Order()
    pipeline = Pipeline(scorer=scorer, threshold=0.95, stages=(_RecordingStage("only", order.log),))
    pipeline.run("vid", Path("unused.zarr"))
    assert scorer.seen_threshold == pytest.approx(0.95)
    assert order.log == ["only"]


def test_build():
    """`build` wires a scorer to the config; the pipeline carries the detector spec's threshold and stages."""
    config = PipelineConfig(detector=DetectorSpec(threshold=0.9), stages=(ShortTrackSpec(),))
    pipeline = config.build(_RecordingScorer(_nodes()), SPACING)
    assert pipeline.threshold == pytest.approx(0.9)
    assert pipeline.stages == (ShortTrackFilter(6),)


def test_detector_threshold_out_of_bounds_is_rejected():
    with pytest.raises(ValidationError):
        DetectorSpec(threshold=1.0)


def test_config_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        PipelineConfig(unknown=1)
