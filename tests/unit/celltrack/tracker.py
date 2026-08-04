from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celltrack import tracker as tracker_module
from celltrack.linking import Linker
from celltrack.pipeline import BlendDetectorScorer
from celltrack.tracker import CellTracker, LinkerStage, TrackerConfig
from celltrack.tunet import DetectorRecipe
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_CHAIN = TrackGraph(
    node_ids=np.array([0, 1], dtype=np.int64),
    coordinates=np.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.int64),
    edges=np.empty((0, 2), dtype=np.int64),
)


@dataclass
class _Video:
    spacing: Spacing


class _StubDetector:
    """A detector whose forward is faked, so `from_packs` mounts without weights or a GPU."""

    def to(self, device: str) -> "_StubDetector":
        return self

    def eval(self) -> "_StubDetector":
        return self


class _StubBlend:
    """A blend detector that returns a fixed two-cell chain, so the linker + post-proc run without a forward."""

    def nodes(self, video_key: str, path: Path, threshold: float) -> TrackGraph:
        return _CHAIN


class _StubEdgeScorer:
    """A blended edge scorer supplying no learned affinity, leaving the linker on pure geometry."""

    def affinities(self, path: Path, nodes: TrackGraph, device: str) -> None:
        return None


def _tracker(monkeypatch: pytest.MonkeyPatch, config: TrackerConfig) -> CellTracker:
    monkeypatch.setattr(
        tracker_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video(Spacing(1.0, 1.0, 1.0)))
    )
    return CellTracker(
        detector=cast(BlendDetectorScorer, _StubBlend()),
        edge_scorer=cast(tracker_module.BlendedEdgeTransformerScorer, _StubEdgeScorer()),
        device="cpu",
        config=config,
    )


def test_run(monkeypatch: pytest.MonkeyPatch):
    """`run` folds detection -> link -> post-processing into one linked track graph for a two-cell chain."""
    tracker = _tracker(monkeypatch, TrackerConfig(min_track_length=1, smooth_strength=0.0))
    graph = tracker.run("m.zarr", Path("m.zarr"))
    assert isinstance(graph, TrackGraph)
    assert graph.edges.tolist() == [[0, 1]]


def test_with_config(monkeypatch: pytest.MonkeyPatch):
    """`with_config` swaps the operating point while keeping the same mounted models."""
    tracker = _tracker(monkeypatch, TrackerConfig())
    swapped = tracker.with_config(TrackerConfig(disappearance_cost=5.0))
    assert swapped.config.disappearance_cost == 5.0
    assert swapped.detector is tracker.detector


def test_from_packs(monkeypatch: pytest.MonkeyPatch):
    """`from_packs` mounts both detector packs (cache-backed) and the blended edge scorer, carrying the config."""
    monkeypatch.setattr(
        tracker_module.TemporalUNetDetector,
        "from_pack",
        classmethod(lambda cls, pack, map_location: (_StubDetector(), DetectorRecipe())),
    )
    monkeypatch.setattr(
        tracker_module.BlendedEdgeTransformerScorer,
        "from_packs",
        staticmethod(lambda packs, weights, device: _StubEdgeScorer()),
    )
    config = TrackerConfig(disappearance_cost=3.0)
    tracker = CellTracker.from_packs(Path("p1"), Path("p2"), Path("cache"), "cpu", config)
    assert tracker.config.disappearance_cost == 3.0
    assert isinstance(tracker.detector, BlendDetectorScorer)


def test_spacing(monkeypatch: pytest.MonkeyPatch):
    """`spacing` reads the video's voxel spacing from its OME metadata."""
    tracker = _tracker(monkeypatch, TrackerConfig())
    assert tracker.spacing(Path("m.zarr")) == Spacing(1.0, 1.0, 1.0)


def test_transform():
    """`LinkerStage.transform` adapts a linker's `link` to the graph-stage contract."""

    @dataclass(frozen=True)
    class _Linker:
        def link(self, graph: TrackGraph) -> TrackGraph:
            return TrackGraph(graph.node_ids, graph.coordinates, np.array([[0, 1]], dtype=np.int64))

    stage = LinkerStage(cast(Linker, _Linker()))
    linked = stage.transform(_CHAIN)
    assert linked.edges.tolist() == [[0, 1]]
