from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celltrack import champion as champion_module
from celltrack.champion import ChampionConfig, ChampionPipeline
from celltrack.pipeline import BlendDetectorScorer
from celltrack.tunet import DetectorRecipe, TemporalUNetDetector
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


def _pipeline(monkeypatch: pytest.MonkeyPatch, config: ChampionConfig) -> ChampionPipeline:
    monkeypatch.setattr(
        champion_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video(Spacing(1.0, 1.0, 1.0)))
    )
    return ChampionPipeline(
        detector=cast(BlendDetectorScorer, _StubBlend()),
        edge_scorer=cast(champion_module.BlendedEdgeTransformerScorer, _StubEdgeScorer()),
        device="cpu",
        config=config,
    )


def test_run(monkeypatch: pytest.MonkeyPatch):
    """`run` threads detection → link → post-processing into one linked track graph for a two-cell chain."""
    pipeline = _pipeline(monkeypatch, ChampionConfig(min_track_length=1, smooth_strength=0.0))
    graph = pipeline.run("m.zarr", Path("m.zarr"))
    assert isinstance(graph, TrackGraph)
    assert graph.edges.tolist() == [[0, 1]]


def test_with_config(monkeypatch: pytest.MonkeyPatch):
    """`with_config` swaps the operating point while keeping the same mounted models."""
    pipeline = _pipeline(monkeypatch, ChampionConfig())
    swapped = pipeline.with_config(ChampionConfig(disappearance_cost=5.0))
    assert swapped.config.disappearance_cost == 5.0
    assert swapped.detector is pipeline.detector


def test_from_packs(monkeypatch: pytest.MonkeyPatch):
    """`from_packs` mounts both detector packs (cache-backed) and the blended edge scorer, carrying the config."""
    monkeypatch.setattr(
        champion_module.TemporalUNetDetector,
        "from_pack",
        classmethod(lambda cls, pack, map_location: (_StubDetector(), DetectorRecipe())),
    )
    monkeypatch.setattr(
        champion_module.BlendedEdgeTransformerScorer,
        "from_packs",
        staticmethod(lambda packs, weights, device: _StubEdgeScorer()),
    )
    config = ChampionConfig(disappearance_cost=3.0)
    pipeline = ChampionPipeline.from_packs(Path("p1"), Path("p2"), Path("cache"), "cpu", config)
    assert pipeline.config.disappearance_cost == 3.0
    assert isinstance(pipeline.detector, BlendDetectorScorer)


def test_spacing(monkeypatch: pytest.MonkeyPatch):
    """`spacing` reads the video's voxel spacing from its OME metadata."""
    pipeline = _pipeline(monkeypatch, ChampionConfig())
    assert pipeline.spacing(Path("m.zarr")) == Spacing(1.0, 1.0, 1.0)
