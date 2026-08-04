from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celltrack import proxy
from celltrack.proxy import TEST_MOVIES, TestMovieProxy
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.geometry import Spacing
from core.paths import DataRoot


@dataclass
class _Truth:
    """A stand-in for AnnotatedTracks exposing only the `.graph` the ceiling probe reads."""

    graph: TrackGraph


class _StubPipeline:
    """Records each run call and returns a per-call sentinel graph, so the score loop can be checked."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def run(self, video_key: str, path: Path) -> TrackGraph:
        self.calls.append((video_key, path))
        return cast(TrackGraph, f"graph:{video_key}")


def test_test_movies_are_the_four_competition_movies():
    """The canonical identifiers are the four test movies, in acquisition order."""
    assert TEST_MOVIES == ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1")


def test_score(monkeypatch: pytest.MonkeyPatch):
    """`score` runs the pipeline once per movie and feeds every per-movie metric into one split score."""
    seen: list[tuple[object, object]] = []
    monkeypatch.setattr(proxy.VideoMetrics, "of", staticmethod(lambda graph, truth, matcher: (graph, truth)))
    monkeypatch.setattr(proxy.SplitScore, "of", staticmethod(lambda metrics: seen.extend(metrics) or "split"))

    pipeline = _StubPipeline()
    subject = TestMovieProxy(
        paths=(Path("a.zarr"), Path("b.zarr")),
        truths=cast(tuple[AnnotatedTracks, ...], ("truth-a", "truth-b")),
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
    )
    result = subject.score(pipeline)

    assert result == "split"
    assert pipeline.calls == [("a.zarr", Path("a.zarr")), ("b.zarr", Path("b.zarr"))]
    assert seen == [("graph:a.zarr", "truth-a"), ("graph:b.zarr", "truth-b")]


def test_metrics(monkeypatch: pytest.MonkeyPatch):
    """`metrics` runs the pipeline once per movie and keys each movie's metrics by its stem."""
    monkeypatch.setattr(proxy.VideoMetrics, "of", staticmethod(lambda graph, truth, matcher: (graph, truth)))
    pipeline = _StubPipeline()
    subject = TestMovieProxy(
        paths=(Path("a.zarr"), Path("b.zarr")),
        truths=cast(tuple[AnnotatedTracks, ...], ("truth-a", "truth-b")),
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
    )
    result = subject.metrics(pipeline)

    assert result == {"a": ("graph:a.zarr", "truth-a"), "b": ("graph:b.zarr", "truth-b")}
    assert pipeline.calls == [("a.zarr", Path("a.zarr")), ("b.zarr", Path("b.zarr"))]


def test_load(monkeypatch: pytest.MonkeyPatch):
    """`load` picks the four movies out of the train set, reads their GEFF truths, and the shared spacing."""

    class _Video:
        def __init__(self, spacing: Spacing) -> None:
            self.spacing = spacing

    class _Root:
        def videos(self, split: str) -> list[Path]:
            assert split == "train"
            return [Path(f"{stem}.zarr") for stem in (*TEST_MOVIES, "44b6_unused")]

        def track_store(self, path: Path) -> Path:
            return path.with_suffix(".geff")

    spacing = Spacing(z=1.6, y=0.4, x=0.4)
    monkeypatch.setattr(proxy.AnnotatedTracks, "from_geff", staticmethod(lambda store: f"truth:{store.stem}"))
    monkeypatch.setattr(proxy.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video(spacing)))

    subject = TestMovieProxy.load(cast(DataRoot, _Root()))

    assert tuple(p.stem for p in subject.paths) == TEST_MOVIES
    assert subject.truths == tuple(f"truth:{stem}" for stem in TEST_MOVIES)
    assert subject.spacing == spacing


def test_load_selects_a_custom_stem_set(monkeypatch: pytest.MonkeyPatch):
    """`load` resolves whatever stems it is given — the CV set, not only the default four test movies."""

    class _Video:
        spacing = Spacing(z=1.0, y=1.0, x=1.0)

    class _Root:
        def videos(self, split: str) -> list[Path]:
            return [Path(f"{stem}.zarr") for stem in (*TEST_MOVIES, "44b6_extra")]

        def track_store(self, path: Path) -> Path:
            return path.with_suffix(".geff")

    monkeypatch.setattr(proxy.AnnotatedTracks, "from_geff", staticmethod(lambda store: store.stem))
    monkeypatch.setattr(proxy.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video()))

    subject = TestMovieProxy.load(cast(DataRoot, _Root()), ("44b6_extra", "44b6_0113de3b"))

    assert tuple(p.stem for p in subject.paths) == ("44b6_extra", "44b6_0113de3b")


def test_linker_ceiling():
    """A perfect linker (returns the GT edges) scores a ceiling of 1.0 on each movie's GT nodes."""
    gt = TrackGraph(
        node_ids=np.array([0, 1], dtype=np.int64),
        coordinates=np.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.int64),
        edges=np.array([[0, 1]], dtype=np.int64),
    )

    class _PerfectLinker:
        def link(self, detections: TrackGraph) -> TrackGraph:
            return TrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=gt.edges)

    subject = TestMovieProxy(
        paths=(Path("m0.zarr"), Path("m1.zarr")),
        truths=cast(tuple[AnnotatedTracks, ...], (_Truth(gt), _Truth(gt))),
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
    )
    ceilings = subject.linker_ceiling(_PerfectLinker())

    assert ceilings == {"m0": 1.0, "m1": 1.0}


def test_load_raises_if_a_movie_is_absent():
    """A missing test movie is a hard error, not a silently short proxy."""

    class _Root:
        def videos(self, split: str) -> list[Path]:
            return [Path("44b6_0113de3b.zarr")]  # only one of the four

        def track_store(self, path: Path) -> Path:
            return path

    with pytest.raises(KeyError):
        TestMovieProxy.load(cast(DataRoot, _Root()))
