"""The four-competition-test-movie proxy — the tightest local stand-in for the hidden leaderboard.

The four test movies also live in the train set with sparse GEFF annotations, and the hidden evaluation scores
additional annotations on these same movies (validated: proxy 0.888 -> public 0.882 transfer). So scoring a
pipeline on these movies' known annotations ranks configurations far better than an unrelated fold. This module
owns the canonical movie identifiers (previously copy-pasted across a dozen eval scripts) and the score loop,
so an experiment is a thin driver that builds a pipeline and calls :meth:`TestMovieProxy.score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from celltrack.linking import Linker
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

TEST_MOVIES: tuple[str, ...] = ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1")


class LinkingPipeline(Protocol):
    """The minimal surface the proxy needs — anything that turns a video into a linked track graph."""

    def run(self, video_key: str, path: Path) -> TrackGraph:
        """Detect, link, and post-process one video into a track graph."""
        ...


@dataclass(frozen=True)
class TestMovieProxy:
    """The four test movies loaded once — score any pipeline against their annotations."""

    paths: tuple[Path, ...]
    truths: tuple[AnnotatedTracks, ...]
    spacing: Spacing

    @classmethod
    def load(cls, root: DataRoot) -> "TestMovieProxy":
        """Resolve the four movies, their GEFF annotations, and the shared voxel spacing from the data root."""
        by_stem = {p.stem: p for p in root.videos("train")}
        paths = tuple(by_stem[stem] for stem in TEST_MOVIES)
        truths = tuple(AnnotatedTracks.from_geff(root.track_store(p)) for p in paths)
        spacing = CellVideo.from_ome_zarr(paths[0]).spacing
        return cls(paths=paths, truths=truths, spacing=spacing)

    def score(self, pipeline: LinkingPipeline) -> SplitScore:
        """Run the pipeline on each movie and aggregate the per-movie metrics into one split score."""
        matcher = DistanceMatcher(spacing=self.spacing)
        metrics = [
            VideoMetrics.of(pipeline.run(path.name, path), truth, matcher)
            for path, truth in zip(self.paths, self.truths, strict=True)
        ]
        return SplitScore.of(metrics)

    def linker_ceiling(self, linker: Linker) -> dict[str, float]:
        """Per-movie edge-jaccard ceiling: link each movie's GT nodes directly (perfect detection).

        Isolates the linker from the detector — a high ceiling next to a low real-detection score means the
        loss is real-detection crowding the linker, not missed or false detection.
        """
        matcher = DistanceMatcher(spacing=self.spacing)
        ceilings: dict[str, float] = {}
        for path, truth in zip(self.paths, self.truths, strict=True):
            linked = linker.link(truth.graph.without_edges())
            counts = EdgeCounts.of(linked, truth.graph, matcher.match(linked, truth.graph))
            ceilings[path.stem] = counts.jaccard()
        return ceilings
