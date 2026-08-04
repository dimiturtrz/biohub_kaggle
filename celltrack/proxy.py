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
# The public frontier's fixed-8 local CV (yusuketogashi): the four test movies plus four fully-annotated
# train videos. The extra four carry ~4x denser 44b6 lineages, so a threshold sweep is not recall-saturated
# on them the way it is on the sparse-annotated test four — a broader, threshold-sensitive local read.
CV_MOVIES: tuple[str, ...] = (
    *TEST_MOVIES,
    "44b6_341df25f",
    "44b6_e57ff5c6",
    "6bba_969618f6",
    "6bba_fc83837d",
)


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
    def load(cls, root: DataRoot, stems: tuple[str, ...] = TEST_MOVIES) -> "TestMovieProxy":
        """Resolve the chosen movies, their GEFF annotations, and the shared voxel spacing from the data root.

        Defaults to the four test movies (the tightest hidden-LB proxy); pass `CV_MOVIES` for the broader,
        threshold-sensitive fixed-8 local CV.
        """
        by_stem = {p.stem: p for p in root.videos("train")}
        paths = tuple(by_stem[stem] for stem in stems)
        truths = tuple(AnnotatedTracks.from_geff(root.track_store(p)) for p in paths)
        spacing = CellVideo.from_ome_zarr(paths[0]).spacing
        return cls(paths=paths, truths=truths, spacing=spacing)

    def metrics(self, pipeline: LinkingPipeline) -> dict[str, VideoMetrics]:
        """Each movie's metrics, keyed by stem — the per-video breakdown a split score aggregates away.

        Exposes raw edge Jaccard, node counts and the node-count ratio per movie, so an experiment can tell a
        real recall/linking gain (raw Jaccard up) from a node-count-bonus artefact (only the ratio moved).
        """
        matcher = DistanceMatcher(spacing=self.spacing)
        return {
            path.stem: VideoMetrics.of(pipeline.run(path.name, path), truth, matcher)
            for path, truth in zip(self.paths, self.truths, strict=True)
        }

    def score(self, pipeline: LinkingPipeline) -> SplitScore:
        """Run the pipeline on each movie and aggregate the per-movie metrics into one split score."""
        return SplitScore.of(list(self.metrics(pipeline).values()))

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
