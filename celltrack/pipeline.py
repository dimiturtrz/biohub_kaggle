"""A tracking pipeline as one validated config: a detector operating point plus an ordered stage list.

The whole pipeline is two moves. A `Scorer` turns a video into nodes — the expensive, GPU, forward-once
step, cached so every later experiment is instant. Then an ordered list of `GraphStage`s turns those nodes
into a track graph — cheap, CPU, replayable. `PipelineConfig` names both declaratively (one `RunConfig`-style
source of truth, serialisable for provenance), and `build`s the concrete `Pipeline` once a scorer is wired.

The split is also what makes the config-space search fast and what makes per-slot attribution possible.
`finish` runs only the stages on nodes handed in, so an evaluation can teacher-force ground-truth nodes into
it and read a slot's ceiling in isolation, and the full eval can replay the stages over a cached detector
response without touching the GPU.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from celltrack.response_cache import ResponseCache
from celltrack.stages import GraphStage, StageSpec
from celltrack.tunet import DetectorRecipe, TemporalUNetDetector
from core.data.tracks import TrackGraph
from core.geometry import Spacing


class Scorer(Protocol):
    """A learned response over a video, forwarded once and read out into nodes. The detector is the first;
    a centre-prior veto or an edge scorer join the family by caching their own response the same way."""

    def nodes(self, video_key: str, path: Path, threshold: float) -> TrackGraph:
        """The detected cell centres for a video, as an edge-free graph — replayed from cache after a first run."""
        ...


@dataclass(frozen=True)
class DetectorScorer:
    """A trained detector plus its response cache: forwards each video once, reads nodes out on replay.

    The forward (probability volumes) is cached by video key; a threshold change re-runs only the cheap
    peak read-out, not the network. A retrain gets a fresh cache keyed by the new weights' stem.
    """

    detector: TemporalUNetDetector
    recipe: DetectorRecipe
    cache: ResponseCache
    device: str

    def nodes(self, video_key: str, path: Path, threshold: float) -> TrackGraph:
        """Detect a video's centres, forwarding-and-caching on a first miss and replaying the read-out after."""
        volumes = self.cache.responses(
            video_key, lambda: self.detector.probability_volumes(path, self.recipe, self.device)
        )
        scale = self.detector.voxel_scale(path)
        return self.detector.graph_from_volumes(volumes, scale, threshold, self.recipe, self.device)


class DetectorSpec(BaseModel):
    """The detector's operating point — the read-out threshold that turns its cached response into nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float = Field(0.99, gt=0, lt=1)


class PipelineConfig(BaseModel):
    """A whole pipeline, declaratively: the detector operating point and the ordered post-detection stages.

    The stage order is the pipeline — gap-close, then divisions, then the short-track filter, then smoothing
    is a different config from any reordering. Immutable and validated, so a loaded config.json or a `--set`
    override is bounds-checked before a run rather than failing deep inside a stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: DetectorSpec = DetectorSpec()
    stages: tuple[StageSpec, ...] = ()

    def build_stages(self, spacing: Spacing) -> tuple[GraphStage, ...]:
        """The concrete ordered stages, each spec resolved to its impl at the video's spacing."""
        return tuple(spec.build(spacing) for spec in self.stages)

    def build(self, scorer: Scorer, spacing: Spacing) -> "Pipeline":
        """Wire this config to a concrete scorer, yielding a runnable pipeline."""
        return Pipeline(scorer=scorer, threshold=self.detector.threshold, stages=self.build_stages(spacing))


@dataclass(frozen=True)
class Pipeline:
    """A runnable pipeline: a scorer for the nodes and the ordered stages that finish them into tracks."""

    scorer: Scorer
    threshold: float
    stages: tuple[GraphStage, ...]

    def detect(self, video_key: str, path: Path) -> TrackGraph:
        """The scorer's nodes for a video (cached forward, cheap read-out)."""
        return self.scorer.nodes(video_key, path, self.threshold)

    def finish(self, nodes: TrackGraph) -> TrackGraph:
        """Run the ordered stages on nodes handed in — the CPU replay half, teacher-forceable from GT nodes."""
        graph = nodes
        for stage in self.stages:
            graph = stage.transform(graph)
        return graph

    def run(self, video_key: str, path: Path) -> TrackGraph:
        """Detect then finish — the full pipeline on one video."""
        return self.finish(self.detect(video_key, path))
