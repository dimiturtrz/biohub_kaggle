"""The shipped cell tracker as one mounted, runnable object: detect, score edges, then fold an ordered stage list.

Turning a video into lineages is a detector plus an ordered sequence of graph transforms. `CellTracker` mounts
the two detector seeds and the blended edge scorer once, then `run` forwards a video to nodes, scores edge
affinities, and folds a fixed list of `GraphStage`s over the graph — link, bridge one-frame gaps, drop short
tracks, smooth. Every post-detection step is the same `graph -> graph` contract (`GraphStage`), so the
sequence is data the tracker folds, not a nested call; the linker joins it through `LinkerStage`, which phrases
its `link` as that same step. It satisfies the proxy's `LinkingPipeline`, so the proxy scores the real tracker,
and a knob sweep varies one `TrackerConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from celltrack.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.gap_closer import DensityGapBridge
from celltrack.linefit_smoother import LinefitSmoother
from celltrack.linkers import LinkerConfig
from celltrack.linking import Linker
from celltrack.motion_linking import EdgeAffinity
from celltrack.pipeline import BlendDetectorScorer
from celltrack.response_cache import ResponseCache
from celltrack.short_track_filter import ShortTrackFilter
from celltrack.topology_repair import TopologyRepair
from celltrack.tunet import TemporalUNetDetector
from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing

# The tracker's two published seeds cache their forward under these keys, so a sweep replays the read-out.
_SEED1_CACHE = "pilkwang_seed1_logits"
_SEED2_CACHE = "pilkwang_seed2_logits"


class GraphStage(Protocol):
    """One post-detection pass: a track graph in, a track graph out. Every post-proc step satisfies this."""

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph after this stage's transformation (edges added/removed, coordinates cleaned)."""
        ...


@dataclass(frozen=True)
class LinkerStage:
    """Adapts a `Linker` (which phrases the step as `link`) to the `GraphStage` contract, so linking is just
    the first stage in the folded list rather than a special case the tracker must know about."""

    linker: Linker

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Wire the graph's nodes into tracks — the linker's `link`, under the stage verb."""
        return self.linker.link(graph)


@dataclass(frozen=True)
class TrackerConfig:
    """Every operating-point knob of the tracker recipe, so a sweep varies one object, not a call site."""

    threshold: float = 0.99
    linker: LinkerConfig = field(default_factory=LinkerConfig)
    edge_blend: tuple[float, float] = (0.8, 0.2)
    min_track_length: int = 6
    bridge_reach_um: float = 10.0
    bridge_max_added_fraction: float = 0.05
    # Off by default: on the shipped path `AssignmentLinker` already guarantees consecutive-frame + single-parent
    # edges and a tighter gate than `topology_edge_max_um`, and `ShortTrackFilter` drops the unbacked nodes, so
    # the repair is a measured no-op (proxy 0.9344 either way). Enable it behind a linker that lacks those
    # invariants (nearest-neighbour, motion) — it is the frontier's output filter, available not orphaned.
    topology_repair: bool = False
    topology_edge_max_um: float = 14.0
    smooth_strength: float = 0.8


@dataclass(frozen=True)
class CellTracker:
    """The mounted dual-seed tracker: forward two seeds, score edges, fold the post-proc stages — `run` a video."""

    detector: BlendDetectorScorer
    edge_scorer: BlendedEdgeTransformerScorer
    device: str
    config: TrackerConfig = field(default_factory=TrackerConfig)

    @classmethod
    def from_packs(
        cls, pack1: Path, pack2: Path, responses: Path, device: str, config: TrackerConfig | None = None
    ) -> "CellTracker":
        """Mount both detector packs (cache-backed) and the blended edge scorer from their support packs."""
        config = config or TrackerConfig()
        seed1, recipe = TemporalUNetDetector.from_pack(pack1, map_location=device)
        seed2, _ = TemporalUNetDetector.from_pack(pack2, map_location=device)
        detector = BlendDetectorScorer(
            detectors=(
                (seed1.to(device).eval(), ResponseCache(responses, _SEED1_CACHE)),
                (seed2.to(device).eval(), ResponseCache(responses, _SEED2_CACHE)),
            ),
            recipe=recipe,
            device=device,
        )
        edge_scorer = BlendedEdgeTransformerScorer.from_packs((pack1, pack2), config.edge_blend, device)
        return cls(detector=detector, edge_scorer=edge_scorer, device=device, config=config)

    def with_config(self, config: TrackerConfig) -> "CellTracker":
        """The same mounted models under a different operating point — reuses the cache across a sweep."""
        return CellTracker(detector=self.detector, edge_scorer=self.edge_scorer, device=self.device, config=config)

    def run(self, video_key: str, path: Path) -> TrackGraph:
        """Detect, score edges, and fold the post-proc stages over one video into a linked track graph."""
        nodes = self.detector.nodes(video_key, path, self.config.threshold)
        affinity = self.edge_scorer.affinities(path, nodes, self.device)
        graph = nodes
        for stage in self._stages(self.spacing(path), affinity):
            graph = stage.transform(graph)
        return graph

    def _stages(self, spacing: Spacing, affinity: EdgeAffinity | None) -> tuple[GraphStage, ...]:
        """The ordered post-detection passes at this video's spacing — link, drop short tracks, bridge gaps, smooth.

        The short-track filter runs *before* the gap bridge: prune spurious fragments first, then reconnect the
        one-frame dropouts the pruning leaves in real tracks. Reversing them (bridge then prune) lets the bridge
        splice fragments that the filter would have removed, and scores lower.
        """
        link = LinkerStage(self.config.linker.build(spacing, affinity))
        short = ShortTrackFilter(min_length=self.config.min_track_length)
        bridge = DensityGapBridge(
            spacing=spacing,
            reach_um=self.config.bridge_reach_um,
            max_added_fraction=self.config.bridge_max_added_fraction,
        )
        smooth = LinefitSmoother(strength=self.config.smooth_strength)
        # Topology repair (enforce-next-frame, single-parent, prune-isolated) runs after the graph's edges are
        # final — after bridging — so it prunes exactly the nodes the finished edge set leaves unbacked. It is
        # the frontier's output filter; optional because prune-isolated changes the node count the metric reads.
        repair = (TopologyRepair(spacing, self.config.topology_edge_max_um),) if self.config.topology_repair else ()
        return (link, short, bridge, *repair, smooth)

    def spacing(self, path: Path) -> Spacing:
        """The video's physical voxel spacing — exposed so a caller can build a matching metric matcher."""
        return CellVideo.from_ome_zarr(path).spacing
