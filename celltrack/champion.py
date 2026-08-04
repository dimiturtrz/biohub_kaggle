"""The shipped champion pipeline as one mounted, runnable object — the dual-seed AssignmentLinker recipe.

The submission kernel and the four-movie proxy were assembling the same pipeline by hand from its parts
(blend two seeds' logits into nodes, score edge affinities, link with a global 1-to-1 assignment, bridge
density gaps, drop short tracks, smooth). That duplication meant a change to the champion had to be mirrored
in two places, and the proxy could only evaluate a *different* linker than the one we ship. `ChampionPipeline`
is the single home: mount the two detector packs and the blended edge scorer once, then `run` a video into a
linked track graph. It satisfies the proxy's `LinkingPipeline` protocol, so `TestMovieProxy.score` evaluates
the real shipped recipe — and a knob sweep (e.g. the linker's disappearance cost) varies one `ChampionConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from celltrack.assignment_linking import AssignmentLinker
from celltrack.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.gap_closer import DensityGapBridge
from celltrack.linefit_smoother import LinefitSmoother
from celltrack.pipeline import BlendDetectorScorer
from celltrack.response_cache import ResponseCache
from celltrack.short_track_filter import ShortTrackFilter
from celltrack.tunet import TemporalUNetDetector
from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing

# The champion's two published seeds cache their forward under these keys, so a sweep replays the read-out.
_SEED1_CACHE = "pilkwang_seed1_logits"
_SEED2_CACHE = "pilkwang_seed2_logits"


@dataclass(frozen=True)
class ChampionConfig:
    """Every operating-point knob of the champion recipe, so a sweep varies one object, not a call site."""

    threshold: float = 0.99
    gate_um: float = 10.0
    edge_bonus: float = 20.0
    disappearance_cost: float = 0.0
    edge_blend: tuple[float, float] = (0.8, 0.2)
    min_track_length: int = 6
    bridge_reach_um: float = 10.0
    bridge_max_added_fraction: float = 0.05
    smooth_strength: float = 0.8


@dataclass(frozen=True)
class ChampionPipeline:
    """The mounted dual-seed AssignmentLinker recipe: forward two seeds, link, post-process — `run` a video."""

    detector: BlendDetectorScorer
    edge_scorer: BlendedEdgeTransformerScorer
    device: str
    config: ChampionConfig = field(default_factory=ChampionConfig)

    @classmethod
    def from_packs(
        cls, pack1: Path, pack2: Path, responses: Path, device: str, config: ChampionConfig | None = None
    ) -> "ChampionPipeline":
        """Mount both detector packs (cache-backed) and the blended edge scorer from their support packs."""
        config = config or ChampionConfig()
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

    def with_config(self, config: ChampionConfig) -> "ChampionPipeline":
        """The same mounted models under a different operating point — reuses the cache across a sweep."""
        return ChampionPipeline(detector=self.detector, edge_scorer=self.edge_scorer, device=self.device, config=config)

    def run(self, video_key: str, path: Path) -> TrackGraph:
        """Detect, link, and post-process one video into a linked track graph — the shipped recipe end to end."""
        spacing = CellVideo.from_ome_zarr(path).spacing
        nodes = self.detector.nodes(video_key, path, self.config.threshold)
        affinity = self.edge_scorer.affinities(path, nodes, self.device)
        linker = AssignmentLinker(
            spacing=spacing,
            max_distance_um=self.config.gate_um,
            affinity=affinity,
            affinity_bonus=self.config.edge_bonus,
            disappearance_cost=self.config.disappearance_cost,
        )
        bridge = DensityGapBridge(
            spacing=spacing,
            reach_um=self.config.bridge_reach_um,
            max_added_fraction=self.config.bridge_max_added_fraction,
        )
        short = ShortTrackFilter(min_length=self.config.min_track_length)
        smooth = LinefitSmoother(strength=self.config.smooth_strength)
        return smooth.transform(bridge.transform(short.transform(linker.link(nodes))))

    def spacing(self, path: Path) -> Spacing:
        """The video's physical voxel spacing — exposed so a caller can build a matching metric matcher."""
        return CellVideo.from_ome_zarr(path).spacing
