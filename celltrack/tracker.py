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

import logging
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from celltrack.affinity import EdgeAffinity
from celltrack.detectors.pipeline import BlendDetectorScorer
from celltrack.detectors.response_cache import EphemeralResponseStore, ResponseCache, ResponseStore
from celltrack.detectors.tunet import TemporalUNetDetector
from celltrack.edges.association_ranker import AssociationRanker
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer, EdgeBlendOptions
from celltrack.linkers.linkers import LinkerConfig
from celltrack.linkers.linking import Linker
from celltrack.postproc.affinity_division_recovery import AffinityDivisionConfig
from celltrack.postproc.gap_closer import BridgeConfig, ReuseConfig
from celltrack.postproc.linefit_smoother import LinefitSmoother
from celltrack.postproc.short_track_filter import ShortTrackFilter, ShortTrackRescueConfig
from celltrack.postproc.topology_repair import TopologyConfig
from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing

logger = logging.getLogger(__name__)

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

    # 0.97 is the MEASURED best, not a round number: the leaderboard ladder is 0.99 -> 0.887,
    # 0.98 -> 0.891, 0.97 -> 0.892, a spread five times our noise floor. The default used to be 0.99,
    # so every caller that omitted a config silently got the worst of the three — the shipped kernel only
    # escaped it by hardcoding 0.97. Threshold is a RECALL lever the sparse proxy cannot see (it reads
    # flat-to-inverted there), so this value comes from the leaderboard and must not be re-tuned on the proxy.
    threshold: float = 0.97
    linker: LinkerConfig = field(default_factory=LinkerConfig)
    edge_blend: tuple[float, float] = (0.8, 0.2)
    # The seed-1 fraction of the detector's logit blend; None is the equal mean the frontier ships. The edge
    # blend favours seed 1 (0.8), so the detector blend may too — a value w tilts the two seeds to (w, 1-w).
    detector_blend: float | None = None
    # Off by default: recovering the edge head's second-daughter fork is a detection-limited lever (the second
    # daughter is often undetected, so the fork cannot be placed). Set it to A/B the division-jaccard term.
    division: AffinityDivisionConfig | None = None
    min_track_length: int = 6
    # Off by default: the confidence rescue that keeps a short high-probability, tight-step component the length
    # rule would drop. It reads the edge head, so it needs an affinity; enable it (with an aggressive
    # min_track_length) to recover the true short tracks the blunt cut removes — a recall lever the sparse proxy
    # understates but the denser hidden annotation rewards. The frontier pairs min_track_length 7 with this.
    rescue: ShortTrackRescueConfig | None = None
    # Off by default: exempt components touching the first or last frame from the length cut. Their
    # brevity is the observation window's doing, not the detector's — a cell entering at frame 96 of 100
    # cannot reach min_track_length however real it is, yet the hidden annotation still scores its edges.
    keep_boundary_tracks: bool = False
    # The one-frame motion-synthetic gap bridge — always on (it helps, +0.0017 at its plateau cap).
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    # Off by default (present = on): reuse an existing isolated t+1 detection to bridge a one-frame gap (geometry
    # only, no invented node), complementary to the motion-synthetic bridge. Inert at thr0.99 (few isolated nodes
    # survive), but the lower-threshold recall regime leaves more isolated detections to reuse — enable it there
    # to A/B whether the precision-safe reuse recovers dropped edges the synthetic bridge misses.
    reuse: ReuseConfig | None = None
    # Off by default (present = on): on the shipped AssignmentLinker path this is a measured no-op (the linker
    # already guarantees consecutive-frame + single-parent edges and a tighter gate), so it is off; enable it
    # behind a linker (nearest-neighbour, motion) that lacks those invariants — the frontier's output filter.
    topology: TopologyConfig | None = None
    # Proxy favours 0.3 but a clean LB A/B refuted it (smooth0.8 = 0.892 vs smooth0.3 = 0.889): the dense
    # raw-Jaccard gain is recall-side and does not transfer, like min_track_length/node-count. Ship the LB
    # value 0.8; 0.3 stays a swept-available knob for the proxy regime.
    smooth_strength: float = 0.8
    # Off by default: score every gap a second time with the temporal pair swapped and fuse the two
    # normalisations by their harmonic mean (`BlendedEdgeTransformerScorer.fuse`). Costs a second affinity
    # forward and no retraining; it targets the affinity-inverted dense mislink, where a wrong near neighbour
    # wins forwards but loses backwards.
    bidirectional_edges: bool = False
    # The edge blend's two candidate transforms (`EdgeBlendOptions`), both off by default: seed-moment alignment
    # before the weighted sum, and four-view flip TTA on the edge head fused by a JS-reliability log pool. They
    # live here rather than at the mount because they are operating-point knobs a sweep re-points — reach them
    # with `--set edge_options.align_seed_moments=true` / `--set edge_options.view_tta=true`.
    edge_options: EdgeBlendOptions = field(default_factory=EdgeBlendOptions)

    def __post_init__(self) -> None:
        """Reject an operating point that cannot mean what it says — the one config level that had no checks.

        Every nested config here is a validated pydantic model; this dataclass was the exception, so a typo at
        the top of the hierarchy constructed fine and misbehaved quietly. The blend is the sharp case: the
        seeds' LOGITS are combined as a raw weighted sum before the softmax, so weights that do not sum to one
        rescale the logits and shift the softmax TEMPERATURE — a different pipeline, not a different mix.
        """
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be a probability in [0, 1], got {self.threshold}")
        if self.min_track_length < 1:
            raise ValueError(f"min_track_length must keep at least one node, got {self.min_track_length}")
        if not math.isclose(sum(self.edge_blend), 1.0, abs_tol=1e-6):
            raise ValueError(f"edge_blend must sum to 1 (it scales logits before the softmax), got {self.edge_blend}")
        if self.detector_blend is not None and not 0.0 <= self.detector_blend <= 1.0:
            raise ValueError(f"detector_blend is seed1's share in [0, 1], got {self.detector_blend}")
        if not 0.0 <= self.smooth_strength <= 1.0:
            raise ValueError(f"smooth_strength must be in [0, 1], got {self.smooth_strength}")


@dataclass(frozen=True)
class TrackedVideo:
    """One video's finished track graph beside the two things it was built FROM — the detections and their affinity.

    `run` returns the graph alone, which discards exactly what a post-hoc association diagnosis needs: the
    per-gap affinity matrices, and the DETECTION node set those matrices are aligned to (post-processing prunes
    short tracks and inserts synthetic bridge nodes, so the finished graph is a different row space). Handing
    all three back makes a mislink analysis free for a caller that already ran the tracker — no second forward.
    """

    graph: TrackGraph
    detections: TrackGraph
    affinity: EdgeAffinity


@dataclass(frozen=True)
class CellTracker:
    """The mounted dual-seed tracker: forward two seeds, score edges, fold the post-proc stages — `run` a video.

    `ranker` is the optional local-association re-ranker (`with_ranker`), mounted like the detector packs are:
    the caller hands in the artifact directory it found, so the same object serves the local data root and the
    Kaggle kernel's own mount point. It is what turns `run_scored` into the two-pass path documented there.
    """

    detector: BlendDetectorScorer
    edge_scorer: BlendedEdgeTransformerScorer
    device: str
    config: TrackerConfig = field(default_factory=TrackerConfig)
    ranker: AssociationRanker | None = None

    @classmethod
    def from_packs(
        cls, pack1: Path, pack2: Path, responses: Path, device: str, config: TrackerConfig | None = None
    ) -> "CellTracker":
        """Mount both detector packs (cache-backed) and the blended edge scorer from their support packs."""
        stores = (ResponseCache(responses, _SEED1_CACHE), ResponseCache(responses, _SEED2_CACHE))
        return cls._mounted(pack1, pack2, stores, device, config)

    @classmethod
    def ephemeral(cls, pack1: Path, pack2: Path, device: str, config: TrackerConfig | None = None) -> "CellTracker":
        """The same mount as `from_packs` but persisting nothing — the tracker a single-pass submission runs.

        A submission forwards each video once and reads it out once, so there is no later threshold or linker
        sweep for a cache to amortise: the disk write buys nothing and the gigabyte-per-seed fp32 logit volumes
        fill the kernel's bounded `/kaggle/working` (the disk-full failure the caching store caused there). Each
        seed therefore gets an `EphemeralResponseStore`, and no responses directory is needed at all.

        Its arguments are exactly two paths, a device string and a config — all picklable — so a spawned
        per-GPU worker can rebuild the identical tracker from them rather than inheriting mounted CUDA models.
        """
        return cls._mounted(pack1, pack2, (EphemeralResponseStore(), EphemeralResponseStore()), device, config)

    @classmethod
    def _mounted(
        cls,
        pack1: Path,
        pack2: Path,
        stores: tuple[ResponseStore, ResponseStore],
        device: str,
        config: TrackerConfig | None,
    ) -> "CellTracker":
        config = config or TrackerConfig()
        seed1, recipe = TemporalUNetDetector.from_pack(pack1, map_location=device)
        seed2, _ = TemporalUNetDetector.from_pack(pack2, map_location=device)
        detector = BlendDetectorScorer(
            detectors=((seed1.to(device).eval(), stores[0]), (seed2.to(device).eval(), stores[1])),
            recipe=recipe,
            device=device,
        )
        edge_scorer = BlendedEdgeTransformerScorer.from_packs(
            (pack1, pack2),
            config.edge_blend,
            device,
            bidirectional=config.bidirectional_edges,
            options=config.edge_options,
        )
        return cls(detector=detector, edge_scorer=edge_scorer, device=device, config=config)

    def with_config(self, config: TrackerConfig) -> "CellTracker":
        """The same mounted models under a different operating point — reuses the cache across a sweep."""
        edge_scorer = self.edge_scorer.with_options(config.edge_options).with_bidirectional(
            bidirectional=config.bidirectional_edges
        )
        return CellTracker(
            detector=self.detector,
            edge_scorer=edge_scorer,
            device=self.device,
            config=config,
            ranker=self.ranker,
        )

    def with_ranker(self, artifact: Path) -> "CellTracker":
        """The same tracker with the local-association re-ranker mounted from the artifact directory given.

        A path argument rather than a constant for the same reason the detector packs take one: the artifact
        sits under the local data root here and wherever the Kaggle kernel's dataset mount lands it there, and
        a tracker that knew one of those two could not run in the other. Mounting is separate from `from_packs`
        so the re-ranker stays optional at every call site that does not price it.
        """
        return replace(self, ranker=AssociationRanker.from_artifact(artifact))

    def run(self, video_key: str, path: Path) -> TrackGraph:
        """Detect, score edges, and fold the post-proc stages over one video into a linked track graph."""
        return self.run_scored(video_key, path).graph

    def run_scored(self, video_key: str, path: Path) -> TrackedVideo:
        """The same pass as `run`, also returning the detections and the affinity the graph was built from.

        Each stage is timed at INFO so a slow submission is diagnosable per stage (detect / affinity / link+post)
        rather than as one opaque wall-clock — the detection forward is the GPU cost, the stages are the CPU cost.
        """
        seed1 = self.config.detector_blend
        blend = None if seed1 is None else (seed1, 1 - seed1)
        detect_start = time.perf_counter()
        nodes = self.detector.nodes(video_key, path, self.config.threshold, blend)
        affinity_start = time.perf_counter()
        affinity = self.edge_scorer.affinities(path, nodes, self.device)
        mutual = self.fused_affinities(path, nodes, affinity)
        stages_start = time.perf_counter()
        graph = nodes
        video = CellVideo.from_ome_zarr(path)
        ranker = self._ranked(video.spacing, nodes, affinity, mutual, video.volume_shape)
        for stage in self._stages(video.spacing, affinity, mutual, video.volume_shape, ranker):
            graph = stage.transform(graph)
        logger.info(
            "%s: detect %.1fs | affinity %.1fs | link+post %.1fs (%d nodes)",
            video_key,
            affinity_start - detect_start,
            stages_start - affinity_start,
            time.perf_counter() - stages_start,
            len(nodes.node_ids),
        )
        return TrackedVideo(graph=graph, detections=nodes, affinity=affinity)

    def fused_affinities(self, path: Path, nodes: TrackGraph, affinity: EdgeAffinity | None) -> EdgeAffinity | None:
        """The bidirectionally fused probabilities the linker's agreement knobs read — `None` when unused.

        Public because a diagnosis that reconstructs the tracker's own first linking pass
        (`celltrack.eval.ranker_distribution`) must resolve the fused affinity by exactly this rule; a second
        copy of it would let the reconstruction silently link under a different cost than the tracker does.

        Scored only when a knob consumes it (`LinkerConfig.needs_mutual`: the admission floor, the cost's
        mutual-agreement term, or both), because it costs a second reversed pass over every gap. When the
        tracker already runs `bidirectional_edges` the mounted affinity IS the fused one, so those knobs read it
        directly and the extra pass is skipped — that pairing then reads one quantity for both roles.
        """
        if not self.config.linker.needs_mutual:
            return None
        if self.config.bidirectional_edges:
            return affinity
        return self.edge_scorer.with_bidirectional(bidirectional=True).affinities(path, nodes, self.device)

    def _ranked(
        self,
        spacing: Spacing,
        nodes: TrackGraph,
        affinity: EdgeAffinity | None,
        mutual: EdgeAffinity | None,
        volume_shape: tuple[int, int, int] | None,
    ) -> EdgeAffinity | None:
        """The re-ranker's per-gap probabilities, scored against a first linking pass — `None` when unused.

        This is the tracker's only TWO-PASS path, and it is two-pass by necessity rather than by choice: seven
        of the ranker's 22 features (`edge_prob`, `has_learned_edge`, the two degrees, the two `has_*` flags,
        `target_best_next_prob`) are facts about an EMITTED EDGE LIST, so nothing can be scored until something
        has linked. Pass 1 links under this operating point with the ranker's weight folded back onto the
        affinity (`LinkerConfig.without_ranker`, which spends the whole learned-evidence budget on the evidence
        that exists yet), pass 2 re-links the SAME detections and the SAME affinity with the ranker's
        probability priced in — a second link is ~11% of the video's runtime, where a second detection or
        affinity forward would be ~88%. Folding the weight rather than dropping it is what makes the context
        INVARIANT to the affinity/ranker split, so sweeping the split measures the ranker rather than measuring
        how degraded a graph its own features were read off (bead ic44).

        The context graph is the RAW output of pass 1's linker, taken before any post-processing. Two reasons,
        one of them fatal: the gap bridge INSERTS synthetic nodes and the short-track filter removes real ones,
        so a post-processed graph no longer shares the detections' row space the features and the affinity are
        both indexed in; and the public pipeline this ports featurises its own relink's output, which is
        likewise pre-post-processing. Post-proc runs once, over pass 2's links.
        """
        if self.ranker is None or not self.config.linker.needs_ranker:
            return None
        context = self.config.linker.without_ranker().build(spacing, affinity, mutual, volume_shape).link(nodes)
        return self.ranker.affinities(spacing, nodes, context, affinity, self.config.linker.gate_um)

    def _stages(
        self,
        spacing: Spacing,
        affinity: EdgeAffinity | None,
        mutual: EdgeAffinity | None = None,
        volume_shape: tuple[int, int, int] | None = None,
        ranker: EdgeAffinity | None = None,
    ) -> tuple[GraphStage, ...]:
        """The ordered post-detection passes at this video's spacing — link, drop short tracks, bridge gaps, smooth.

        The short-track filter runs *before* the gap bridge: prune spurious fragments first, then reconnect the
        one-frame dropouts the pruning leaves in real tracks. Reversing them (bridge then prune) lets the bridge
        splice fragments that the filter would have removed, and scores lower.

        `volume_shape` is the imaged `(z, y, x)` extent, which a detection graph does not carry: it is what lets a
        linker tell a track leaving the field of view from one breaking mid-volume. `ranker` carries the
        re-ranker's probabilities from `_ranked`, which is why the list is folded once over pass 2's cost rather
        than once per pass — post-processing sees only the final links.
        """
        config = self.config
        link = LinkerStage(config.linker.build(spacing, affinity, mutual, volume_shape, ranker))
        # Division recovery reads the edge affinity, so it needs a learned head and runs before the short-track
        # filter (whose division-preserving carve-out can only protect a fork that already exists).
        divide = (
            (config.division.build(spacing, affinity),) if config.division is not None and affinity is not None else ()
        )
        # Reuse-bridge runs before the short-track filter so the isolated t+1 nodes it links through survive as
        # part of a bridged track rather than being pruned as length-1 fragments first.
        reuse = (config.reuse.build(spacing),) if config.reuse is not None else ()
        rescue = config.rescue.build(spacing, affinity) if config.rescue is not None and affinity else None
        short = ShortTrackFilter(
            min_length=config.min_track_length, rescue=rescue, keep_boundary=config.keep_boundary_tracks
        )
        bridge = config.bridge.build(spacing)
        smooth = LinefitSmoother(strength=config.smooth_strength)
        # Topology repair (enforce-next-frame, single-parent, prune-isolated) runs after the graph's edges are
        # final — after bridging — so it prunes exactly the nodes the finished edge set leaves unbacked. It is
        # the frontier's output filter; optional because prune-isolated changes the node count the metric reads.
        repair = (config.topology.build(spacing),) if config.topology is not None else ()
        return (link, *reuse, *divide, short, bridge, *repair, smooth)

    def spacing(self, path: Path) -> Spacing:
        """The video's physical voxel spacing — exposed so a caller can build a matching metric matcher."""
        return CellVideo.from_ome_zarr(path).spacing
