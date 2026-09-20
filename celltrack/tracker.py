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
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from celltrack.affinity import EdgeAffinity
from celltrack.detectors.pipeline import BlendDetectorScorer, DetectionFusionOptions
from celltrack.detectors.response_cache import EphemeralResponseStore, ResponseCache, ResponseStore
from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector
from celltrack.edges.association_ranker import AssociationRanker
from celltrack.edges.blended_edge_scoring import BlendedEdgeTransformerScorer
from celltrack.linkers.linking import Linker
from celltrack.linkers.motion_linking import _KERNEL_EDGE_ADMISSION_THRESHOLD
from celltrack.models.edge_transformer import EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.operating_point import TrackerConfig
from celltrack.postproc.linefit_smoother import LinefitSmoother
from celltrack.postproc.short_track_filter import ShortTrackFilter
from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing

if TYPE_CHECKING:
    # Type-only: the tracker holds an already-mounted `CenterPriorScorer` and passes its `CenterPriorVeto`
    # through opaquely to the recovery stage. Construction lives on the scorer (`CenterPriorScorer.from_pack`),
    # so the centre-prior package is a submission concern the core tracker never imports at runtime.
    from celltrack.detectors.center_prior import CenterPriorScorer, CenterPriorVeto

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
class StageEvidence:
    """The optional mounted extras the post-detection stages read: the re-ranker's per-gap probabilities and
    the centre-prior veto. Each is `None` unless its model is mounted and the config prices it, so bundling
    them keeps `_stages` a step over per-video linking inputs plus this one evidence object rather than a
    widening argument list as more mounted heads are added."""

    ranker: EdgeAffinity | None = None
    veto: CenterPriorVeto | None = None


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
    # The optional DeepCenter centre-prior (`with_center_prior`), mounted like the ranker: the caller hands in
    # the pack directory it found. Present -> `run_scored` forwards it once per video into a `CenterPriorVeto`
    # the confirmed-recovery stages (reuse-bridge synthetic insertion) resolve their confirmer from. None keeps
    # the shipped path byte-identical (no veto is built, every stage resolves its confirmer to None).
    center_scorer: CenterPriorScorer | None = None

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
    def from_joint(
        cls,
        checkpoint: Path,
        recipe: DetectorRecipe,
        device: str,
        config: TrackerConfig | None = None,
        responses: Path | None = None,
    ) -> "CellTracker":
        """The shipped tracker with BOTH heads from one jointly-trained checkpoint — no pilkwang packs.

        `ephemeral` mounts pilkwang's edge affinity behind the detector, which discards the association half a
        joint run trained. This mounts the model's OWN detector and transformer as the one-member logit blend
        `ModelEvaluator.evaluate_joint` selects on — the identical assembly, so a submission runs the whole
        trained pipeline the proxy scored. `recipe` is the detector's input pipeline (downsample / TTA) the model
        was read out under; a joint checkpoint carries no pack to bake it, so the caller states it.
        """
        config = config or TrackerConfig.shipped()
        model = JointModel.from_checkpoint(checkpoint, device)
        if tuple(recipe.downsample) != tuple(model.downsample):
            raise ValueError(
                f"recipe downsample {tuple(recipe.downsample)} != checkpoint {tuple(model.downsample)}: the "
                "detector would decode at a grid the transformer was never trained on — a silent mismatch that "
                "returns garbage at slot cost. Build the recipe from the checkpoint's own downsample."
            )
        detector = TemporalUNetDetector.of(model.detector)
        scorer = EdgeTransformerScorer.of(detector, model.transformer, recipe)
        edge_scorer = BlendedEdgeTransformerScorer(
            (scorer,), (1.0,), bidirectional=config.bidirectional_edges, options=config.edge_options
        )
        # A submission persists nothing (ephemeral). But the merge_referee GT-free gate reads cached fp32 logit
        # volumes off disk (`<video>.logit.<grid>_tta1_pc0.npy`), so a diagnostic dump run threads a persistent
        # ResponseCache under `cache_key` (default: the checkpoint stem + "_logits"). Same fingerprint the
        # pilkwang packs cache under, so the referee compares like-for-like across weights-keys.
        store: ResponseStore = (
            ResponseCache(responses, f"{checkpoint.stem}_logits") if responses is not None else EphemeralResponseStore()
        )
        blended = BlendDetectorScorer(detectors=((detector, store),), recipe=recipe, device=device)
        return cls(detector=blended, edge_scorer=edge_scorer, device=device, config=config)

    @classmethod
    def _mounted(
        cls,
        pack1: Path,
        pack2: Path,
        stores: tuple[ResponseStore, ResponseStore],
        device: str,
        config: TrackerConfig | None,
    ) -> "CellTracker":
        # The SHIPPED recipe, not TrackerConfig's neutral field defaults: those are the per-frame assignment
        # linker with no fusion (the 0.892 tier), so every caller that omitted a config silently mounted a
        # pipeline we do not submit. Measured consequence, not a tidiness argument — re-ranking saved
        # checkpoints under both pipelines INVERTS which one wins (0.9175/0.9141 becomes 0.9153/0.9206).
        config = config or TrackerConfig.shipped()
        seed1, recipe = TemporalUNetDetector.from_pack(pack1, map_location=device)
        seed2, _ = TemporalUNetDetector.from_pack(pack2, map_location=device)
        # pair_context is a mount-time knob: the pack bakes it False, so thread the operating point onto the
        # recipe here (the fingerprint keys on it, so the two arms never share cached logit volumes).
        recipe = replace(recipe, pair_context=config.pair_context)
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
        # All THREE of the scorer's operating-point knobs, not two: the weights were missing, so a sweep over
        # `edge_blend` re-pointed the config and scored the mount's original mix every time.
        edge_scorer = (
            self.edge_scorer.with_options(config.edge_options)
            .with_bidirectional(bidirectional=config.bidirectional_edges)
            .with_weights(config.edge_blend)
        )
        return CellTracker(
            detector=self.detector,
            edge_scorer=edge_scorer,
            device=self.device,
            config=config,
            ranker=self.ranker,
            center_scorer=self.center_scorer,
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
        nodes = self.detector.nodes(
            video_key,
            path,
            self.config.threshold,
            blend,
            fusion=DetectionFusionOptions(
                align_moments=self.config.detector_align_moments, kernel_faithful=self.config.kernel_faithful
            ),
        )
        affinity_start = time.perf_counter()
        affinity = self.edge_scorer.affinities(path, nodes, self.device)
        mutual = self.fused_affinities(path, nodes, affinity)
        stages_start = time.perf_counter()
        graph = nodes
        video = CellVideo.from_ome_zarr(path)
        evidence = StageEvidence(
            ranker=self._ranked(video.spacing, nodes, affinity, mutual, video.volume_shape),
            veto=self.center_scorer.veto(video_key, path) if self.center_scorer is not None else None,
        )
        elapsed: list[tuple[str, float]] = []
        for stage in self._stages(video.spacing, affinity, mutual, video.volume_shape, evidence):
            stage_start = time.perf_counter()
            graph = stage.transform(graph)
            elapsed.append((type(stage).__name__, time.perf_counter() - stage_start))
        logger.info("%s: stages %s", video_key, self._stage_breakdown(elapsed))
        logger.info(
            "%s: detect %.1fs | affinity %.1fs | link+post %.1fs (%d nodes)",
            video_key,
            affinity_start - detect_start,
            stages_start - affinity_start,
            time.perf_counter() - stages_start,
            len(nodes.node_ids),
        )
        return TrackedVideo(graph=graph, detections=nodes, affinity=affinity)

    @staticmethod
    def _stage_breakdown(elapsed: list[tuple[str, float]]) -> str:
        """The post-proc fold's cost per stage, slowest first — which stage owns `link+post`.

        The aggregate `link+post` number was 11% of a video when the fold was linking alone; it is now the
        majority of it, and one pooled number cannot say which of the accreted stages took it. Ordering by
        cost puts the optimisation target at the front of the line rather than in the reader's arithmetic.
        """
        return " | ".join(f"{name} {seconds:.1f}s" for name, seconds in sorted(elapsed, key=lambda row: -row[1]))

    def fused_affinities(self, path: Path, nodes: TrackGraph, affinity: EdgeAffinity | None) -> EdgeAffinity | None:
        """The bidirectionally fused probabilities the linker's agreement knobs read — `None` when unused.

        Public because a diagnosis that reconstructs the tracker's own first linking pass
        (`celltrack.analysis.ranker_distribution`) must resolve the fused affinity by exactly this rule; a second
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
        linker = self.config.linker
        context = linker.without_ranker().build(spacing, affinity, mutual, volume_shape).link(nodes)
        return self.ranker.affinities(spacing, nodes, context, affinity, linker.gate_um)

    def _stages(
        self,
        spacing: Spacing,
        affinity: EdgeAffinity | None,
        mutual: EdgeAffinity | None = None,
        volume_shape: tuple[int, int, int] | None = None,
        evidence: StageEvidence | None = None,
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
        evidence = evidence if evidence is not None else StageEvidence()
        # E5: in faithful mode the linker admits an edge candidate only above the kernel's 0.48 probability floor.
        linker = config.linker
        if config.kernel_faithful:
            linker = linker.model_copy(update={"edge_admission_prob": _KERNEL_EDGE_ADMISSION_THRESHOLD})
        link = LinkerStage(linker.build(spacing, affinity, mutual, volume_shape, evidence.ranker))
        # Division recovery reads the edge affinity, so it needs a learned head and runs before the short-track
        # filter (whose division-preserving carve-out can only protect a fork that already exists).
        divide = (
            (config.division.build(spacing, affinity, config.min_track_length, config.linker.gate_um),)
            if config.division is not None and affinity is not None
            else ()
        )
        # Geometry-only single-child division repair — needs no affinity (it reads the linked graph's geometry),
        # so it runs even where the affinity fork above cannot. Same slot: before the short-track filter.
        geo_divide = (config.division_recovery.build(spacing),) if config.division_recovery is not None else ()
        # Reuse-bridge runs before the short-track filter so the isolated t+1 nodes it links through survive as
        # part of a bridged track rather than being pruned as length-1 fragments first.
        reuse = (config.reuse.build(spacing, evidence.veto),) if config.reuse is not None else ()
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
        return (link, *reuse, *divide, *geo_divide, short, bridge, *repair, smooth)

    def spacing(self, path: Path) -> Spacing:
        """The video's physical voxel spacing — exposed so a caller can build a matching metric matcher."""
        return CellVideo.from_ome_zarr(path).spacing
