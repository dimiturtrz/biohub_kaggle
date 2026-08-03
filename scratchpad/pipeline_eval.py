"""Exercise the config-space spine end to end: full pipeline from cache + per-slot oracle attribution.

Two things this proves for wor.7. (1) A `PipelineConfig` for the current best reproduces the local fold-0
score, replayed from a cached detector response — the detector forwards once, every stack variant after is
CPU seconds. (2) The oracle attribution sweep teacher-forces ground-truth nodes through each prefix of the
stage list, so each slot's ceiling is read in isolation: where is 0.9 actually lost — detection or linking?
"""

import logging
import time
from pathlib import Path

from celltrack.bracket import ValidationFold
from celltrack.linkers import LinkerConfig
from celltrack.pipeline import DetectorScorer, DetectorSpec, PipelineConfig
from celltrack.response_cache import ResponseCache
from celltrack.stages import DivisionSpec, GraphStage, LinkerSpec, ShortTrackSpec, SmoothSpec
from celltrack.tunet import TemporalUNetDetector
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

_CONFIG = Path("D:/personal_projects/biohub_kaggle/paths.yaml")
logger = logging.getLogger(__name__)

# The current best local fold-0 pipeline: motion linker @thr0.99, short-track filter (min 6), line-fit smooth.
_BEST = PipelineConfig(
    detector=DetectorSpec(threshold=0.99),
    stages=(
        LinkerSpec(linker=LinkerConfig(name="motion", tight_um=6.0, gate_um=10.0)),
        ShortTrackSpec(min_length=6),
        SmoothSpec(strength=0.8),
    ),
)

# A division-inclusive probe: motion linker then division recovery, to read the fork ceiling on GT nodes.
_DIV_PROBE = PipelineConfig(
    detector=DetectorSpec(threshold=0.99),
    stages=(
        LinkerSpec(linker=LinkerConfig(name="motion", tight_um=6.0, gate_um=10.0)),
        DivisionSpec(),
    ),
)


def _score(graphs: list[tuple[TrackGraph, AnnotatedTracks]], matcher: DistanceMatcher) -> SplitScore:
    """Aggregate per-video metrics for a set of (prediction, truth) pairs into the competition score."""
    return SplitScore.of([VideoMetrics.of(graph, truth, matcher) for graph, truth in graphs])


def _full(config: PipelineConfig, scorer: DetectorScorer, fold: ValidationFold) -> None:
    """Run the whole pipeline over the fold, replaying detection from cache, and log the reproduced score."""
    pipeline = config.build(scorer, fold.spacing)
    matcher = DistanceMatcher(spacing=fold.spacing)
    graphs: list[tuple[TrackGraph, AnnotatedTracks]] = []
    for done, (path, truth) in enumerate(zip(fold.videos, fold.annotations, strict=True), start=1):
        t0 = time.perf_counter()
        graphs.append((pipeline.run(path.name, path), truth))
        logger.info("  [%d/%d] %s %.1fs", done, len(fold.videos), path.name, time.perf_counter() - t0)
    scored = _score(graphs, matcher)
    logger.info("FULL (from cache): score=%.4f div_jac=%.4f", scored.score, scored.division_jaccard)


def _attribution(config: PipelineConfig, fold: ValidationFold) -> None:
    """Teacher-force GT nodes through each prefix of the stage list — the per-slot ceiling given perfect detection.

    On GT-only nodes (~6% of the estimated cell count) the node-count adjustment hands a large, unrepresentative
    bonus, so the adjusted *score* is meaningless here — the linking ceiling is read off the unadjusted edge
    Jaccard (and the division Jaccard for the division slot), exactly as the bracket run reports it.
    """
    stages = config.build_stages(fold.spacing)
    matcher = DistanceMatcher(spacing=fold.spacing)
    names = ["nodes", *(type(stage).__name__ for stage in stages)]
    for depth in range(len(stages) + 1):
        metrics = [
            VideoMetrics.of(_teacher_forced(truth, stages[:depth]), truth, matcher) for truth in fold.annotations
        ]
        edge_jaccard = EdgeCounts.pooled(metric.edges for metric in metrics).jaccard()
        div_jaccard = SplitScore.of(metrics).division_jaccard
        label = " -> ".join(names[: depth + 1])
        logger.info("ORACLE %-52s edge_jac=%.4f div_jac=%.4f", label, edge_jaccard, div_jaccard)


def _teacher_forced(truth: AnnotatedTracks, stages: tuple[GraphStage, ...]) -> TrackGraph:
    """Run a stage prefix on this video's ground-truth nodes — perfect detection, isolating the stages' ceiling."""
    graph = truth.graph.without_edges()
    for stage in stages:
        graph = stage.transform(graph)
    return graph


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = DataRoot.from_config(_CONFIG)
    pack = root.processed("biohub_cell_tracking") / "reference/pilkwang/split_0"
    fold = ValidationFold.load(root, 0)
    detector, recipe = TemporalUNetDetector.from_pack(pack, map_location="cuda")
    detector = detector.to("cuda").eval()
    cache = ResponseCache(root.processed("biohub_cell_tracking") / "cache/responses", "pilkwang_split0_t99")
    scorer = DetectorScorer(detector=detector, recipe=recipe, cache=cache, device="cuda")

    logger.info("=== attribution: current best (GT-teacher-forced, perfect detection) ===")
    _attribution(_BEST, fold)
    logger.info("=== attribution: division probe (the fork ceiling on GT nodes) ===")
    _attribution(_DIV_PROBE, fold)
    logger.info("=== full pipeline (real detection, from cache) ===")
    _full(_BEST, scorer, fold)


if __name__ == "__main__":
    main()
