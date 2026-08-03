"""Score our pipeline on the 4 TEST movies directly — the tightest local proxy for the hidden LB.

The four competition test movies also live in train WITH sparse GEFF annotations (EDA: the test scores
additional hidden annotations on these same movies). So scoring here — on the actual test movies' known
annotations — is a far better hidden proxy than an unrelated fold. Single-seed vs dual-seed, per movie.
"""

import logging
from pathlib import Path

from celltrack.linkers import LinkerConfig
from celltrack.pipeline import BlendDetectorScorer, DetectorScorer, DetectorSpec, PipelineConfig, Scorer
from celltrack.response_cache import ResponseCache
from celltrack.stages import LinkerSpec, ShortTrackSpec, SmoothSpec
from celltrack.tunet import TemporalUNetDetector
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.metrics.matching import DistanceMatcher
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

_CONFIG = Path("D:/personal_projects/biohub_kaggle/paths.yaml")
_TEST = ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1")
_BEST = PipelineConfig(
    detector=DetectorSpec(threshold=0.99),
    stages=(
        LinkerSpec(linker=LinkerConfig(name="motion", tight_um=6.0, gate_um=10.0)),
        ShortTrackSpec(min_length=6),
        SmoothSpec(strength=0.8),
    ),
)
logger = logging.getLogger(__name__)


def _score(label: str, scorer: Scorer, paths: list[Path], truths: list[AnnotatedTracks], spacing: object) -> None:
    pipeline = _BEST.build(scorer, spacing)
    matcher = DistanceMatcher(spacing=spacing)
    metrics = []
    for path, truth in zip(paths, truths, strict=True):
        graph = pipeline.run(path.name, path)
        vm = VideoMetrics.of(graph, truth, matcher)
        metrics.append(vm)
        logger.info("  %-14s %s adj_edge_jac=%.4f nodes=%d", path.stem, label, vm.adjusted_edge_jaccard(), vm.predicted_nodes)
    scored = SplitScore.of(metrics)
    logger.info("%-12s over 4 test movies: score=%.4f div_jac=%.4f", label, scored.score, scored.division_jaccard)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = DataRoot.from_config(_CONFIG)
    proc = root.processed("biohub_cell_tracking")
    responses = proc / "cache/responses"
    by_stem = {p.stem: p for p in root.videos("train")}
    paths = [by_stem[stem] for stem in _TEST]
    truths = [AnnotatedTracks.from_geff(root.track_store(p)) for p in paths]
    spacing = CellVideo.from_ome_zarr(paths[0]).spacing
    logger.info("annotated nodes on the 4 test movies: %s", [len(t.graph.node_ids) for t in truths])

    seed1 = TemporalUNetDetector.from_pack(proc / "reference/pilkwang/split_0", "cuda")[0].to("cuda").eval()
    _, recipe = TemporalUNetDetector.from_pack(proc / "reference/pilkwang/split_0", map_location="cpu")
    single = DetectorScorer(seed1, recipe, ResponseCache(responses, "pilkwang_split0_t99"), "cuda")
    seed2 = TemporalUNetDetector.from_pack(proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0", "cuda")[0].to("cuda").eval()
    blend = BlendDetectorScorer(
        detectors=((seed1, ResponseCache(responses, "pilkwang_seed1_logits")), (seed2, ResponseCache(responses, "pilkwang_seed2_logits"))),
        recipe=recipe,
        device="cuda",
    )

    _score("single-seed", single, paths, truths, spacing)
    _score("dual-seed", blend, paths, truths, spacing)


if __name__ == "__main__":
    main()
