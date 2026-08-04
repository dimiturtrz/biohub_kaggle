"""CLI: score the shipped tracker pipeline across an operating-point sweep on the local proxy.

`python -m celltrack.proxy_eval` mounts the dual-seed AssignmentLinker tracker (`CellTracker`) once and
scores it over a range rather than at a single guessed point — `--threshold 0.97,0.98,0.99` sweeps the
detection operating point, `--disappearance a,b,c` the linker's disappearance cost. `--cv` scores the
fixed-8 local CV (four test movies plus four denser-annotated train videos) instead of the four test movies
alone; the broader set is not recall-saturated, so its threshold sweep is not flat. `TestMovieProxy` owns the
movies and the score loop; `CellTracker` owns the recipe.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from celltrack.linkers import LinkerConfig
from celltrack.proxy import CV_MOVIES, TEST_MOVIES, TestMovieProxy
from celltrack.tracker import CellTracker, TrackerConfig
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# The acquisition a movie belongs to is its filename prefix; the two acquisitions differ by ~10x in
# annotation density, and the node-count term of the score bites hardest where annotation is densest.
_ACQUISITION_PREFIX = 4


@dataclass(frozen=True)
class TrackerProxyEval:
    """Scores the tracker across a threshold × disappearance grid on a chosen proxy, mounting once."""

    device: str
    thresholds: tuple[float, ...] = (0.99,)
    disappearance_costs: tuple[float, ...] = (0.0,)
    stems: tuple[str, ...] = TEST_MOVIES
    linker: str = "assignment"
    reuse_gap: bool = False

    def _mount(self, root: DataRoot) -> tuple[TestMovieProxy, CellTracker]:
        """The loaded proxy and the tracker mounted once — models and per-seed caches shared across the sweep."""
        proc = root.processed("biohub_cell_tracking")
        proxy = TestMovieProxy.load(root, self.stems)
        pipeline = CellTracker.from_packs(
            proc / "reference/pilkwang/split_0",
            proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0",
            proc / "cache/responses",
            self.device,
        )
        return proxy, pipeline

    def scores(self, root: DataRoot) -> dict[tuple[float, float], float]:
        """The proxy score for each `(threshold, disappearance)` pair — the tracker mounted once, cache replayed."""
        proxy, pipeline = self._mount(root)
        results: dict[tuple[float, float], float] = {}
        for threshold in self.thresholds:
            for cost in self.disappearance_costs:
                linker = LinkerConfig(name=self.linker, disappearance_cost=cost)
                config = TrackerConfig(threshold=threshold, linker=linker, reuse_gap=self.reuse_gap)
                results[(threshold, cost)] = proxy.score(pipeline.with_config(config)).score
        return results

    def breakdown(self, root: DataRoot) -> dict[str, VideoMetrics]:
        """Per-movie metrics of the tracker at its first swept threshold — separates recall from bonus-farming."""
        proxy, pipeline = self._mount(root)
        linker = LinkerConfig(name=self.linker, disappearance_cost=self.disappearance_costs[0])
        config = TrackerConfig(threshold=self.thresholds[0], linker=linker, reuse_gap=self.reuse_gap)
        return proxy.metrics(pipeline.with_config(config))

    @staticmethod
    def by_acquisition(breakdown: dict[str, VideoMetrics]) -> dict[str, SplitScore]:
        """The split score per acquisition — the operating point read where the node-count term bites unevenly.

        The two acquisitions differ by ~10x in annotation density, so a pooled number hides a shift that helps
        one and hurts the other; grouping the per-movie metrics by prefix is the density-stratified operating
        point the old node-budget sweep reported (now on the shipped tracker, not a fold-0 own-detector).
        """
        groups: dict[str, list[VideoMetrics]] = {}
        for stem, metric in breakdown.items():
            groups.setdefault(stem[:_ACQUISITION_PREFIX], []).append(metric)
        return {prefix: SplitScore.of(metrics) for prefix, metrics in groups.items()}


@dataclass(frozen=True)
class _Args:
    """The parsed CLI options, so the sweep is one object handed to the evaluator."""

    config: Path
    device: str
    thresholds: tuple[float, ...]
    disappearance_costs: tuple[float, ...]
    stems: tuple[str, ...]
    linker: str = field(default="assignment")
    reuse_gap: bool = field(default=False)
    stems_label: str = field(default="test")
    per_movie: bool = field(default=False)

    @classmethod
    def from_argv(cls) -> "_Args":
        """Read the sweep options — threshold and disappearance ranges, and which proxy to score."""
        parser = argparse.ArgumentParser(description="Sweep the tracker's operating point on the local proxy.")
        parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--threshold", default="0.99", help="comma-separated detection thresholds to sweep")
        parser.add_argument("--disappearance", default="0.0", help="comma-separated disappearance costs to sweep")
        parser.add_argument("--cv", action="store_true", help="score the fixed-8 CV instead of the four test movies")
        parser.add_argument("--per-movie", action="store_true", help="also log each movie's raw Jaccard and node ratio")
        parser.add_argument("--linker", default="assignment", help="the linker to score (e.g. assignment, flow)")
        parser.add_argument("--reuse-gap", action="store_true", help="enable the reuse gap-bridge stage (GapCloser)")
        parsed = parser.parse_args()
        return cls(
            config=parsed.config,
            device=parsed.device,
            thresholds=tuple(float(value) for value in parsed.threshold.split(",")),
            disappearance_costs=tuple(float(value) for value in parsed.disappearance.split(",")),
            stems=CV_MOVIES if parsed.cv else TEST_MOVIES,
            linker=parsed.linker,
            reuse_gap=parsed.reuse_gap,
            stems_label="cv-8" if parsed.cv else "test-4",
            per_movie=parsed.per_movie,
        )


def main() -> None:
    """Mount the tracker, sweep its operating point on the chosen proxy, and log each cell of the grid."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _Args.from_argv()
    root = DataRoot.from_config(args.config)
    evaluator = TrackerProxyEval(
        args.device, args.thresholds, args.disappearance_costs, args.stems, args.linker, args.reuse_gap
    )
    logger.info("proxy=%s linker=%s reuse_gap=%s", args.stems_label, args.linker, args.reuse_gap)
    if args.per_movie:
        breakdown = evaluator.breakdown(root)
        for stem, metric in breakdown.items():
            logger.info(
                "  %-16s raw_jac=%.4f adj_jac=%.4f nodes=%-6d ratio=%+.3f",
                stem,
                metric.edges.jaccard(),
                metric.adjusted_edge_jaccard(),
                metric.predicted_nodes,
                metric.total_node_ratio(),
            )
        for prefix, split in evaluator.by_acquisition(breakdown).items():
            logger.info("  acquisition %-6s score=%.4f", prefix, split.score)
    for (threshold, cost), score in evaluator.scores(root).items():
        logger.info("threshold=%-6.4f disappearance=%-6.2f proxy score=%.4f", threshold, cost, score)


if __name__ == "__main__":
    main()
