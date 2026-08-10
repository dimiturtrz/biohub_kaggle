"""CLI: score the shipped tracker pipeline across an operating-point sweep on the local proxy.

`python -m celltrack.eval.proxy_eval` mounts the dual-seed AssignmentLinker tracker (`CellTracker`) once and
scores it over a range rather than at a single guessed point — `--threshold 0.97,0.98,0.99` sweeps the
detection operating point, `--disappearance a,b,c` the linker's disappearance cost. Any other knob is a
general `--set key=value` override (repeatable, dotted keys reach the nested linker config), so
`--set smooth_strength=0.3 --set linker.name=flow` needs no per-parameter flag; a knob is swept by running
the CLI across its values. `--cv` scores the fixed-8 local CV (four test movies plus four denser-annotated
train videos) instead of the four test movies alone; the broader set is not recall-saturated, so its
threshold sweep is not flat. `TestMovieProxy` owns the movies and the score loop; `CellTracker` owns the recipe.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import get_args, get_type_hints

from pydantic import BaseModel

from celltrack.eval.proxy import CV_MOVIES, TEST_MOVIES, TestMovieProxy
from celltrack.eval.sweep_tracking import SweepTracking
from celltrack.linkers.linkers import LinkerConfig
from celltrack.tracker import CellTracker, TrackerConfig
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# The acquisition a movie belongs to is its filename prefix; the two acquisitions differ by ~10x in
# annotation density, and the node-count term of the score bites hardest where annotation is densest.
_ACQUISITION_PREFIX = 4


class _ConfigOverride:
    """Applying one `--set key=value` to a `TrackerConfig`, keeping each field's declared type across the parse."""

    @staticmethod
    def concrete_type(owner: type, field: str, current: object) -> type:
        """The field's concrete type — from its value, or (when that is None) the non-None half of its annotation.

        An optional field left at its `None` default carries no runtime type, so the declared annotation
        (`float | None`, `ShortTrackRescueConfig | None`) is the only place the real type lives.
        """
        if current is not None:
            return type(current)
        return next(argument for argument in get_args(get_type_hints(owner)[field]) if argument is not type(None))

    @staticmethod
    def cast(target: type, raw: str) -> object:
        """Parse a `--set` string into `target`, so an overridden config value keeps its declared type."""
        if target is bool:  # bool is an int subclass, so "false" must parse as a flag, not int("false")
            return raw.lower() in ("1", "true", "yes")
        return target(raw)

    @staticmethod
    def nested(config: TrackerConfig, head: str) -> BaseModel:
        """The nested config at `head`, instantiating an optional one left at `None` from its declared type.

        So `--set rescue.min_mean_probability=0.9` can reach into an off-by-default sub-config, not only one that
        is already set — the config's default for that field's type stands in, then the dotted key updates it.
        """
        current = getattr(config, head)
        return current if current is not None else _ConfigOverride.concrete_type(type(config), head, None)()

    @staticmethod
    def apply(config: TrackerConfig, assignment: str) -> TrackerConfig:
        """Apply one `key=value` override; a dotted `linker.x` reaches the nested (pydantic) linker config."""
        key, separator, raw = assignment.partition("=")
        if not separator:
            raise ValueError(f"--set expects key=value, got {assignment!r}")
        # `replace` here is dynamic dispatch: the field name is a runtime string, so the value's static type
        # cannot match the specific field it lands in — an untyped boundary the checker can't see through.
        if "." in key:
            head, tail = key.split(".", 1)
            nested = _ConfigOverride.nested(config, head)
            value = _ConfigOverride.cast(_ConfigOverride.concrete_type(type(nested), tail, getattr(nested, tail)), raw)
            updated = nested.model_copy(update={tail: value})
            return replace(config, **{head: updated})  # pyrefly: ignore[bad-argument-type]
        base_value = _ConfigOverride.cast(_ConfigOverride.concrete_type(type(config), key, getattr(config, key)), raw)
        return replace(config, **{key: base_value})  # pyrefly: ignore[bad-argument-type]


@dataclass(frozen=True)
class TrackerProxyEval:
    """Scores the tracker across a threshold × disappearance grid on a chosen proxy, mounting once."""

    device: str
    thresholds: tuple[float, ...] = (0.99,)
    disappearance_costs: tuple[float, ...] = (0.0,)
    stems: tuple[str, ...] = TEST_MOVIES
    overrides: tuple[str, ...] = ()

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
                config = self.config_at(threshold, cost)
                results[(threshold, cost)] = proxy.score(pipeline.with_config(config)).score
        return results

    def config_at(self, threshold: float, cost: float) -> TrackerConfig:
        """The resolved config of one grid cell — the swept point with every `--set` override folded on top."""
        config = TrackerConfig(threshold=threshold, linker=LinkerConfig(disappearance_cost=cost))
        for assignment in self.overrides:
            config = _ConfigOverride.apply(config, assignment)
        return config

    def breakdown(self, root: DataRoot) -> dict[str, VideoMetrics]:
        """Per-movie metrics of the tracker at its first swept threshold — separates recall from bonus-farming."""
        proxy, pipeline = self._mount(root)
        config = self.config_at(self.thresholds[0], self.disappearance_costs[0])
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
    overrides: tuple[str, ...] = field(default=())
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
        parser.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override any tracker knob (repeatable, dotted keys reach the linker), e.g. --set smooth_strength=0.3",
        )
        parsed = parser.parse_args()
        return cls(
            config=parsed.config,
            device=parsed.device,
            thresholds=tuple(float(value) for value in parsed.threshold.split(",")),
            disappearance_costs=tuple(float(value) for value in parsed.disappearance.split(",")),
            stems=CV_MOVIES if parsed.cv else TEST_MOVIES,
            overrides=tuple(parsed.overrides),
            stems_label="cv-8" if parsed.cv else "test-4",
            per_movie=parsed.per_movie,
        )


def main() -> None:
    """Mount the tracker, sweep its operating point on the chosen proxy, and log each cell of the grid."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _Args.from_argv()
    root = DataRoot.from_config(args.config)
    evaluator = TrackerProxyEval(args.device, args.thresholds, args.disappearance_costs, args.stems, args.overrides)
    logger.info("proxy=%s set=%s", args.stems_label, list(args.overrides))
    breakdown: dict[str, VideoMetrics] = evaluator.breakdown(root) if args.per_movie else {}
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
    tracking = SweepTracking(args.stems_label)
    first_cell = (args.thresholds[0], args.disappearance_costs[0])  # the cell `breakdown` was measured at
    for (threshold, cost), score in evaluator.scores(root).items():
        logger.info("threshold=%-6.4f disappearance=%-6.2f proxy score=%.4f", threshold, cost, score)
        per_movie = breakdown if (threshold, cost) == first_cell else {}
        tracking.cell(evaluator.config_at(threshold, cost), score, per_movie)


if __name__ == "__main__":
    main()
