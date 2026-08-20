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
from celltrack.operating_point import TrackerConfig
from celltrack.tracker import CellTracker
from core.metrics.score import SplitScore, VideoMetrics
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_SHIPPED = TrackerConfig.shipped()

# The acquisition a movie belongs to is its filename prefix; the two acquisitions differ by ~10x in
# annotation density, and the node-count term of the score bites hardest where annotation is densest.
_ACQUISITION_PREFIX = 4

# Where the downloaded CC0 re-ranker artifact sits UNDER THE DATA ROOT — a relative reference resolved against
# `paths.yaml`, so no machine's directory layout is written into the repo (the Kaggle kernel resolves its own).
# Public because the ranker drift diagnosis mounts the SAME artifact, and two spellings of one path is exactly
# how a diagnosis ends up characterising a different mount than the sweep it is diagnosing.
RANKER_ARTIFACT = "reference/association_ranker"


class ConfigOverride:
    """Applying one `--set key=value` to a `TrackerConfig`, keeping each field's declared type across the parse.

    Public because a DIAGNOSIS has to be able to run the same config a sweep cell ran: `dense_diagnosis` takes
    the identical `--set` strings through this, so the mechanism check and the score it explains are not two
    spellings of one operating point.
    """

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
        if target is tuple:  # `tuple("0.5,0.5")` is that string's CHARACTERS; a weight pair is comma-separated
            return tuple(float(part) for part in raw.split(","))
        return target(raw)

    @staticmethod
    def nested(config: TrackerConfig, head: str) -> BaseModel:
        """The nested config at `head`, instantiating an optional one left at `None` from its declared type.

        So `--set rescue.min_mean_probability=0.9` can reach into an off-by-default sub-config, not only one that
        is already set — the config's default for that field's type stands in, then the dotted key updates it.
        """
        current = getattr(config, head)
        return current if current is not None else ConfigOverride.concrete_type(type(config), head, None)()

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
            nested = ConfigOverride.nested(config, head)
            value = ConfigOverride.cast(ConfigOverride.concrete_type(type(nested), tail, getattr(nested, tail)), raw)
            updated = nested.model_copy(update={tail: value})
            return replace(config, **{head: updated})  # pyrefly: ignore[bad-argument-type]
        base_value = ConfigOverride.cast(ConfigOverride.concrete_type(type(config), key, getattr(config, key)), raw)
        return replace(config, **{key: base_value})  # pyrefly: ignore[bad-argument-type]


@dataclass(frozen=True)
class TrackerProxyEval:
    """Scores the tracker across a threshold × disappearance grid on a chosen proxy, mounting once."""

    device: str
    # The swept coordinates DEFAULT to the shipped operating point's own values, so an unswept run measures
    # what we submit. They used to be (0.99,) and (0.0,) — neither shipped, and the second one silently
    # reduced the global flow linker to the per-frame assignment linker it provably equals at boundary 0.
    thresholds: tuple[float, ...] = (_SHIPPED.threshold,)
    disappearance_costs: tuple[float, ...] = (_SHIPPED.linker.disappearance_cost,)
    stems: tuple[str, ...] = TEST_MOVIES
    overrides: tuple[str, ...] = ()

    def _mount(self, root: DataRoot) -> tuple[TestMovieProxy, CellTracker]:
        """The loaded proxy and the tracker mounted once — models and per-seed caches shared across the sweep.

        The local-association re-ranker is mounted only when the resolved sweep prices it (`--set
        linker.ranker_bonus=…`), so a sweep that does not ask for it neither reads the artifact nor pays the
        second linking pass it would trigger.
        """
        proc = root.processed("biohub_cell_tracking")
        proxy = TestMovieProxy.load(root, self.stems)
        # Mount under the resolved base config, not the bare shipped default: mount-time knobs (pair_context)
        # must land on the detector recipe here, since the per-cell `with_config` sweep reuses this detector and
        # only re-points the post-detection stages. The swept coordinates are re-applied per cell regardless.
        base = self.config_at(self.thresholds[0], self.disappearance_costs[0])
        pipeline = CellTracker.from_packs(
            proc / "reference/pilkwang/split_0",
            proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0",
            proc / "cache/responses",
            self.device,
            base,
        )
        if self.config_at(self.thresholds[0], self.disappearance_costs[0]).linker.needs_ranker:
            pipeline = pipeline.with_ranker(proc / RANKER_ARTIFACT)
        return proxy, pipeline

    def scores(self, root: DataRoot) -> dict[tuple[float, float], SplitScore]:
        """Both scores for each `(threshold, disappearance)` pair — the tracker mounted once, cache replayed.

        The FAITHFUL score stays the headline (it is what the leaderboard reports), but the clamped
        selection score is reported beside it because the difference between them IS the node-count bonus —
        the one component measured to anti-transfer (threshold 0.995 vs 0.99: +0.006 proxy, -0.007 LB).
        A cell whose two scores diverge is a cell whose proxy standing is being paid for by under-detection.
        """
        proxy, pipeline = self._mount(root)
        results: dict[tuple[float, float], SplitScore] = {}
        for threshold in self.thresholds:
            for cost in self.disappearance_costs:
                config = self.config_at(threshold, cost)
                results[(threshold, cost)] = proxy.score(pipeline.with_config(config))
        return results

    def config_at(self, threshold: float, cost: float) -> TrackerConfig:
        """The resolved config of one grid cell — the swept point with every `--set` override folded on top.

        The BASE is the shipped recipe, not a bare TrackerConfig: bare defaults are the per-frame assignment
        linker with no fusion (the 0.892 tier), so every cell of every sweep that did not explicitly `--set
        linker.name=flow` was measuring a pipeline we do not submit. Only the two swept coordinates are
        replaced on top of it.
        """
        config = replace(
            _SHIPPED, threshold=threshold, linker=_SHIPPED.linker.model_copy(update={"disappearance_cost": cost})
        )
        for assignment in self.overrides:
            config = ConfigOverride.apply(config, assignment)
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
        # DERIVED from the shipped operating point, not restated: a literal here overrides the dataclass
        # default silently, which is how an unswept run kept measuring threshold 0.99 at boundary 0 — the
        # per-frame assignment linker — long after the shipped recipe became flow at boundary 3.
        parser.add_argument(
            "--threshold",
            default=str(_SHIPPED.threshold),
            help="comma-separated detection thresholds to sweep (default: the shipped operating point's)",
        )
        parser.add_argument(
            "--disappearance",
            default=str(_SHIPPED.linker.disappearance_cost),
            help="comma-separated disappearance costs to sweep (default: the shipped operating point's)",
        )
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
            "  %-16s raw_jac=%.4f adj_jac=%.4f nodes=%-6d ratio=%+.3f div=%d/%d/%d",
            stem,
            metric.edges.jaccard(),
            metric.adjusted_edge_jaccard(),
            metric.predicted_nodes,
            metric.total_node_ratio(),
            metric.divisions.tp,
            metric.divisions.fp,
            metric.divisions.fn,
        )
    for prefix, split in evaluator.by_acquisition(breakdown).items():
        logger.info("  acquisition %-6s score=%.4f", prefix, split.score)
    tracking = SweepTracking(args.stems_label)
    first_cell = (args.thresholds[0], args.disappearance_costs[0])  # the cell `breakdown` was measured at
    for (threshold, cost), split in evaluator.scores(root).items():
        resolved = evaluator.config_at(threshold, cost)  # `--set` wins over the swept coordinate, so report THIS
        # BOTH TERMS, never just the sum. score = adjusted_edge_jaccard + 0.1 * division_jaccard, and printing
        # only the total makes a division arm that RAISES div_jac while costing more edge jaccard
        # indistinguishable from one that does nothing — which is exactly the ambiguity that left the
        # 2026-08-11 division re-test with an unresolved mechanism after it had already been measured. Both
        # terms are already on `SplitScore`; only the reporting dropped them.
        logger.info(
            "threshold=%-6.4f disappearance=%-6.2f proxy score=%.4f (clamped %.4f, bonus %+.4f) "
            "= edge %.4f + 0.1 * div %.4f",
            resolved.threshold,
            resolved.linker.disappearance_cost,
            split.score,
            split.selection_score,
            split.score - split.selection_score,
            split.adjusted_edge_jaccard,
            split.division_jaccard,
        )
        per_movie = breakdown if (threshold, cost) == first_cell else {}
        tracking.cell(resolved, split.score, per_movie)


if __name__ == "__main__":
    main()
