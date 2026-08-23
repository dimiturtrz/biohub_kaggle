"""Edge-agreement between the several tracker configs that each score 0.900 on the leaderboard.

The absolute local proxy blinds at the saturated top (a 0.900-tying config's held-out number is not
evidence for its LB rank), but DECORRELATION is still locally measurable: two trackers that link the same
cells differently disagree on edges regardless of which is "right". This module runs each 0.900-tying
config over the fixed-8 CV once, then for every pair computes the micro-averaged edge Jaccard of one
tracker's links against the other's, under the competition's own centroid node-matching. A pair whose
agreement is near 1.0 is common-mode (nothing to ensemble); a pair well below 1.0 carries the
decorrelation an ensemble ADD criterion needs. The detector responses are cached, so this is CPU-only.

CLI: `python -m celltrack.analysis.tracker_decorrelation [--device cpu]`.
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from celltrack.edges.blended_edge_scoring import EdgeBlendOptions
from celltrack.eval.proxy import CV_MOVIES, TestMovieProxy
from celltrack.operating_point import TrackerConfig
from celltrack.postproc.gap_closer import ReuseConfig
from celltrack.tracker import CellTracker
from core.data.tracks import TrackGraph
from core.metrics.edges import EdgeCounts
from core.metrics.matching import DistanceMatcher
from core.paths import DataRoot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecorrelationRun:
    """Runs each config over the CV movies once and reduces every pair to a pooled edge-agreement Jaccard."""

    device: str

    @staticmethod
    def configs(base: TrackerConfig) -> dict[str, TrackerConfig]:
        """The 0.900-tying operating points, each the shipped recipe with one knob moved (per the kernels)."""
        configs = {
            "shipped_thr097": base,
            "flow_thr080": replace(base, threshold=0.80),
            "flow_thr085": replace(base, threshold=0.85),
            "view_tta_thr090": replace(base, threshold=0.90, edge_options=EdgeBlendOptions(view_tta=True)),
            "align_seed_thr090": replace(base, threshold=0.90, edge_options=EdgeBlendOptions(align_seed_moments=True)),
            "recall_thr085": replace(base, threshold=0.85, keep_boundary_tracks=True, reuse=ReuseConfig()),
        }
        if base.division is not None:
            geometric = base.division.model_copy(update={"candidacy": "geometric"})
            configs["geometric_div"] = replace(base, division=geometric)
        else:
            logger.warning("shipped().division is None — skipping geometric-div config")
        return configs

    def _graphs(self, root: DataRoot) -> tuple[TestMovieProxy, dict[str, dict[str, TrackGraph]]]:
        """Each config's per-movie track graph, the tracker mounted once and re-pointed per config."""
        proc = root.processed("biohub_cell_tracking")
        proxy = TestMovieProxy.load(root, CV_MOVIES)
        pipeline = CellTracker.from_packs(
            proc / "reference/pilkwang/split_0",
            proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0",
            proc / "cache/responses",
            self.device,
            TrackerConfig.shipped(),
        )
        graphs: dict[str, dict[str, TrackGraph]] = {}
        for name, config in self.configs(TrackerConfig.shipped()).items():
            configured = pipeline.with_config(config)
            graphs[name] = {path.stem: configured.run(path.name, path) for path in proxy.paths}
            logger.info("ran config %s over %d movies", name, len(proxy.paths))
        return proxy, graphs

    def agreement(self, root: DataRoot) -> dict[tuple[str, str], float]:
        """Pooled (micro-averaged) edge Jaccard between every pair of configs over the CV movies.

        Symmetrised: the node-matching and edge repairs are direction-sensitive, so each pair is scored both
        ways (X links judged against Y, then Y against X) and the two pooled counts are summed before the
        ratio — the same micro-average the leaderboard uses across movies, here across the two directions too.
        """
        proxy, graphs = self._graphs(root)
        matcher = DistanceMatcher(spacing=proxy.spacing)
        stems = [path.stem for path in proxy.paths]
        pairwise: dict[tuple[str, str], float] = {}
        for left, right in itertools.combinations(graphs, 2):
            counts = []
            for stem in stems:
                a, b = graphs[left][stem], graphs[right][stem]
                counts.append(EdgeCounts.of(a, b, matcher.match(a, b)))
                counts.append(EdgeCounts.of(b, a, matcher.match(b, a)))
            pairwise[(left, right)] = EdgeCounts.pooled(counts).jaccard()
        return pairwise


def main() -> None:
    """Run the decorrelation measurement and log the pairwise agreement, lowest (most decorrelated) first."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Edge-agreement between the 0.900-tying tracker configs on CV8.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cpu", help="cpu keeps the card free; detector responses are cached")
    parsed = parser.parse_args()
    root = DataRoot.from_config(parsed.config)
    pairwise = DecorrelationRun(parsed.device).agreement(root)
    for (left, right), jaccard in sorted(pairwise.items(), key=lambda item: item[1]):
        logger.info("agree=%.4f  %-20s %-20s", jaccard, left, right)
    values = list(pairwise.values())
    logger.info("mean off-diagonal agreement = %.4f over %d pairs", sum(values) / len(values), len(values))


if __name__ == "__main__":
    main()
