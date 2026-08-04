"""CLI: score the shipped champion pipeline on the four-test-movie proxy — the tightest local read on the LB.

`python -m celltrack.proxy_eval` mounts the dual-seed AssignmentLinker champion (`ChampionPipeline`) and
scores it on the four test movies. `--disappearance a,b,c` sweeps the linker's disappearance cost, mounting
once and replaying the cached forward per value, so a knob is characterised across a range rather than at a
single guessed point. `TestMovieProxy` owns the movies and the score loop; `ChampionPipeline` owns the recipe.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from celltrack.champion import ChampionConfig, ChampionPipeline
from celltrack.proxy import TestMovieProxy
from core.paths import DataRoot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChampionProxyEval:
    """Scores the champion pipeline on the four test movies, optionally sweeping the disappearance cost."""

    device: str
    disappearance_costs: tuple[float, ...] = (0.0,)

    def scores(self, root: DataRoot) -> dict[float, float]:
        """The proxy score for each disappearance cost, keyed by the cost — the champion mounted once."""
        proc = root.processed("biohub_cell_tracking")
        proxy = TestMovieProxy.load(root)
        pipeline = ChampionPipeline.from_packs(
            proc / "reference/pilkwang/split_0",
            proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0",
            proc / "cache/responses",
            self.device,
        )
        results: dict[float, float] = {}
        for cost in self.disappearance_costs:
            variant = pipeline.with_config(ChampionConfig(disappearance_cost=cost))
            results[cost] = proxy.score(variant).score
        return results


def main() -> None:
    """Mount the champion, score it on the four test movies for each disappearance cost, and log the sweep."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Score the champion pipeline on the four test movies.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disappearance", default="0.0", help="comma-separated disappearance costs to sweep")
    args = parser.parse_args()
    costs = tuple(float(value) for value in args.disappearance.split(","))
    root = DataRoot.from_config(args.config)
    for cost, score in ChampionProxyEval(args.device, costs).scores(root).items():
        logger.info("disappearance=%-6.2f proxy score=%.4f", cost, score)


if __name__ == "__main__":
    main()
