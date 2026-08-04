"""CLI: score the champion pipeline on the four-test-movie proxy — the tightest local read on the hidden LB.

`python -m celltrack.proxy_eval` mounts the published pilkwang packs (single seed and the dual-seed logit
blend), runs each through the champion post-detection stack on the four test movies, and prints the proxy
score. The reusable pieces live elsewhere — `TestMovieProxy` owns the movies and the score loop, `PipelineConfig`
owns the stack — so this only says which detectors and which config to compare.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from celltrack.linkers import LinkerConfig
from celltrack.pipeline import BlendDetectorScorer, DetectorScorer, PipelineConfig, Scorer
from celltrack.proxy import TestMovieProxy
from celltrack.response_cache import ResponseCache
from celltrack.stages import LinkerSpec, ShortTrackSpec, SmoothSpec
from celltrack.tunet import TemporalUNetDetector
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# The current champion post-detection stack: motion linker @thr0.99, short-track filter (min 6), line-fit smooth.
_CHAMPION = PipelineConfig(
    stages=(
        LinkerSpec(linker=LinkerConfig(name="motion", tight_um=6.0, gate_um=10.0)),
        ShortTrackSpec(min_length=6),
        SmoothSpec(strength=0.8),
    ),
)


@dataclass(frozen=True)
class ChampionProxyEval:
    """Scores the champion pipeline on the four test movies for each published-pack scorer."""

    device: str
    config: PipelineConfig = _CHAMPION

    def _scorers(self, proc: Path) -> dict[str, Scorer]:  # pragma: no cover - needs the published weight packs
        """The published-pack scorers to compare: a single seed and the two-seed logit blend."""
        responses = proc / "cache/responses"
        split0 = proc / "reference/pilkwang/split_0"
        seed1, recipe = TemporalUNetDetector.from_pack(split0, map_location=self.device)
        seed1 = seed1.to(self.device).eval()
        seed2, _ = TemporalUNetDetector.from_pack(
            proc / "reference/pilkwang/seed2/weights/unet_transformer/split_0", map_location=self.device
        )
        seed2 = seed2.to(self.device).eval()
        single = DetectorScorer(seed1, recipe, ResponseCache(responses, "pilkwang_split0_t99"), self.device)
        blend = BlendDetectorScorer(
            detectors=(
                (seed1, ResponseCache(responses, "pilkwang_seed1_logits")),
                (seed2, ResponseCache(responses, "pilkwang_seed2_logits")),
            ),
            recipe=recipe,
            device=self.device,
        )
        return {"single-seed": single, "dual-seed": blend}

    def scores(self, root: DataRoot) -> dict[str, float]:
        """The champion-config proxy score for each mounted scorer, keyed by its label."""
        proxy = TestMovieProxy.load(root)
        scorers = self._scorers(root.processed("biohub_cell_tracking"))
        return {label: proxy.score(self.config.build(scorer, proxy.spacing)).score for label, scorer in scorers.items()}


def main() -> None:
    """Mount the packs, score the champion on the four test movies, and log each scorer's proxy score."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Score the champion pipeline on the four test movies.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = DataRoot.from_config(args.config)
    for label, score in ChampionProxyEval(args.device).scores(root).items():
        logger.info("%-12s proxy score=%.4f", label, score)


if __name__ == "__main__":
    main()
