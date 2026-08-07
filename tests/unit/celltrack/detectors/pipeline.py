from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from celltrack.detectors.pipeline import BlendDetectorScorer
from celltrack.detectors.response_cache import ResponseCache
from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector


@dataclass
class _FakeSeed:
    """Duck-typed detector returning one canned logit volume with a single peak, for the blend test."""

    peak_logit: float

    def logit_volumes(self, path: Path, recipe: DetectorRecipe, device: str) -> list[np.ndarray]:
        volume = np.full((2, 4, 4), -10.0, dtype=np.float32)
        volume[1, 2, 3] = self.peak_logit
        return [volume]

    def voxel_scale(self, path: Path) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0)


def test_blend_detector_scorer_nodes(tmp_path: Path):
    """The blend averages both seeds' logits and reads a peak both seeds fire on into a node."""
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    seeds = (
        (cast(TemporalUNetDetector, _FakeSeed(10.0)), ResponseCache(tmp_path, "seed1")),
        (cast(TemporalUNetDetector, _FakeSeed(10.0)), ResponseCache(tmp_path, "seed2")),
    )
    graph = BlendDetectorScorer(detectors=seeds, recipe=recipe, device="cpu").nodes("vid", Path("v.zarr"), 0.5)
    assert graph.coordinates.tolist() == [[0, 1, 2, 3]]  # (t, z, y, x) — the shared peak


def test_blend_detector_scorer_forwards_each_seed_once(tmp_path: Path):
    """The cache holds the forward, so a second read-out replays without re-forwarding either seed."""

    @dataclass
    class _CountingSeed(_FakeSeed):
        forwards: list[int] | None = None

        def logit_volumes(self, path: Path, recipe: DetectorRecipe, device: str) -> list[np.ndarray]:
            assert self.forwards is not None
            self.forwards.append(1)
            return super().logit_volumes(path, recipe, device)

    counts: list[int] = []
    seed = _CountingSeed(10.0, forwards=counts)
    scorer = BlendDetectorScorer(
        detectors=((cast(TemporalUNetDetector, seed), ResponseCache(tmp_path, "seed1")),),
        recipe=DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False),
        device="cpu",
    )
    scorer.nodes("vid", Path("v.zarr"), 0.5)
    scorer.nodes("vid", Path("v.zarr"), 0.5)
    assert counts == [1]  # forwarded once, second call hit the cache
