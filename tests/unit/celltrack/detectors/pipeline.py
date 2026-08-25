from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import numpy as np

from celltrack.detectors.pipeline import BlendDetectorScorer, DetectionFusionOptions
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


def test_blend_detector_scorer_aligns_secondary_moments(tmp_path: Path):
    """With `align_moments` the secondary seed is rescaled onto the primary's mean and spread before the blend.

    The two seeds share a peak location but the secondary is scaled up 3x, so under a plain weighted blend the
    secondary dominates the summed logit. Alignment puts it back on the primary's scale, so the aligned blend's
    peak logit lands near the primary's own — the property the calibrated dual-seed detection step relies on.
    """
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    seeds = (
        (cast(TemporalUNetDetector, _FakeSeed(10.0)), ResponseCache(tmp_path, "seed1")),
        (cast(TemporalUNetDetector, _FakeSeed(30.0)), ResponseCache(tmp_path, "seed2")),
    )
    scorer = BlendDetectorScorer(detectors=seeds, recipe=recipe, device="cpu")
    frames = tuple(seed.logit_volumes(Path("v"), recipe, "cpu")[0] for seed, _ in seeds)
    primary_peak = float(frames[0].max())
    plain = scorer._blend(frames, (0.5, 0.5))
    aligned = scorer._blend(frames, (0.5, 0.5), align_moments=True)
    assert plain.max() > aligned.max()  # the un-scaled secondary inflates the plain blend's peak
    assert abs(aligned.max() - primary_peak) < abs(plain.max() - primary_peak)  # aligned sits nearer the primary


def test_blend_detector_scorer_forwards_each_seed_once(tmp_path: Path):
    """The cache holds the forward, so a second read-out replays without re-forwarding either seed."""

    @dataclass
    class _CountingSeed(_FakeSeed):
        forwards: list[int] | None = None

        @override
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


def _collapsing_scorer(tmp_path: Path) -> BlendDetectorScorer:
    """A primary seed that fires a peak and a secondary that drags the equal-weight blend below the floor.

    Primary logit +10 is a peak at any sane threshold; secondary −12 pulls the mean to −1 at that voxel, so the
    plain blend detects nothing there while the primary alone detects one cell — the collapse the guard reverts.
    """
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    seeds = (
        (cast(TemporalUNetDetector, _FakeSeed(10.0)), ResponseCache(tmp_path, "seed1")),
        (cast(TemporalUNetDetector, _FakeSeed(-12.0)), ResponseCache(tmp_path, "seed2")),
    )
    return BlendDetectorScorer(detectors=seeds, recipe=recipe, device="cpu")


def test_kernel_faithful_reverts_a_collapsed_frame_to_the_primary_detection(tmp_path: Path):
    """A frame whose blended peak count falls below 90% of the primary's is detected from the primary seed alone."""
    scorer = _collapsing_scorer(tmp_path)
    plain = scorer.nodes("vid", Path("v.zarr"), 0.5)
    guarded = scorer.nodes("vid", Path("v.zarr"), 0.5, fusion=DetectionFusionOptions(kernel_faithful=True))
    assert plain.coordinates.tolist() == []  # the washed-out blend detects nothing on the collapsed frame
    assert guarded.coordinates.tolist() == [[0, 1, 2, 3]]  # the guard reverts to the primary seed's peak


def test_kernel_faithful_keeps_the_blend_when_retention_holds(tmp_path: Path):
    """Both seeds firing the same peak retains 100% of the primary count, so the guard is a byte-identical no-op."""
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    seeds = (
        (cast(TemporalUNetDetector, _FakeSeed(10.0)), ResponseCache(tmp_path, "seed1")),
        (cast(TemporalUNetDetector, _FakeSeed(10.0)), ResponseCache(tmp_path, "seed2")),
    )
    scorer = BlendDetectorScorer(detectors=seeds, recipe=recipe, device="cpu")
    plain = scorer.nodes("vid", Path("v.zarr"), 0.5)
    guarded = scorer.nodes("vid", Path("v.zarr"), 0.5, fusion=DetectionFusionOptions(kernel_faithful=True))
    assert guarded.coordinates.tolist() == plain.coordinates.tolist() == [[0, 1, 2, 3]]  # no collapse, no revert
