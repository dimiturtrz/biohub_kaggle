"""Unit tests for the DeepCenter centre-prior veto — the recipe, the read-out, and the cached mount.

The architecture is pilkwang's (mounted, not trained here), so these cover the code this repo owns: the
inference recipe, the max-pool confirmation read-out, the pad-and-crop forward, and the forward-once cache.
Synthetic CPU fixtures throughout — a tiny `base_channels` model and a small zarr, never the real weights.
"""

import json
from pathlib import Path

import numpy as np
import torch
import zarr

from celltrack.center_prior import (
    CenterConfirmer,
    CenterPrior,
    CenterPriorRecipe,
    CenterPriorScorer,
    CenterPriorVeto,
    CenterVetoConfig,
)
from celltrack.response_cache import ResponseCache


def _tiny() -> CenterPrior:
    """A minimal centre prior for exercising the recipe/read-out contract without the trained model size."""
    torch.manual_seed(0)
    return CenterPrior(base_channels=2).eval()


def _confirmer(heatmap: np.ndarray, threshold: float) -> CenterConfirmer:
    """A one-frame confirmer over a hand-built heatmap at pool factor 4 and the frontier's (1, 2) window."""
    return CenterConfirmer(
        heatmaps=(heatmap.astype(np.float32),), pool_factor=4, threshold=threshold, window_z=1, window_yx=2
    )


def _peaked_heatmap() -> np.ndarray:
    """A pooled heatmap whose only strong response sits at pooled voxel (0, 1, 1)."""
    heatmap = np.zeros((1, 9, 9), dtype=np.float32)
    heatmap[0, 1, 1] = 0.9
    return heatmap


def test_from_config():
    """The saved DeepCenter config maps to the inference recipe, reading only the preprocessing keys it needs."""
    recipe = CenterPriorRecipe.from_config(
        {"pool_factor": 4, "norm_lo_pct": 50.0, "norm_hi_pct": 99.5, "norm_clip_lo": -0.5, "norm_clip_hi": 6.0}
    )
    assert recipe == CenterPriorRecipe(pool_factor=4, norm_lo_pct=50.0, norm_hi_pct=99.5, clip_lo=-0.5, clip_hi=6.0)


def test_pooled_and_normalised():
    """A frame is XY-block-mean-pooled by the factor, then percentile-normalised into the clip range."""
    frame = np.zeros((1, 8, 8), dtype=np.float32)
    frame[0, :4, :4] = 10.0  # one pooled cell is bright, the rest dark
    pooled = CenterPriorRecipe(pool_factor=4).pooled_and_normalised(frame)
    assert pooled.shape == (1, 2, 2)  # 8/4 = 2 in each XY dim
    assert pooled.max() <= 6.0 and pooled.min() >= -0.5  # inside the clip range
    assert pooled[0, 0, 0] > pooled[0, 1, 1]  # the bright block normalises above the dark one


def test_confirm():
    """A candidate is confirmed when the max heatmap over its pooled window meets the threshold, else vetoed."""
    confirmer = _confirmer(_peaked_heatmap(), threshold=0.5)
    assert confirmer.confirm(0, np.array([0, 4, 4])) is True  # voxel (4,4)->pooled (1,1), on the peak
    assert confirmer.confirm(0, np.array([0, 28, 28])) is False  # voxel (28,28)->pooled (7,7), off the peak
    assert confirmer.confirm(5, np.array([0, 4, 4])) is True  # a missing frame fails open


def test_for_gate():
    """A veto binds one video's heatmaps; `for_gate` resolves a stage's config to a confirmer over them."""
    veto = CenterPriorVeto(heatmaps=(_peaked_heatmap(),), recipe=CenterPriorRecipe(pool_factor=4))
    confirmer = veto.for_gate(CenterVetoConfig(threshold=0.5, window_z=1, window_yx=2))
    assert confirmer.threshold == 0.5
    assert confirmer.confirm(0, np.array([0, 4, 4])) is True


def test_confirmer():
    """The veto hands out a confirmer carrying the video's heatmaps at the requested threshold and window."""
    veto = CenterPriorVeto(heatmaps=(_peaked_heatmap(),), recipe=CenterPriorRecipe(pool_factor=4))
    confirmer = veto.confirmer(threshold=0.99, window_z=1, window_yx=2)
    assert confirmer.pool_factor == 4
    assert confirmer.confirm(0, np.array([0, 4, 4])) is False  # 0.9 peak is below the 0.99 threshold


def test_resolve():
    """A stage's gate config resolves to a bound confirmer against a video's veto, or None without heatmaps."""
    gate = CenterVetoConfig(threshold=0.5, window_z=1, window_yx=2)
    veto = CenterPriorVeto(heatmaps=(_peaked_heatmap(),), recipe=CenterPriorRecipe(pool_factor=4))
    assert gate.resolve(veto) is not None
    assert gate.resolve(None) is None


def test_forward():
    """The U-Net emits one logit channel at the input resolution (dims are multiples of the pooling factor)."""
    out = _tiny().forward(torch.zeros(1, 1, 8, 8, 8))
    assert out.shape == (1, 1, 8, 8, 8)


_BLOCK_MODULES = {"enc1", "enc2", "enc3", "bottleneck", "dec1", "dec2", "dec3"}


def _pilkwang_state(model: CenterPrior) -> dict[str, object]:
    """The model's weights re-nested under a per-level `.block.` prefix, as the published checkpoint stores them."""
    nested: dict[str, object] = {}
    for key, value in model.state_dict().items():
        head, rest = key.split(".", 1)
        nested[f"{head}.block.{rest}" if head in _BLOCK_MODULES else key] = value
    return nested


def test_from_pack(tmp_path: Path):
    """The loader rebuilds the model, flattening the published `.block.` key nesting, and reads config.json."""
    model = _tiny()
    torch.save({"model_state": _pilkwang_state(model), "config": {"base_channels": 2}}, tmp_path / "best.pt")
    (tmp_path / "config.json").write_text(json.dumps({"base_channels": 2, "pool_factor": 4}))
    loaded, recipe = CenterPrior.from_pack(tmp_path)
    assert recipe.pool_factor == 4
    assert torch.equal(model.head.weight, loaded.head.weight)
    original, restored = model.state_dict(), loaded.state_dict()
    assert torch.equal(original["enc1.0.weight"], restored["enc1.0.weight"])  # a re-nested block weight round-trips


def _video(tmp_path: Path) -> Path:
    """A synthetic single-frame zarr whose XY pools cleanly for the centre-prior forward."""
    group = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    array = group.create_array("0", shape=(1, 8, 16, 16), dtype="float32")
    array[:] = 0.0
    array[0, 4, 8, 8] = 1.0
    return tmp_path / "v.zarr"


def test_heatmaps(tmp_path: Path):
    """The cacheable forward yields one pooled heatmap per frame, padded up for the U-Net then cropped back."""
    heatmaps = _tiny().heatmaps(_video(tmp_path), CenterPriorRecipe(pool_factor=4), "cpu")
    assert len(heatmaps) == 1  # one frame
    assert heatmaps[0].shape == (8, 4, 4)  # 16/4 = 4 in each XY dim, cropped back from the padded forward
    assert float(heatmaps[0].max()) <= 1.0  # a probability


def test_veto(tmp_path: Path):
    """The scorer forwards a video once, caches it, and returns a `CenterPriorVeto` over its heatmaps."""
    scorer = CenterPriorScorer(
        model=_tiny(),
        recipe=CenterPriorRecipe(pool_factor=4),
        cache=ResponseCache(tmp_path / "cache", "tiny"),
        device="cpu",
    )
    veto = scorer.veto("v", _video(tmp_path))
    assert isinstance(veto, CenterPriorVeto)
    assert len(veto.heatmaps) == 1
