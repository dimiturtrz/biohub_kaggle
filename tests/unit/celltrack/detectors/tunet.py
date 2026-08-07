"""Unit tests for the detector's own logic — the recipe config and the peak read-out.

The `TemporalUNet3D` backbone is third-party (the pinned `external/` checkout), so it is not exercised here;
these cover the code this repo owns: the reproducible-inference recipe and local-maximum extraction.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from torch import Tensor, nn

from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector


class _StubBackbone(nn.Module):
    """A 1x1x1-conv stand-in for the third-party `TemporalUNet3D`, so construction/forward run without it.

    It honours the same window contract — `(B, T, C_in, Z, Y, X)` in, `(B, T, C_out, Z, Y, X)` out — which
    is all the detector's forward/inference path depends on; the real backbone is exercised in the field, not
    in CI (it lives in the pinned `external/` checkout, absent here).
    """

    def __init__(self, in_channels: int, out_channels: int, layers: list[int]) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, window: Tensor) -> Tensor:
        batch, frames = window.shape[:2]
        merged = window.reshape(batch * frames, *window.shape[2:])
        out = self.conv(merged)
        return out.reshape(batch, frames, *out.shape[1:])


@pytest.fixture(autouse=True)
def _stub_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the external `TemporalUNet3D` for `_StubBackbone` so no test reaches the absent third-party dep."""
    monkeypatch.setattr(TemporalUNetDetector, "_backbone_cls", staticmethod(lambda: _StubBackbone))


def _tiny_detector() -> TemporalUNetDetector:
    """A minimal detector for exercising the weight/inference contract without the real backbone size."""
    torch.manual_seed(0)
    return TemporalUNetDetector(out_channels=2, layers=(2, 4))


def _video(tmp_path: Path, scale: list[float]) -> Path:
    """A synthetic OME-zarr: two frames, one bright voxel, with a given `(t, z, y, x)` voxel scale."""
    group = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    array = group.create_array("0", shape=(2, 4, 8, 8), dtype="float32")
    array[:] = 0.0
    array[:, 2, 4, 4] = 1.0
    group.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}
    group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [{"scale": scale}]}]}]
    return tmp_path / "v.zarr"


def test_as_config():
    """A recipe serialised and rebuilt is unchanged — inference is reproducible from the saved config."""
    recipe = DetectorRecipe(downsample=(1, 4, 4), pool_kernel_um=5.0, tta=True)
    assert DetectorRecipe.from_config(recipe.as_config()) == recipe


def test_from_config():
    """The published config carries extra keys (window_size, unet_layers); from_config reads only its own."""
    config = {"downsample": [1, 4, 4], "pool_kernel_um": 5.0, "window_size": 2, "unet_layers": [32, 64, 128]}
    recipe = DetectorRecipe.from_config(config)
    assert recipe.downsample == (1, 4, 4)
    assert recipe.tta is True  # defaulted when absent


def test_fingerprint():
    """The cache fingerprint changes with downsample/TTA (they change the forward), not the read-out params."""
    base = DetectorRecipe(downsample=(1, 4, 4), tta=True, pool_kernel_um=5.0, keep_per_frame=2000)
    assert (
        base.fingerprint()
        == DetectorRecipe(downsample=(1, 4, 4), tta=True, pool_kernel_um=9.0, keep_per_frame=50).fingerprint()
    )
    assert base.fingerprint() != DetectorRecipe(downsample=(1, 2, 2), tta=True).fingerprint()
    assert base.fingerprint() != DetectorRecipe(downsample=(1, 4, 4), tta=False).fingerprint()


def test_recipe_defaults_are_pilkwangs_input_pipeline():
    """The zero-argument recipe is the ×4 Y/X downsample the detector was trained under."""
    assert DetectorRecipe().downsample == (1, 4, 4)


def test_forward():
    """The detector maps one downsampled frame to a same-shape per-voxel logit volume."""
    frame = torch.zeros(4, 8, 8)
    frame[2, 4, 4] = 1.0
    logits = _tiny_detector().eval().forward(frame)
    assert logits.shape == (4, 8, 8)


def test_save_checkpoint(tmp_path: Path):
    """A saved checkpoint holds the weights, the architecture shape, and the recipe."""
    path = tmp_path / "detector.pt"
    _tiny_detector().save_checkpoint(path, DetectorRecipe(downsample=(1, 2, 2)))
    blob = torch.load(path, weights_only=True)
    assert blob["out_channels"] == 2
    assert blob["layers"] == [2, 4]
    assert blob["recipe"]["downsample"] == [1, 2, 2]


def test_from_checkpoint(tmp_path: Path):
    """A detector round-trips through a checkpoint — same weights, same recipe — for the eval/kernel path."""
    path = tmp_path / "detector.pt"
    original = _tiny_detector()
    original.save_checkpoint(path, DetectorRecipe(downsample=(1, 4, 4)))
    restored, recipe = TemporalUNetDetector.from_checkpoint(path)
    assert recipe == DetectorRecipe(downsample=(1, 4, 4))
    assert torch.equal(original.detect_head.weight, restored.detect_head.weight)


def test_from_pack(tmp_path: Path):
    """The published-pack loader takes the `unet.*`/`detect_head.*` half and reads the recipe from config.json."""
    detector = _tiny_detector()
    state = {k: v for k, v in detector.state_dict().items() if k.startswith(("unet.", "detect_head."))}
    torch.save(state, tmp_path / "edge_predictor_best.pth")
    (tmp_path / "config.json").write_text(
        json.dumps({"unet_out_channels": 2, "unet_layers": [2, 4], "downsample": [1, 4, 4], "pool_kernel_um": 5.0})
    )
    loaded, recipe = TemporalUNetDetector.from_pack(tmp_path)
    assert recipe.downsample == (1, 4, 4)
    assert torch.equal(detector.detect_head.weight, loaded.detect_head.weight)


def test_detections(tmp_path: Path):
    """End to end on a synthetic video: a frame with one bright voxel yields an edge-free graph of centres."""
    group = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    array = group.create_array("0", shape=(2, 4, 8, 8), dtype="float32")
    array[:] = 0.0
    array[:, 2, 4, 4] = 1.0
    group.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}
    group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [{"scale": [1.0, 1.0, 1.0, 1.0]}]}]}]
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    video = _video(tmp_path, [1.0, 1.0, 1.0, 1.0])
    graph = _tiny_detector().detections(video, threshold=0.0, recipe=recipe, device="cpu")
    assert graph.coordinates.shape[1] == 4
    assert graph.edges.shape == (0, 2)
    assert set(graph.timepoints().tolist()) <= {0, 1}


def test_probability_volumes(tmp_path: Path):
    """The cacheable forward yields one downsampled probability volume per timepoint, in [0, 1], as host arrays."""
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    volumes = _tiny_detector().probability_volumes(_video(tmp_path, [1.0, 1.0, 1.0, 1.0]), recipe, "cpu")
    assert len(volumes) == 2  # one per timepoint
    assert volumes[0].shape == (4, 8, 8)  # the downsampled grid
    assert volumes[0].min() >= 0.0 and volumes[0].max() <= 1.0  # probabilities


def test_logit_volumes(tmp_path: Path):
    """The blend-space response is the pre-sigmoid logits — `probability_volumes` is exactly their sigmoid."""
    detector = _tiny_detector()
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    video = _video(tmp_path, [1.0, 1.0, 1.0, 1.0])
    logits = detector.logit_volumes(video, recipe, "cpu")
    probs = detector.probability_volumes(video, recipe, "cpu")
    assert len(logits) == 2
    assert np.allclose(probs[0], 1.0 / (1.0 + np.exp(-logits[0])), atol=1e-5)


def test_graph_from_volumes():
    """The cheap read-out lifts a peak above threshold into a stamped, edge-free node — no GPU, no video."""
    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[1, 2, 3] = 1.0
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    graph = TemporalUNetDetector.graph_from_volumes([volume], (1.0, 1.0, 1.0), 0.5, recipe, "cpu")
    assert graph.coordinates.tolist() == [[0, 1, 2, 3]]  # (t, z, y, x)
    assert graph.edges.shape == (0, 2)


def test_voxel_scale(tmp_path: Path):
    """The voxel scale is read from the OME metadata, dropping the time axis for the peak read-out."""
    assert _tiny_detector().voxel_scale(_video(tmp_path, [1.0, 2.0, 3.0, 4.0])) == (2.0, 3.0, 4.0)
