"""Unit tests for the detector network's own logic — the recipe config and the weight/forward contract.

The `TemporalUNet3D` backbone is third-party (the pinned `external/` checkout), stubbed by the autouse
fixture in `conftest`; these cover the code this repo owns: the reproducible-inference recipe, the forward,
and the pack/checkpoint (de)serialisation. The peak read-out is tested with the detector strategy that owns
it (`tests/unit/celltrack/detectors/tunet.py`).
"""

import json
from pathlib import Path

import torch

from celltrack.models.temporal_unet_detector import DetectorRecipe, TemporalUNetDetector


def _tiny_detector() -> TemporalUNetDetector:
    """A minimal detector for exercising the weight/inference contract without the real backbone size."""
    torch.manual_seed(0)
    return TemporalUNetDetector(out_channels=2, layers=(2, 4))


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


def test_forward_batch():
    """A batch of frames maps to a batch of same-shape logit volumes; item 0 matches the single-frame path."""
    detector = _tiny_detector().eval()
    frames = torch.zeros(3, 4, 8, 8)
    frames[0, 2, 4, 4] = 1.0
    batched = detector.forward_batch(frames)
    assert batched.shape == (3, 4, 8, 8)
    assert torch.allclose(batched[0], detector.forward(frames[0]))
    single = detector.forward_batch(frames, single_frame=True)  # detection-only T=1 window
    assert single.shape == (3, 4, 8, 8)


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


def test_forward_pair():
    """Both frames of a real window get their own detection map — the second is not a discarded duplicate."""
    torch.manual_seed(0)
    detector = TemporalUNetDetector(out_channels=2, layers=(2, 4))
    frames = torch.stack([torch.randn(4, 8, 8), torch.randn(4, 8, 8)])

    logits = detector.forward_pair(frames)

    assert logits.shape == (2, 4, 8, 8)  # one map per frame of the pair
    assert not torch.allclose(logits[0], logits[1])  # genuinely different frames -> different responses
