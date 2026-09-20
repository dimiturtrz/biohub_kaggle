"""Unit tests for the detector network's own logic — the recipe config and the weight/forward contract.

The `TemporalUNet3D` backbone is third-party (the pinned `external/` checkout), stubbed by the autouse
fixture in `conftest`; these cover the code this repo owns: the reproducible-inference recipe, the forward,
and the pack/checkpoint (de)serialisation. The peak read-out is tested with the detector strategy that owns
it (`tests/unit/celltrack/detectors/tunet.py`).
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from celltrack.models.temporal_unet_detector import IDENTITY_VIEW, DetectorRecipe, FlipView, TemporalUNetDetector


class _AttnBlock(nn.Module):
    """Stands in for the external `_TemporalAttention`: the `norm`/`attn` pair `install` detects and wraps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, 2, batch_first=True)


def test_install_temporal_position():
    """`install_temporal_position` replaces every real attention block, skips Identity, and flips the flag.

    The full-resolution stage is an `nn.Identity` (no attention), so it must be left alone; every other block
    becomes a position-embedded wrapper. The stubbed backbone carries no temporal blocks, so a representative
    pair is assigned onto it — the install logic is what is under test, not the third-party module.
    """
    detector = TemporalUNetDetector(8, (8, 16))
    detector.unet.temporal_blocks = nn.ModuleList([nn.Identity(), _AttnBlock(16)])

    detector.install_temporal_position()

    assert detector.temporal_position is True
    assert [type(block).__name__ for block in detector.unet.temporal_blocks] == [
        "Identity",
        "TemporalPositionAttention",
    ]


def test_install_temporal_position_is_idempotent():
    """A second install is a no-op — it must not wrap the wrappers it already installed."""
    detector = TemporalUNetDetector(8, (8, 16))
    detector.unet.temporal_blocks = nn.ModuleList([_AttnBlock(16)])

    detector.install_temporal_position()
    detector.install_temporal_position()

    assert [type(block).__name__ for block in detector.unet.temporal_blocks] == ["TemporalPositionAttention"]


def test_install_temporal_position_reuses_the_blocks_weights():
    """The wrapper shares the original block's trained `norm`/`attn`, so a warm start keeps its weights."""
    detector = TemporalUNetDetector(8, (8, 16))
    original = _AttnBlock(16)
    detector.unet.temporal_blocks = nn.ModuleList([original])

    detector.install_temporal_position()

    wrapped = detector.unet.temporal_blocks[0]
    assert wrapped.attn is original.attn
    assert torch.equal(wrapped.position, torch.zeros_like(wrapped.position))  # zero-init: byte-identical at step 0


def _tiny_detector() -> TemporalUNetDetector:
    """A minimal detector for exercising the weight/inference contract without the real backbone size."""
    torch.manual_seed(0)
    return TemporalUNetDetector(out_channels=2, layers=(2, 4))


def test_norm_defaults_to_batch():
    """The zero-argument detector keeps the published BatchNorm3d backbone — `norm` opts INTO the swap."""
    assert TemporalUNetDetector(8, (8, 16)).norm == "batch"


def test_invalid_norm_raises():
    """An unknown norm is a construction error, not a silently-ignored string that ships the wrong backbone."""
    with pytest.raises(ValueError, match="norm must be one of"):
        TemporalUNetDetector(8, (8, 16), norm="layer")


def test_groupnorm_backbone_swaps_every_batchnorm():
    """`_groupnorm_backbone` replaces each BatchNorm3d with a same-width GroupNorm at `gcd(width, 8)` groups.

    The stubbed backbone carries no BatchNorm3d (a 1x1x1 conv), so a representative nested tree is assigned
    onto `detector.unet` — the surgery logic is what is under test, not the third-party module. GroupNorm has
    affine weight/bias but no running buffers, which is the whole point: no source-domain statistic is stored.
    """
    detector = TemporalUNetDetector(8, (8, 16))
    detector.unet.block = nn.Sequential(
        nn.Conv3d(1, 32, 1), nn.BatchNorm3d(32), nn.Conv3d(32, 64, 1), nn.BatchNorm3d(64)
    )

    detector._groupnorm_backbone()

    norms = [m for m in detector.unet.block if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm))]
    assert all(isinstance(m, nn.GroupNorm) for m in norms)  # no BatchNorm3d survives
    assert [(m.num_groups, m.num_channels) for m in norms] == [(8, 32), (8, 64)]  # gcd(width, 8) groups


def test_group_norm_detector_constructs_and_runs():
    """A `norm='group'` detector builds and forwards — the from-scratch operating point is reachable end to end."""
    detector = TemporalUNetDetector(out_channels=2, layers=(2, 4), norm="group").eval()
    assert detector.norm == "group"
    assert detector.forward(torch.zeros(4, 8, 8)).shape == (4, 8, 8)


def test_checkpoint_round_trips_norm(tmp_path: Path):
    """The `norm` choice persists so a trained checkpoint rebuilds the same backbone before its weights load."""
    path = tmp_path / "detector.pt"
    TemporalUNetDetector(out_channels=2, layers=(2, 4), norm="group").save_checkpoint(path, DetectorRecipe())
    assert torch.load(path, weights_only=True)["norm"] == "group"
    restored, _ = TemporalUNetDetector.from_checkpoint(path)
    assert restored.norm == "group"


def test_checkpoint_without_norm_defaults_to_batch(tmp_path: Path):
    """A checkpoint written before `norm` existed carries none; it rebuilds as the published BN backbone."""
    path = tmp_path / "old.pt"
    _tiny_detector().save_checkpoint(path, DetectorRecipe())
    blob = torch.load(path, weights_only=True)
    del blob["norm"]
    torch.save(blob, path)
    restored, _ = TemporalUNetDetector.from_checkpoint(path)
    assert restored.norm == "batch"


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


def test_tta_ensemble():
    """The one flip set every stage ensembles: identity first, then the three in-plane mirrorings, no rotations."""
    assert FlipView.tta_ensemble() == (FlipView(), FlipView((-1,)), FlipView((-2,)), FlipView((-2, -1)))
    assert FlipView.tta_ensemble()[0] == IDENTITY_VIEW


def test_identity_view_applies_nothing():
    """The identity returns the very same tensor, so an ensemble starting there matches the plain forward."""
    volume = torch.arange(8.0).reshape(2, 2, 2)
    assert IDENTITY_VIEW.apply(volume) is volume
    positions = torch.tensor([[1.0, 2.0, 3.0]])
    assert IDENTITY_VIEW.mirror(positions, torch.tensor([9.0, 9.0, 9.0])) is positions


def test_apply():
    """Every view undoes itself, which is what lets the detector un-flip a flipped forward."""
    volume = torch.arange(24.0).reshape(2, 3, 4)
    for view in FlipView.tta_ensemble():
        assert torch.equal(view.apply(view.apply(volume)), volume)


def test_mirror():
    """A mirrored position indexes the flipped volume at the value the original indexed the unflipped one."""
    volume = torch.arange(24.0).reshape(2, 3, 4)
    extent = torch.tensor([1.0, 2.0, 3.0])
    position = torch.tensor([[1.0, 0.0, 3.0]])
    for view in FlipView.tta_ensemble():
        z, y, x = view.mirror(position, extent)[0].long().tolist()
        assert view.apply(volume)[z, y, x] == volume[1, 0, 3]


def test_forward_heads():
    """One backbone pass feeds both heads; without an offset head the offsets are absent, not zeros."""
    detector = TemporalUNetDetector(8, (8, 16), offset_head=True)
    frames = torch.zeros((1, 2, 4, 4))

    heads = detector.forward_heads(frames, single_frame=True)

    assert heads.logits.shape == (1, 2, 4, 4)
    assert heads.offsets is not None and heads.offsets.shape == (1, 3, 2, 4, 4)
    assert TemporalUNetDetector(8, (8, 16)).forward_heads(frames, single_frame=True).offsets is None
