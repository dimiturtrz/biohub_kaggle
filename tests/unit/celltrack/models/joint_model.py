"""Unit test for the joint forward — one backbone pass yields both detection maps and the edge logits.

The real backbone and transformer live in the pinned `external/` checkout (absent in CI); the autouse stubs
in `conftest` stand in with the same shape contracts, so this exercises the joint plumbing (single UNet pass
→ two detect-head reads + node-feature indexing → transformer) without them.
"""

from pathlib import Path

import torch

from celltrack.models.edge_transformer import _POS_EMBED_DIM, EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.models.temporal_unet_detector import TemporalUNetDetector


def _joint_model() -> JointModel:
    out_channels = 2
    detector = TemporalUNetDetector(out_channels, (2, 4))  # stubbed backbone via conftest
    feat_dim = out_channels + 4 * _POS_EMBED_DIM
    transformer = EdgeTransformerScorer._transformer_cls()(feat_dim=feat_dim, hidden_dim=8, n_heads=1, n_blocks=1)
    return JointModel(detector, transformer, downsample=(1, 1, 1))


def test_forward():
    """One pair → detection maps for both frames (frame shape) and an s×u edge-logit matrix."""
    model = _joint_model().eval()
    frame_t, frame_t1 = torch.zeros(4, 8, 8), torch.zeros(4, 8, 8)
    source_positions = torch.tensor([[2, 4, 4]])  # 1 source
    target_positions = torch.tensor([[2, 4, 4], [1, 2, 2]])  # 2 targets
    detection_t, detection_t1, edge_logits = model.forward(frame_t, frame_t1, source_positions, target_positions)
    assert detection_t.shape == (4, 8, 8) and detection_t1.shape == (4, 8, 8)
    assert edge_logits.shape == (1, 2)


def test_from_checkpoint(tmp_path: Path):
    """A checkpoint written by the trainer rebuilds both heads at the shape its weights were trained in."""
    out_channels = 2
    detector = TemporalUNetDetector(out_channels, (2, 4))
    transformer = EdgeTransformerScorer._transformer_cls()(
        feat_dim=out_channels + 4 * _POS_EMBED_DIM, hidden_dim=128, n_heads=4, n_blocks=4
    )
    saved = JointModel(detector, transformer, downsample=(1, 4, 4))
    path = tmp_path / "joint.pt"
    torch.save(  # exactly the payload JointTrainer._save_checkpoint writes
        {
            "detector_state": saved.detector.state_dict(),
            "transformer_state": saved.transformer.state_dict(),
            "config": {"out_channels": out_channels, "layers": [2, 4], "downsample": [1, 4, 4]},
        },
        path,
    )

    loaded = JointModel.from_checkpoint(path)

    assert loaded.downsample == (1, 4, 4)
    for restored, original in zip(
        loaded.detector.state_dict().values(), saved.detector.state_dict().values(), strict=True
    ):
        assert torch.equal(restored, original)
