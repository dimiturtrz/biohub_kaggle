"""The joint detector+edge model's forward — one backbone pass feeding both heads on a real (t, t+1) pair.

Detection-only training freezes association; the joint model runs the temporal backbone ONCE on the real
consecutive pair and reads both heads off the shared features — the detection head on each frame (the two
detection losses) and the edge transformer on the node features (the edge loss). Gradient flows through the
UNet into all three, so the features learn to serve detection and association at once, the way pilkwang
trained jointly. The forward mirrors the inference scorer's `_gap_logits` exactly (full-resolution positions
in, divided by the downsample for the feature grid) so a jointly-trained model is scored the way it is run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from celltrack.models.edge_transformer import EdgeTransformerScorer, NodeWindowSlot
from celltrack.models.temporal_unet_detector import TemporalUNetDetector

# The published head's shape, which a checkpoint's weights were trained in and must be rebuilt at.
_HIDDEN_DIM, _N_HEADS, _N_BLOCKS = 128, 4, 4


@dataclass(frozen=True)
class JointForward:
    """Everything one joint pass produces — both detection maps, the edge logits, and the node features behind them.

    The features are what the edge transformer was handed, exposed so a loss can act on the REPRESENTATION and
    not only on the head's ranking of it (see `celltrack.losses.info_nce`). Five outputs is past a readable
    tuple, so the pass names them.
    """

    detection_t: Float[Tensor, "z y x"]
    detection_t1: Float[Tensor, "z y x"]
    edge_logits: Float[Tensor, "s u"]
    source_features: Float[Tensor, "s d"]
    target_features: Float[Tensor, "u d"]


class JointModel(nn.Module):
    """A trainable detector (UNet + detect head) and edge transformer, forwarded together over a real pair."""

    def __init__(
        self, detector: TemporalUNetDetector, transformer: nn.Module, downsample: tuple[int, int, int]
    ) -> None:
        super().__init__()
        self.detector = detector
        self.transformer = transformer
        self.downsample = downsample

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> "JointModel":
        """Rebuild a trained joint model from the checkpoint the trainer writes — both heads and their shape.

        A run's saved weights are only worth what can be read back: scoring a finished checkpoint (or
        re-scoring one whose run reported nothing) needs the same two heads at the same sizes, which the
        checkpoint carries alongside them. The node-feature width is read off the saved input projection
        rather than recomputed, so a run that widened it (the prior-velocity columns) reloads without a
        second declaration of the same fact.
        """
        state = torch.load(path, map_location=device, weights_only=False)
        config = state["config"]
        detector = TemporalUNetDetector(config["out_channels"], tuple(config["layers"]))
        if config.get("temporal_position"):  # rebuild the position-embedded attention before its weights load
            detector.install_temporal_position()
        transformer = EdgeTransformerScorer._transformer_cls()(  # noqa: SLF001 — the pack's head class, mounted
            feat_dim=state["transformer_state"]["proj.weight"].shape[1],
            hidden_dim=_HIDDEN_DIM,
            n_heads=_N_HEADS,
            n_blocks=_N_BLOCKS,
        )
        detector.load_state_dict(state["detector_state"])
        transformer.load_state_dict(state["transformer_state"])
        model = cls(detector, transformer, tuple(config["downsample"]))
        return model.to(device).eval()

    @override
    def forward(
        self,
        frame_t: Float[Tensor, "z y x"],
        frame_t1: Float[Tensor, "z y x"],
        source_positions: Int[Tensor, "s 3"],
        target_positions: Int[Tensor, "u 3"],
        source_velocity: Float[Tensor, "s 3"] | None = None,
    ) -> JointForward:
        """One backbone pass on the pair → both detection maps, the source→target edge logits, and the features.

        Positions are full-resolution voxels; the feature grid divides them by the downsample, exactly as the
        inference scorer does, so the edge logits match what the shipped pipeline would compute for this model.

        `source_velocity` is the sources' prior step (`PriorVelocity.expected_incoming` over the t-1→t gap),
        given in full-resolution voxels like the positions and divided by the same downsample. Only the
        sources carry one: a target's prior gap IS the gap being scored, so its history is unavailable
        without circularity, and targets take the zero vector — the same value a first-frame node takes.
        Omitted, both frames' features are exactly what they were before the feature existed.
        """
        device = str(frame_t.device)
        window = torch.stack([frame_t, frame_t1], dim=0).unsqueeze(0).unsqueeze(2)  # (1, 2, 1, Z, Y, X)
        features = self.detector.unet(window)[0]  # (2, C, Z, Y', X')
        detection_t = self.detector.detect_head(features[0:1])[0, 0]  # (Z, Y', X')
        detection_t1 = self.detector.detect_head(features[1:2])[0, 0]
        spatial = torch.tensor(features.shape[2:], dtype=torch.float32, device=device)
        downsample = torch.tensor(self.downsample, dtype=torch.float32, device=device)
        source_voxel = source_positions.to(torch.float32)
        target_voxel = target_positions.to(torch.float32)
        source_step = None if source_velocity is None else source_velocity / downsample
        target_step = None if source_velocity is None else torch.zeros_like(target_voxel)
        feat_source = EdgeTransformerScorer.node_features(
            features[0], source_voxel / downsample, NodeWindowSlot(spatial, 0.0, device), source_step
        )
        feat_target = EdgeTransformerScorer.node_features(
            features[1], target_voxel / downsample, NodeWindowSlot(spatial, 1.0, device), target_step
        )
        edge_logits = self.transformer(feat_source, feat_target, source_voxel, target_voxel)  # (s, u)
        return JointForward(detection_t, detection_t1, edge_logits, feat_source, feat_target)
