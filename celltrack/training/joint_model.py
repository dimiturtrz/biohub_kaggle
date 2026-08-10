"""The joint detector+edge model's forward — one backbone pass feeding both heads on a real (t, t+1) pair.

Detection-only training freezes association; the joint model runs the temporal backbone ONCE on the real
consecutive pair and reads both heads off the shared features — the detection head on each frame (the two
detection losses) and the edge transformer on the node features (the edge loss). Gradient flows through the
UNet into all three, so the features learn to serve detection and association at once, the way pilkwang
trained jointly. The forward mirrors the inference scorer's `_gap_logits` exactly (full-resolution positions
in, divided by the downsample for the feature grid) so a jointly-trained model is scored the way it is run.
"""

from __future__ import annotations

from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from celltrack.detectors.tunet import TemporalUNetDetector
from celltrack.edges.edge_scoring import EdgeTransformerScorer


class JointModel(nn.Module):
    """A trainable detector (UNet + detect head) and edge transformer, forwarded together over a real pair."""

    def __init__(
        self, detector: TemporalUNetDetector, transformer: nn.Module, downsample: tuple[int, int, int]
    ) -> None:
        super().__init__()
        self.detector = detector
        self.transformer = transformer
        self.downsample = downsample

    @override
    def forward(
        self,
        frame_t: Float[Tensor, "z y x"],
        frame_t1: Float[Tensor, "z y x"],
        source_positions: Int[Tensor, "s 3"],
        target_positions: Int[Tensor, "u 3"],
    ) -> tuple[Float[Tensor, "z y x"], Float[Tensor, "z y x"], Float[Tensor, "s u"]]:
        """One backbone pass on the pair → (detection logits t, detection logits t+1, source→target edge logits).

        Positions are full-resolution voxels; the feature grid divides them by the downsample, exactly as the
        inference scorer does, so the edge logits match what the shipped pipeline would compute for this model.
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
        feat_source = EdgeTransformerScorer.node_features(features[0], source_voxel / downsample, spatial, 0.0, device)
        feat_target = EdgeTransformerScorer.node_features(features[1], target_voxel / downsample, spatial, 1.0, device)
        edge_logits = self.transformer(feat_source, feat_target, source_voxel, target_voxel)  # (s, u)
        return detection_t, detection_t1, edge_logits
