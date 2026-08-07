"""The pilkwang `SimpleNodeTransformer` edge predictor, mounted pure-torch to score linker candidates.

The detector half of the pilkwang pack (`unet.*` + `detect_head.*`) gives us cell centres; the same pack
carries a third head we have not used — a `SimpleNodeTransformer` that scores every source→target pair
across a frame gap from the UNet feature vectors at the two centres plus a sinusoidal position embedding.
This mounts that head the same way `tunet.py` mounts the detector (their architecture from the pinned
`external/` checkout, the weights and the inference recipe ours), and turns it into an `EdgeAffinity`: a
per-gap probability matrix aligned to the linker's node order, ready to blend into the assignment cost.

The forward mirrors the reference `predict_video` exactly — the UNet runs on the *real* consecutive pair
(not the detector's duplicated single frame), features are integer-indexed at the downsampled node grid,
concatenated with the window-relative position embedding, and the transformer's logits are soft-maxed over
the sources of each target (the training normalisation: divisions allowed, merges are not).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from celltrack.detectors.tunet import _EXT_SRC, DetectorRecipe, TemporalUNetDetector, _VideoSource
from core.data.tracks import TrackGraph

_POS_EMBED_DIM = 8


@dataclass(frozen=True)
class PrecomputedEdgeAffinity:
    """Per-gap learned association matrices, keyed by the source timepoint — one video's edge scores."""

    by_timepoint: dict[int, Float[np.ndarray, "s t"]]

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:
        """The source→target probability matrix for the `t -> t+1` gap, or `None` if the gap was not scored."""
        return self.by_timepoint.get(timepoint)


class EdgeTransformerScorer(nn.Module):
    """The pilkwang edge transformer over UNet features — a candidate-pair association probability."""

    def __init__(self, detector: TemporalUNetDetector, recipe: DetectorRecipe) -> None:
        super().__init__()
        self.detector = detector
        self.recipe = recipe
        pos_feat_dim = 4 * _POS_EMBED_DIM
        self.transformer = self._transformer_cls()(
            feat_dim=detector.out_channels + pos_feat_dim, hidden_dim=128, n_heads=4, n_blocks=4
        )

    @staticmethod
    def _transformer_cls() -> type[nn.Module]:
        """The `SimpleNodeTransformer` class, from the pinned `external/` checkout or the mounted Kaggle pack."""
        if str(_EXT_SRC) not in sys.path:
            sys.path.insert(0, str(_EXT_SRC))
        try:
            from tracking_cellmot.models import SimpleNodeTransformer  # type: ignore[missing-import]  # noqa: PLC0415
        except ImportError:
            from biohub_tracking.models import SimpleNodeTransformer  # type: ignore[missing-import]  # noqa: PLC0415
        return SimpleNodeTransformer

    @classmethod
    def from_pack(cls, pack: Path, device: str = "cpu") -> "EdgeTransformerScorer":
        """Load the whole pilkwang head — `unet.*` + `detect_head.*` into the detector, `transformer.*` here."""
        detector, recipe = TemporalUNetDetector.from_pack(pack, map_location=device)
        scorer = cls(detector, recipe)
        state = torch.load(pack / "edge_predictor_best.pth", map_location=device, weights_only=True)
        transformer_state = {k[len("transformer.") :]: v for k, v in state.items() if k.startswith("transformer.")}
        scorer.transformer.load_state_dict(transformer_state)
        return scorer.to(device).eval()

    @torch.no_grad()
    def affinities(self, path: Path, detections: TrackGraph, device: str) -> PrecomputedEdgeAffinity:
        """Score every candidate pair on every frame gap of a video, aligned to the linker's per-gap node order."""
        source = TemporalUNetDetector._open_source(path)  # noqa: SLF001
        timepoints = detections.timepoints()
        positions = detections.positions().astype(np.float32)
        by_timepoint: dict[int, Float[np.ndarray, "s t"]] = {}
        for timepoint in np.unique(timepoints)[:-1].tolist():
            src_rows = np.flatnonzero(timepoints == timepoint)
            tgt_rows = np.flatnonzero(timepoints == timepoint + 1)
            if len(src_rows) == 0 or len(tgt_rows) == 0:
                continue
            probability = self._score_gap(source, timepoint, positions[src_rows], positions[tgt_rows], device)
            by_timepoint[int(timepoint)] = probability
        return PrecomputedEdgeAffinity(by_timepoint)

    def _score_gap(
        self,
        source: _VideoSource,
        timepoint: int,
        src_positions: Float[np.ndarray, "s 3"],
        tgt_positions: Float[np.ndarray, "t 3"],
        device: str,
    ) -> Float[np.ndarray, "s t"]:
        """The source→target probability matrix for one gap: transformer logits soft-maxed over the sources."""
        logits = self._gap_logits(source, timepoint, src_positions, tgt_positions, device)
        return torch.softmax(logits, dim=0).cpu().numpy()

    def _gap_logits(
        self,
        source: _VideoSource,
        timepoint: int,
        src_positions: Float[np.ndarray, "s 3"],
        tgt_positions: Float[np.ndarray, "t 3"],
        device: str,
    ) -> Float[Tensor, "s t"]:
        """The pre-softmax association logits for one gap: UNet features + position embed → transformer.

        Kept separate from the softmax so several seeds' logits can be blended *before* normalisation — the
        detector-blend lesson, that probabilities saturate and logits don't (see `BlendedEdgeTransformerScorer`).
        """
        downsample = torch.tensor(self.recipe.downsample, dtype=torch.float32, device=device)
        frame_t = TemporalUNetDetector._read_frame(source, timepoint, self.recipe.downsample, device)  # noqa: SLF001
        frame_t1 = TemporalUNetDetector._read_frame(source, timepoint + 1, self.recipe.downsample, device)  # noqa: SLF001
        window = torch.stack([frame_t, frame_t1], dim=0).unsqueeze(0).unsqueeze(2)
        features = self.detector.unet(window)[0]  # (2, C, Z, Y, X)
        spatial = torch.tensor(frame_t.shape, dtype=torch.float32, device=device)
        src_voxel = torch.as_tensor(src_positions, device=device)
        tgt_voxel = torch.as_tensor(tgt_positions, device=device)
        feat_src = self._node_features(features[0], src_voxel / downsample, spatial, 0.0, device)
        feat_tgt = self._node_features(features[1], tgt_voxel / downsample, spatial, 1.0, device)
        return self.transformer(feat_src, feat_tgt, src_voxel, tgt_voxel)  # (s, t)

    def _node_features(
        self,
        feature_map: Float[Tensor, "c z y x"],
        grid_positions: Float[Tensor, "n 3"],
        spatial: Float[Tensor, "3"],
        time_fraction: float,
        device: str,
    ) -> Float[Tensor, "n d"]:
        """A node's UNet feature vector at its voxel, concatenated with the window-relative position embed."""
        clamped = grid_positions.round().long().clamp(min=torch.zeros(3, dtype=torch.long, device=device))
        clamped = torch.minimum(clamped, (spatial - 1).long())
        indexed = feature_map[:, clamped[:, 0], clamped[:, 1], clamped[:, 2]].T  # (n, c)
        position = self._position_embedding(grid_positions, spatial, time_fraction, device)
        return torch.cat([indexed, position], dim=-1)

    @staticmethod
    def _position_embedding(
        grid_positions: Float[Tensor, "n 3"],
        spatial: Float[Tensor, "3"],
        time_fraction: float,
        device: str,
    ) -> Float[Tensor, "n d"]:
        """Sinusoidal embedding of `(t, z, y, x)` on the downsampled window grid — the transformer's position term."""
        window_time = 2.0
        time_column = torch.full((grid_positions.shape[0], 1), time_fraction, device=device)
        coordinates = torch.cat([time_column, grid_positions], dim=-1)  # (n, 4)
        norms = coordinates / torch.cat([torch.tensor([window_time], device=device), spatial]).clamp(min=1.0)
        freqs = (2.0 ** torch.arange(_POS_EMBED_DIM // 2, device=device, dtype=torch.float32)) * torch.pi
        parts: list[Tensor] = []
        for axis in range(4):
            angles = norms[:, axis].unsqueeze(-1) * freqs
            parts.extend([angles.sin(), angles.cos()])
        return torch.cat(parts, dim=-1)
