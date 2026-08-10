"""The pilkwang `SimpleNodeTransformer` edge predictor, mounted pure-torch to score linker candidates.

The detector half of the pilkwang pack (`unet.*` + `detect_head.*`) gives us cell centres; the same pack
carries a third head we have not used — a `SimpleNodeTransformer` that scores every source→target pair
across a frame gap from the UNet feature vectors at the two centres plus a sinusoidal position embedding.
This mounts that head the same way the detector mounts the backbone (their architecture from the pinned
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

from celltrack.models.prior_velocity import PriorVelocity
from celltrack.models.temporal_unet_detector import _EXT_SRC, DetectorRecipe, TemporalUNetDetector, _VideoSource
from celltrack.precision import AutocastPolicy
from core.data.tracks import TrackGraph

_POS_EMBED_DIM = 8


@dataclass(frozen=True)
class NodeWindowSlot:
    """Which slot of the two-frame window a node's features are read from — grid extent, time fraction, device."""

    spatial: Float[Tensor, "3"]
    time_fraction: float
    device: str


@dataclass(frozen=True)
class PrecomputedEdgeAffinity:
    """Per-gap learned association matrices, keyed by the source timepoint — one video's edge scores."""

    by_timepoint: dict[int, Float[np.ndarray, "s t"]]

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:
        """The source→target probability matrix for the `t -> t+1` gap, or `None` if the gap was not scored."""
        return self.by_timepoint.get(timepoint)


@dataclass(frozen=True)
class EdgeGap:
    """One `t -> t+1` gap's scoring inputs — the video, the timepoint, and both frames' node positions."""

    source: _VideoSource
    timepoint: int
    src_positions: Float[np.ndarray, "s 3"]
    tgt_positions: Float[np.ndarray, "t 3"]
    device: str


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
    def of(
        cls, detector: TemporalUNetDetector, transformer: nn.Module, recipe: DetectorRecipe
    ) -> "EdgeTransformerScorer":
        """Mount already-trained in-memory heads as a scorer — a jointly-trained pair, no pack on disk.

        `from_pack` reads pilkwang's published weights; this takes the modules a joint run is optimising right
        now, sharing them, so an eval scores the LIVE association head through the same `_gap_logits` path.
        """
        scorer = cls(detector, recipe)
        scorer.transformer = transformer
        return scorer

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
            gap = EdgeGap(source, int(timepoint), positions[src_rows], positions[tgt_rows], device)
            by_timepoint[gap.timepoint] = self._score_gap(gap)
        return PrecomputedEdgeAffinity(by_timepoint)

    def _score_gap(self, gap: EdgeGap) -> Float[np.ndarray, "s t"]:
        """The source→target probability matrix for one gap: transformer logits soft-maxed over the sources."""
        return torch.softmax(self._gap_logits(gap), dim=0).cpu().numpy()

    def _gap_logits(self, gap: EdgeGap, *, reverse: bool = False) -> Float[Tensor, "early late"]:
        """The pre-softmax association logits for one gap: UNet features + position embed → transformer.

        Kept separate from the softmax so several seeds' logits can be blended *before* normalisation — the
        detector-blend lesson, that probabilities saturate and logits don't (see `BlendedEdgeTransformerScorer`).
        The backbone and transformer run at the device's accelerated precision; the logits return fp32, the
        space the seed blend and the softmax over sources are taken in.

        `reverse` scores the same gap with the temporal pair swapped: the window is stacked `[t+1, t]`, so the
        t+1 nodes take the window's leading slot (features[0], time fraction 0.0) and the t nodes the trailing
        one, and the returned matrix is `(t, s)`.

        How much this differs from the forward pass is LESS than it looks, and the honest account matters. The
        backbone's temporal attention carries no positional encoding, so it is permutation-invariant over time:
        re-ordering the window returns the same features, merely swapped. What actually changes is which nodes
        the transformer treats as sources and which time fraction (0.0 / 1.0) their features carry. So this is
        not the wholly new question an earlier version of this docstring claimed — though it is not a pure
        re-normalisation of the forward logits either, since the head's source/target roles genuinely swap.
        A consequence worth taking: the UNet recompute here is avoidable, because the forward pass's features
        could be reused with the fractions swapped.
        """
        device, recipe = gap.device, self.recipe
        downsample = torch.tensor(recipe.downsample, dtype=torch.float32, device=device)
        frame_t = TemporalUNetDetector._read_frame(gap.source, gap.timepoint, recipe.downsample, device)  # noqa: SLF001
        frame_t1 = TemporalUNetDetector._read_frame(gap.source, gap.timepoint + 1, recipe.downsample, device)  # noqa: SLF001
        frames = [frame_t1, frame_t] if reverse else [frame_t, frame_t1]
        early, late = (gap.tgt_positions, gap.src_positions) if reverse else (gap.src_positions, gap.tgt_positions)
        window = torch.stack(frames, dim=0).unsqueeze(0).unsqueeze(2)
        spatial = torch.tensor(frame_t.shape, dtype=torch.float32, device=device)
        early_voxel = torch.as_tensor(early, device=device)
        late_voxel = torch.as_tensor(late, device=device)
        with AutocastPolicy.of(device):
            features = self.detector.unet(window)[0]  # (2, C, Z, Y, X)
            feat_early = self.node_features(features[0], early_voxel / downsample, NodeWindowSlot(spatial, 0.0, device))
            feat_late = self.node_features(features[1], late_voxel / downsample, NodeWindowSlot(spatial, 1.0, device))
            logits = self.transformer(feat_early, feat_late, early_voxel, late_voxel)  # (early, late)
        return logits.float()

    @staticmethod
    def node_features(
        feature_map: Float[Tensor, "c z y x"],
        grid_positions: Float[Tensor, "n 3"],
        slot: NodeWindowSlot,
        prior_velocity: Float[Tensor, "n 3"] | None = None,
    ) -> Float[Tensor, "n d"]:
        """A node's UNet feature vector at its voxel, the window-relative position embed, and its prior step.

        Stateless (the whole node→transformer-input step), so the joint trainer builds the same inputs the
        shipped scorer does — one definition, both callers.

        `prior_velocity` is the node's expected incoming displacement (see `PriorVelocity`), in the same
        space as `grid_positions`, appended RAW and scaled rather than sinusoidally embedded: the pair head
        already consumes its candidate displacement `rel` as three raw numbers over the same divisor, so a
        raw column keeps history and candidate directly comparable — a sinusoidal encoding would make that
        comparison nonlinear and alias large steps — and it costs the widened projection only three columns.
        Omitting it returns exactly the tensor this built before the feature existed.
        """
        device = slot.device
        clamped = grid_positions.round().long().clamp(min=torch.zeros(3, dtype=torch.long, device=device))
        clamped = torch.minimum(clamped, (slot.spatial - 1).long())
        indexed = feature_map[:, clamped[:, 0], clamped[:, 1], clamped[:, 2]].T  # (n, c)
        columns = [indexed, EdgeTransformerScorer._position_embedding(grid_positions, slot)]
        if prior_velocity is not None:
            columns.append(prior_velocity / PriorVelocity.SCALE)
        return torch.cat(columns, dim=-1)

    @staticmethod
    def _position_embedding(
        grid_positions: Float[Tensor, "n 3"],
        slot: NodeWindowSlot,
    ) -> Float[Tensor, "n d"]:
        """Sinusoidal embedding of `(t, z, y, x)` on the downsampled window grid — the transformer's position term."""
        window_time = 2.0
        device = slot.device
        time_column = torch.full((grid_positions.shape[0], 1), slot.time_fraction, device=device)
        coordinates = torch.cat([time_column, grid_positions], dim=-1)  # (n, 4)
        norms = coordinates / torch.cat([torch.tensor([window_time], device=device), slot.spatial]).clamp(min=1.0)
        freqs = (2.0 ** torch.arange(_POS_EMBED_DIM // 2, device=device, dtype=torch.float32)) * torch.pi
        parts: list[Tensor] = []
        for axis in range(4):
            angles = norms[:, axis].unsqueeze(-1) * freqs
            parts.extend([angles.sin(), angles.cos()])
        return torch.cat(parts, dim=-1)
