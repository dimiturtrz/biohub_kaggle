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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from celltrack.models.prior_velocity import GapHistory, PriorVelocity
from celltrack.models.temporal_unet_detector import (
    _EXT_SRC,
    IDENTITY_VIEW,
    DetectorRecipe,
    FlipView,
    TemporalUNetDetector,
    _VideoSource,
)
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

    @staticmethod
    def over(path: Path, detections: TrackGraph, device: str) -> Iterator[EdgeGap]:
        """Every scorable `t -> t+1` gap of a video, in ASCENDING time — one enumeration, every scorer.

        Ascending order is load-bearing, not incidental: a gap's prior velocity is read from the gap before it, so
        a scorer can only carry that state forward if it meets the gaps in time order. Rows follow the linker's own
        `np.flatnonzero(timepoints == t)` order, so each matrix drops straight onto the assignment cost.
        """
        source = TemporalUNetDetector._open_source(path)  # noqa: SLF001
        timepoints = detections.timepoints()
        positions = detections.positions().astype(np.float32)
        for timepoint in np.unique(timepoints)[:-1].tolist():
            src_rows = np.flatnonzero(timepoints == timepoint)
            tgt_rows = np.flatnonzero(timepoints == timepoint + 1)
            if len(src_rows) == 0 or len(tgt_rows) == 0:
                continue
            yield EdgeGap(source, int(timepoint), positions[src_rows], positions[tgt_rows], device)

    source: _VideoSource
    timepoint: int
    src_positions: Float[np.ndarray, "s 3"]
    tgt_positions: Float[np.ndarray, "t 3"]
    device: str


@dataclass(frozen=True)
class _GapWindow:
    """One gap's backbone output and node voxels — the expensive half of a scoring pass, reusable across velocities.

    A gap is forwarded through the head twice whenever the prior-velocity feature is on (once with the carried
    history, once with zeros to define the history the NEXT gap reads). The 3D backbone dominates that cost and
    depends on neither velocity, so it runs once and both head passes read this.
    """

    features: Float[Tensor, "2 c z y x"]
    early: Float[Tensor, "e 3"]
    late: Float[Tensor, "l 3"]
    spatial: Float[Tensor, "3"]
    downsample: Float[Tensor, "3"]
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
        """Score every candidate pair on every frame gap of a video, aligned to the linker's per-gap node order.

        The gaps arrive in ascending time, so each one hands the next its `GapHistory` — the state the sources'
        prior velocity is read from. A head widened for that feature is therefore scored with the history it was
        trained on instead of crashing on a missing input; a head without the extra columns carries an empty
        history, never runs the extra pass, and computes exactly the matrices it computed before this existed.
        """
        history = GapHistory()
        previous_timepoint: int | None = None
        by_timepoint: dict[int, Float[np.ndarray, "s t"]] = {}
        for gap in EdgeGap.over(path, detections, device):
            if previous_timepoint is not None and gap.timepoint != previous_timepoint + 1:
                history = GapHistory()  # a skipped frame breaks the chain; the carried history is over other nodes
            logits, history = self._seed_logits(gap, history)
            by_timepoint[gap.timepoint] = torch.softmax(logits, dim=0).cpu().numpy()
            previous_timepoint = gap.timepoint
        return PrecomputedEdgeAffinity(by_timepoint)

    def _seed_logits(self, gap: EdgeGap, history: GapHistory) -> tuple[Float[Tensor, "s t"], GapHistory]:
        """This gap's forward logits under the carried `history`, AND the history the NEXT gap reads.

        Both come out of ONE backbone pass. Returning them together is what lets a blend of seeds carry a
        SEPARATE history per seed without the caller knowing how a history is made: a velocity has to be the
        one that seed's own weights produced, since that is what its trainer fed it.
        """
        window = self._gap_window(gap, reverse=False)
        return self._head_logits(window, history), self._history_after(window)

    def _history_after(self, window: _GapWindow) -> GapHistory:
        """The history this gap hands the next one: its sources' voxels and its ZERO-velocity logits.

        Zero-velocity, NOT the logits actually scored, because zero-velocity is what the trainer's history is:
        a step optimises one gap, and the previous gap it reads the velocity off is forwarded with zeros for
        its own sources (`PriorVelocitySource.of`), which stops the recursion one step back. Chaining the
        scored logits instead would be free and would feed the head a history distribution it never saw —
        drift dressed as an optimisation.

        A head that was never widened reads no velocity, carries no history, and skips this second head pass.
        """
        if not self.uses_prior_velocity:
            return GapHistory()
        return GapHistory(window.early, self._head_logits(window, GapHistory()))

    @property
    def uses_prior_velocity(self) -> bool:
        """Whether the MOUNTED head reads a prior velocity — answered by the weights, never by a caller's flag.

        The feature is a property of the trained weights: `PriorVelocitySource.widen` appended three input
        columns to the head's projection, and `JointModel.from_checkpoint` rebuilds the head at whatever width
        was saved. Asking the projection how wide it is therefore cannot disagree with the checkpoint, while a
        flag threaded down from a config can — and the disagreement's failure mode is the crash this replaces.
        """
        projection = cast(nn.Linear, self.transformer.proj)
        return projection.in_features == self.detector.out_channels + 4 * _POS_EMBED_DIM + PriorVelocity.DIM

    def _gap_window(self, gap: EdgeGap, *, reverse: bool, view: FlipView = IDENTITY_VIEW) -> _GapWindow:
        """Read the gap's two frames and run the backbone once — the half of a pass no velocity can change.

        `view` mirrors the frames the backbone sees AND the node positions read out of it, so a flipped view is
        the same scene under a different presentation rather than a different scene: the returned logit matrix
        stays in the caller's node order and needs no un-flipping. Positions mirror in FULL-RESOLUTION voxels
        about `downsample * (spatial - 1)`, which is exactly the grid extent they land on once divided by the
        downsample. The identity view rebuilds today's window, tensor for tensor.
        """
        device, recipe = gap.device, self.recipe
        read_t = TemporalUNetDetector._read_frame(gap.source, gap.timepoint, recipe.downsample, device)  # noqa: SLF001
        read_t1 = TemporalUNetDetector._read_frame(gap.source, gap.timepoint + 1, recipe.downsample, device)  # noqa: SLF001
        frame_t, frame_t1 = view.apply(read_t), view.apply(read_t1)
        frames = [frame_t1, frame_t] if reverse else [frame_t, frame_t1]
        early, late = (gap.tgt_positions, gap.src_positions) if reverse else (gap.src_positions, gap.tgt_positions)
        window = torch.stack(frames, dim=0).unsqueeze(0).unsqueeze(2)
        spatial = torch.tensor(frame_t.shape, dtype=torch.float32, device=device)
        downsample = torch.tensor(recipe.downsample, dtype=torch.float32, device=device)
        extent = downsample * (spatial - 1.0)
        with AutocastPolicy.of(device):
            features = self.detector.unet(window)[0]  # (2, C, Z, Y, X)
        return _GapWindow(
            features=features,
            early=view.mirror(torch.as_tensor(early, device=device), extent),
            late=view.mirror(torch.as_tensor(late, device=device), extent),
            spatial=spatial,
            downsample=downsample,
            device=device,
        )

    def _head_logits(self, window: _GapWindow, history: GapHistory) -> Float[Tensor, "early late"]:
        """The transformer's logits over one window's nodes, the early slot carrying `history`'s velocity."""
        early_step, late_step = self._velocity_columns(window, history)
        with AutocastPolicy.of(window.device):
            feat_early = self.node_features(
                window.features[0],
                window.early / window.downsample,
                NodeWindowSlot(window.spatial, 0.0, window.device),
                early_step,
            )
            feat_late = self.node_features(
                window.features[1],
                window.late / window.downsample,
                NodeWindowSlot(window.spatial, 1.0, window.device),
                late_step,
            )
            logits = self.transformer(feat_early, feat_late, window.early, window.late)  # (early, late)
        return logits.float()

    def _velocity_columns(
        self, window: _GapWindow, history: GapHistory
    ) -> tuple[Float[Tensor, "e 3"] | None, Float[Tensor, "l 3"] | None]:
        """The two slots' prior-step inputs — `(None, None)` unless the mounted head was widened to read them.

        Only the EARLY slot carries a history: the late nodes' previous gap IS the gap being scored, so their
        history is unavailable without circularity and they take the zero vector, exactly as `JointModel.forward`
        gives its targets. The velocity arrives in full-resolution voxels and is divided by the downsample into
        the feature grid's space — the same two steps, in the same order, the trainer's forward takes.
        """
        if not self.uses_prior_velocity:
            return None, None
        return history.velocity(window.early) / window.downsample, torch.zeros_like(window.late)

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

        NO PRIOR VELOCITY is carried through this entry point, in either direction. A reverse pass's sources are
        the LATER frame's nodes, whose only previous gap is the gap under test — there is no history to read
        that is not circular — so a widened head takes the zero vector here, the same value a first-gap or an
        entering node takes. Feeding the forward direction's history instead would attach the earlier frame's
        motion to a different frame's nodes, which is not a weaker signal but a wrong one. A FORWARD pass that
        wants its history goes through `_seed_logits`, which is where the state is carried; this entry point
        stays the stateless one, used for the reverse direction and by the edge finetuner.
        """
        return self._head_logits(self._gap_window(gap, reverse=reverse), GapHistory())

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
    def batched_node_features(
        feature_maps: Float[Tensor, "b c z y x"],
        grid_positions: Float[Tensor, "b n 3"],
        slot: NodeWindowSlot,
        prior_velocity: Float[Tensor, "b n 3"] | None = None,
    ) -> Float[Tensor, "b n d"]:
        """`node_features` over a PADDED batch — one gather and one embed for every pair at once, no Python loop.

        The per-pair `node_features` is a small index-plus-embed run once per pair; over a batch that is a loop
        of tiny kernels, and the loop's launch overhead is the residual GPU idle after the transformer itself is
        batched. This gathers `feature_maps[b, :, z, y, x]` for every pair with a single advanced index and
        embeds the whole `(b, n, 3)` grid at once, so `[b, :len]` equals the unbatched `node_features` for pair b.
        Pad rows index voxel 0 (clamped) and are masked out by the transformer.

        `prior_velocity` is the padded `(b, n, 3)` velocity block — appended RAW and scaled exactly as the per-pair
        `node_features` appends its column, so a batched velocity forward equals the per-pair one. `None` omits it
        (the widen-free path), the exact tensor this built before velocity was carried.
        """
        clamped = torch.minimum(grid_positions.round().long().clamp(min=0), (slot.spatial - 1).long())
        batch, nodes = grid_positions.shape[0], grid_positions.shape[1]
        rows = torch.arange(batch, device=feature_maps.device)[:, None].expand(batch, nodes)
        gathered = feature_maps[rows, :, clamped[..., 0], clamped[..., 1], clamped[..., 2]]  # (b, n, c)
        columns = [gathered, EdgeTransformerScorer._batched_position_embedding(grid_positions, slot)]
        if prior_velocity is not None:
            columns.append(prior_velocity / PriorVelocity.SCALE)
        return torch.cat(columns, dim=-1)

    @staticmethod
    def _batched_position_embedding(
        grid_positions: Float[Tensor, "b n 3"],
        slot: NodeWindowSlot,
    ) -> Float[Tensor, "b n d"]:
        """`_position_embedding` over a `(b, n, 3)` batch — the same sinusoidal embed with a leading batch axis."""
        window_time = 2.0
        batch, nodes = grid_positions.shape[0], grid_positions.shape[1]
        time_column = torch.full((batch, nodes, 1), slot.time_fraction, device=slot.device)
        coordinates = torch.cat([time_column, grid_positions], dim=-1)  # (b, n, 4)
        norms = coordinates / torch.cat([torch.tensor([window_time], device=slot.device), slot.spatial]).clamp(min=1.0)
        freqs = (2.0 ** torch.arange(_POS_EMBED_DIM // 2, device=slot.device, dtype=torch.float32)) * torch.pi
        parts: list[Tensor] = []
        for axis in range(4):
            angles = norms[..., axis].unsqueeze(-1) * freqs
            parts.extend([angles.sin(), angles.cos()])
        return torch.cat(parts, dim=-1)

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
