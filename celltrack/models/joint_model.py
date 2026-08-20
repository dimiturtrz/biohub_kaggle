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
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from celltrack.models.edge_transformer import EdgeTransformerScorer, NodeWindowSlot
from celltrack.models.temporal_unet_detector import TemporalUNetDetector

# The published head's shape, which a checkpoint's weights were trained in and must be rebuilt at.
_HIDDEN_DIM, _N_HEADS, _N_BLOCKS = 128, 4, 4
# The fixed node count the batched training forward pads every pair to, so the whole GPU forward is ONE static
# shape (the prerequisite for the batched masked loss to read a padded block, and for a graph to capture it).
# Sized from the corpora AND the synthetic generator: real pairs max 33 (GT) / 102 (gated) nodes, but a
# synthetic training scene carries up to `SyntheticSceneConfig.n_cells` (700) per frame, and with velocity OFF
# a dense pair routes through the batched path (which pads to this fixed block) rather than the ragged per-pair
# loop. 768 clears the 700-cell generator with margin; attention over it is still microseconds beside the 64^3
# backbone. A pair beyond it RAISES rather than truncating (dropping a node would drop a real candidate).
_MAX_NODES = 768


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


@dataclass(frozen=True)
class BatchedForward:
    """The batched forward as PADDED tensors on one `_MAX_NODES` shape — plus the masks and each pair's real length.

    The batched masked loss reads this block directly (no per-pair Python loop over ragged logits); `unpad`
    recovers the ragged per-pair `JointForward` list the eval, the difficulty sampler and per-term logging still
    consume. Keeping both means the gradient path is one static shape while the diagnostics stay per pair.
    """

    detect_t: Float[Tensor, "b z y x"]
    detect_t1: Float[Tensor, "b z y x"]
    logits: Float[Tensor, "b max max"]
    source_features: Float[Tensor, "b max d"]
    target_features: Float[Tensor, "b max d"]
    source_mask: Bool[Tensor, "b max"]
    target_mask: Bool[Tensor, "b max"]
    source_lengths: tuple[int, ...]
    target_lengths: tuple[int, ...]

    def unpad(self) -> list["JointForward"]:
        """Each pair's `[:s, :u]` block as a `JointForward` — identical to what a per-pair forward would return."""
        return [
            JointForward(
                self.detect_t[i],
                self.detect_t1[i],
                self.logits[i, :s, :t],
                self.source_features[i, :s],
                self.target_features[i, :t],
            )
            for i, (s, t) in enumerate(zip(self.source_lengths, self.target_lengths, strict=True))
        ]


class JointModel(nn.Module):
    """A trainable detector (UNet + detect head) and edge transformer, forwarded together over a real pair."""

    def __init__(
        self, detector: TemporalUNetDetector, transformer: nn.Module, downsample: tuple[int, int, int]
    ) -> None:
        super().__init__()
        self.detector = detector
        self.transformer = transformer
        self.downsample = downsample
        self._scene_transformer: nn.Module | None = None  # compiled clone for the uniform-N generated scenes
        self._scene_nodes = 0

    def compile_scene_head(self, node_count: int) -> None:
        """Compile the edge transformer for the generated scenes' FIXED node count — routed by count in `_heads`.

        The scenes all carry `node_count` nodes, so a static-shape `torch.compile` fuses the 700-node attention
        (measured 4x per call) and compiles exactly once, while the ragged real pairs — every one a different
        node count that would thrash a static compile past the recompile cap — keep the eager transformer. Same
        weights, so the gradient is identical; only the launch schedule differs.
        """
        self._scene_transformer = torch.compile(self.transformer, dynamic=False)  # type: ignore[assignment]
        self._scene_nodes = node_count

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
        detector = TemporalUNetDetector(
            config["out_channels"], tuple(config["layers"]), norm=config.get("norm", "batch")
        )
        if config.get("temporal_position"):  # rebuild the position-embedded attention before its weights load
            detector.install_temporal_position()
        transformer = EdgeTransformerScorer._transformer_cls()(  # noqa: SLF001 — the pack's head class, mounted
            feat_dim=state["transformer_state"]["proj.weight"].shape[1],
            hidden_dim=_HIDDEN_DIM,
            n_heads=_N_HEADS,
            n_blocks=_N_BLOCKS,
        )
        detector.load_state_dict(cls._align_state_to(detector, state["detector_state"]))
        transformer.load_state_dict(cls._align_state_to(transformer, state["transformer_state"]))
        model = cls(detector, transformer, tuple(config["downsample"]))
        return model.to(device).eval()

    @staticmethod
    def _align_state_to(module: nn.Module, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Remap a saved head's keys onto whatever `torch.compile` wrapping the LIVE module currently carries.

        `torch.compile` prefixes a wrapped submodule's keys with `_orig_mod.`; the weights are identical (compile
        wraps the forward, not the parameters), only the names differ. A snapshot saved compiled and reloaded into
        an eager module — or the reverse, when triton is missing on one run and present on the next — otherwise
        fails `load_state_dict` on names alone. Canonicalise the checkpoint to plain keys, then map each onto the
        target module's OWN key, which carries `_orig_mod.` exactly where the target is compiled and nowhere else.
        """
        plain_to_target = {key.replace("._orig_mod.", "."): key for key in module.state_dict()}
        return {plain_to_target.get(key.replace("._orig_mod.", "."), key): value for key, value in state.items()}

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
        window = torch.stack([frame_t, frame_t1], dim=0).unsqueeze(0).unsqueeze(2)  # (1, 2, 1, Z, Y, X)
        features = self.detector.unet(window)[0]  # (2, C, Z, Y', X')
        return self._heads(features, source_positions, target_positions, source_velocity)

    def forward_batch(
        self,
        windows: Float[Tensor, "b two z y x"],
        source_positions: list[Int[Tensor, "s 3"]],
        target_positions: list[Int[Tensor, "u 3"]],
        source_velocities: list[Float[Tensor, "s 3"] | None],
    ) -> list[JointForward]:
        """One backbone pass over B pairs, then the two heads per pair — the batched training path.

        The backbone is the GPU cost and it takes any `(B, T, ...)`, so stacking B pairs' `(2, Z, Y, X)` windows
        into one `(B, 2, 1, Z, Y, X)` call fills the device where a per-pair forward leaves it idle. The heads
        run per pair because each pair has its own ragged node sets (they are cheap — small transformers over
        tens of nodes), so the result is a list, one `JointForward` per pair. `forward` is exactly the `B = 1`
        case, so a batched run computes the identical thing a per-pair run does, one optimiser step later.
        """
        # Generated scenes route per-pair to the compiled uniform-N head; a pair whose node set overflows the
        # fixed `_MAX_NODES` pad takes the same ragged per-pair path (the batched block would raise on it). The
        # `_heads` loop carries any node count — it is the reference the padded block only shortcuts. Everything
        # else — prior velocity included — takes the fully padded batched block.
        oversized = any(
            source.shape[0] > _MAX_NODES or target.shape[0] > _MAX_NODES
            for source, target in zip(source_positions, target_positions, strict=True)
        )
        if self._scene_transformer is not None or oversized:
            features = self.detector.unet(windows.unsqueeze(2))  # (B, 2, C, Z, Y', X')
            return [
                self._heads(features[i], source_positions[i], target_positions[i], source_velocities[i])
                for i in range(features.shape[0])
            ]
        return self.batched_forward(windows, source_positions, target_positions, source_velocities).unpad()

    def batched_forward(
        self,
        windows: Float[Tensor, "b two z y x"],
        source_positions: list[Int[Tensor, "s 3"]],
        target_positions: list[Int[Tensor, "u 3"]],
        source_velocities: list[Float[Tensor, "s 3"] | None] | None = None,
    ) -> BatchedForward:
        """The batched forward as ONE padded block — backbone, detection heads, node gather, transformer.

        The heads used to run once per pair — tens of tiny attention kernels, plus a per-pair feature gather —
        leaving the GPU idle between launches. Everything here is batched: detection is a plain batched conv, the
        node features are gathered for all pairs at once, and the transformer takes a padded `(B, _MAX_NODES, D)`
        batch with a key-padding mask so the pad nodes are inert. Returns the padded tensors so the loss can read
        the block directly; `BatchedForward.unpad` recovers the per-pair `[:s, :u]` views for the diagnostics.

        `source_velocities` carries the prior-velocity feature through the SAME padded block rather than the
        per-pair head loop `forward_batch` used to fall back to — the sources' step padded to `(B, _MAX_NODES, 3)`
        and divided into the feature grid's space exactly as `_heads` does, the targets' a zero block. `None`
        (the widen-free run) omits the columns, the exact forward this computed before velocity was batched.
        """
        features = self.detector.unet(windows.unsqueeze(2))  # (B, 2, C, Z, Y', X')
        detect_t = self.detector.detect_head(features[:, 0])[:, 0]  # (B, Z, Y', X')
        detect_t1 = self.detector.detect_head(features[:, 1])[:, 0]
        device = str(features.device)
        spatial = torch.tensor(features.shape[3:], dtype=torch.float32, device=device)
        downsample = torch.tensor(self.downsample, dtype=torch.float32, device=device)
        source_voxels = self._pad_fixed(source_positions)  # (B, _MAX_NODES, 3)
        target_voxels = self._pad_fixed(target_positions)
        source_lengths = tuple(p.shape[0] for p in source_positions)
        target_lengths = tuple(p.shape[0] for p in target_positions)
        source_step, target_step = self._padded_velocity(source_velocities, source_positions, downsample, device)
        feat_source = EdgeTransformerScorer.batched_node_features(
            features[:, 0], source_voxels / downsample, NodeWindowSlot(spatial, 0.0, device), source_step
        )
        feat_target = EdgeTransformerScorer.batched_node_features(
            features[:, 1], target_voxels / downsample, NodeWindowSlot(spatial, 1.0, device), target_step
        )
        index = torch.arange(_MAX_NODES, device=features.device)
        mask_source = index[None, :] < torch.tensor(source_lengths, device=features.device)[:, None]
        mask_target = index[None, :] < torch.tensor(target_lengths, device=features.device)[:, None]
        logits = self.transformer(feat_source, feat_target, source_voxels, target_voxels, mask_source, mask_target)
        return BatchedForward(
            detect_t,
            detect_t1,
            logits,
            feat_source,
            feat_target,
            mask_source,
            mask_target,
            source_lengths,
            target_lengths,
        )

    @staticmethod
    def fits_batched(source_len: int, target_len: int) -> bool:
        """Whether a pair's node counts fit the padded batched block — beyond it `_pad_fixed` would raise.

        The batched path pads to the fixed `_MAX_NODES`; a pair above it must take the ragged per-pair loop
        instead. The eligibility check asks this so a raised default batch size can never route an oversized
        pair into a pad it overflows — the guard lives beside the pad it protects.
        """
        return source_len <= _MAX_NODES and target_len <= _MAX_NODES

    @staticmethod
    def _pad_fixed(positions: list[Int[Tensor, "n 3"]]) -> Float[Tensor, "b max 3"]:
        """Every pair's positions zero-padded to the FIXED `_MAX_NODES` (positions already on the model device).

        pad_sequence reaches the batch max; the second pad reaches the constant `_MAX_NODES` so the forward is one
        shape every step. A pair above it raises rather than truncating — dropping a node would drop a real
        candidate from the matching and silently change the score. The pad rows index voxel 0 and are masked out.
        """
        padded = pad_sequence([position.to(torch.float32) for position in positions], batch_first=True)
        if padded.shape[1] > _MAX_NODES:
            raise ValueError(f"a pair carries {padded.shape[1]} nodes, above _MAX_NODES {_MAX_NODES} — raise it")
        return torch.nn.functional.pad(padded, (0, 0, 0, _MAX_NODES - padded.shape[1]))

    @staticmethod
    def _padded_velocity(
        source_velocities: list[Float[Tensor, "s 3"] | None] | None,
        source_positions: list[Int[Tensor, "s 3"]],
        downsample: Float[Tensor, "3"],
        device: str,
    ) -> tuple[Float[Tensor, "b max 3"] | None, Float[Tensor, "b max 3"] | None]:
        """The sources' padded prior step (grid space) and the targets' zero step — `(None, None)` when off.

        Mirrors `_heads`: the source velocity is divided by the downsample into the grid the features live on, then
        `batched_node_features` scales it by `PriorVelocity.SCALE`; the target slot carries zeros. A pair whose entry
        is `None` (no history) takes zeros of its own source length, so padding aligns with `source_voxels`.
        """
        if source_velocities is None or all(velocity is None for velocity in source_velocities):
            return None, None

        def _step(velocity: Tensor | None, source: Tensor) -> Tensor:
            present = velocity.to(torch.float32) if velocity is not None else source.new_zeros((source.shape[0], 3))
            return present / downsample

        steps = [_step(v, s) for v, s in zip(source_velocities, source_positions, strict=True)]
        source_step = JointModel._pad_fixed(steps)
        return source_step, torch.zeros_like(source_step)

    def _heads(
        self,
        features: Float[Tensor, "two c z y x"],
        source_positions: Int[Tensor, "s 3"],
        target_positions: Int[Tensor, "u 3"],
        source_velocity: Float[Tensor, "s 3"] | None,
    ) -> JointForward:
        """Both detection maps, the source→target edge logits and the node features, from one pair's features."""
        device = str(features.device)
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
        # A uniform-N scene routes to the compiled head (fused 700-node attention); ragged real pairs stay eager.
        compiled = self._scene_transformer
        is_scene = compiled is not None and source_voxel.shape[0] == self._scene_nodes
        transformer = compiled if is_scene else self.transformer
        edge_logits = transformer(feat_source, feat_target, source_voxel, target_voxel)  # (s, u)
        return JointForward(detection_t, detection_t1, edge_logits, feat_source, feat_target)
