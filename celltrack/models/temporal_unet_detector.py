"""The temporal-U-Net detector network — pilkwang's architecture, trained by us, as a pure `nn.Module`.

The detection half of the pilkwang model is a `TemporalUNet3D` backbone with a 1x1x1 detection head. It
reads a frame at a fixed spatial downsample (Z,Y,X = 1,4,4 by default — 16x fewer voxels, the input recipe
that makes the whole frame trainable and inference fast), quantile-normalises the intensities per video, and
emits a per-voxel cellness logit. The architecture lives in the pinned `external/` checkout (their code, not
vendored); everything around it — the inference recipe, the weight (de)serialisation, the frame read — is ours.

This is the learnable definition alone: the backbone, the head, the forward, the weight round-trip, and the
per-frame video read the trainer and both readouts share. The peak read-out that turns logit volumes into a
graph lives with the detector strategy in `celltrack.detectors`, which imports this net (never the reverse).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Self, cast, override

import numpy as np
import torch
import zarr
from jaxtyping import Float
from torch import Tensor, nn
from zarr.core.metadata import ArrayV3Metadata

from core.data.video import ImageStatistics, Multiscale

_EXT_SRC = Path(__file__).parents[2] / "external" / "kaggle-cell-tracking-competition" / "src"
_SPATIAL_AXES = 3
# Target group count for the GroupNorm swap — the same value the center-prior detector settled on
# (`center_prior._POOL_MULTIPLE`). A channel width takes `gcd(width, 8)` groups so the count always divides
# it (8 for the 32/64/128 stage widths, a safe divisor for any other), never the raise `GroupNorm(8, 20)` is.
_NORM_GROUPS = 8
NORMS = ("batch", "group")
# Width of the dedicated center-point contrastive embedding head (CELLECT-style). A per-cell vector this wide
# gives the InfoNCE term its own representation to reshape, separate from the 1-channel detection logit — so the
# backbone learns cell-discriminating features under a strong gradient without trading the detection head away.
_EMBED_CHANNELS = 64


@dataclass(frozen=True)
class FlipView:
    """One spatial mirroring of a frame — the unit of flip-TTA, defined ONCE for every stage that ensembles.

    `dims` are negative tensor dims into a trailing `(z, y, x)`, so one view applies unchanged to a bare frame,
    a stacked pair, or a batched window. The empty tuple is the identity and returns its input untouched rather
    than calling `flip(())`, which keeps an ensemble whose first member is the identity numerically identical
    to the plain forward it replaces.
    """

    dims: tuple[int, ...] = ()

    @classmethod
    def tta_ensemble(cls) -> tuple["FlipView", ...]:
        """Identity plus the three in-plane mirrorings — the four views, identity first.

        In-plane `(y, x)` only, and no rotations: the network is trained with in-plane flips alone, so a
        mirrored frame is a scene it has seen and a rotated or z-mirrored one is a different input
        distribution wearing the same label.
        """
        return (cls(), cls((-1,)), cls((-2,)), cls((-2, -1)))

    def apply(self, volume: Float[Tensor, "*batch z y x"]) -> Float[Tensor, "*batch z y x"]:
        """The volume mirrored along this view's dims — the identity view returns the very same tensor."""
        return volume.flip(self.dims) if self.dims else volume

    def mirror(self, positions: Float[Tensor, "n 3"], extent: Float[Tensor, "3"]) -> Float[Tensor, "n 3"]:
        """Voxel positions mirrored to track `apply`, in a grid whose last valid coordinate per axis is `extent`."""
        if not self.dims:
            return positions
        mirrored = positions.clone()
        for dim in self.dims:
            axis = dim % _SPATIAL_AXES
            mirrored[:, axis] = extent[axis] - mirrored[:, axis]
        return mirrored


IDENTITY_VIEW = FlipView()


@dataclass(frozen=True)
class DetectorRecipe:
    """The fixed input pipeline around the network — downsample, peak-suppression radius, TTA, node cap."""

    downsample: tuple[int, int, int] = (1, 4, 4)
    pool_kernel_um: float = 5.0
    tta: bool = True
    # Hard cap on detected centres per frame. A well-trained detector emits far fewer, but an undertrained one
    # can fire on noise everywhere; without a cap that flood explodes the linker's per-frame O(N^3) assignment.
    keep_per_frame: int = 2000
    # Detect each frame inside a REAL consecutive pair instead of a duplicate of itself. The network is
    # TRAINED on real pairs (the detect head is supervised on both frames of a (t, t+1) window), but the
    # shipped read-out feeds it (t, t) — a degenerate window it never saw, whose temporal attention has
    # nothing to attend to. Every frame is still covered: slide the pair and take the last frame from the
    # trailing slot of the final window.
    pair_context: bool = False

    def fingerprint(self) -> str:
        """A short cache key of the inputs that change the forward response — the downsample and the TTA.

        The read-out params (`pool_kernel_um`, `keep_per_frame`) shape peaks, not the cached logit volumes, so
        they are excluded: a threshold or NMS-radius change must replay the same cache, not re-forward.
        `pair_context` IS included — it changes the window the network sees, so it changes the response, and
        omitting it would let one arm of an A/B silently replay the other's cached volumes.
        """
        return f"ds{'x'.join(map(str, self.downsample))}_tta{int(self.tta)}_pc{int(self.pair_context)}"

    def as_config(self) -> dict[str, object]:
        """The recipe as a JSON-serialisable dict, saved beside the weights so inference is reproducible."""
        return {
            "downsample": list(self.downsample),
            "pool_kernel_um": self.pool_kernel_um,
            "tta": self.tta,
            "keep_per_frame": self.keep_per_frame,
            "pair_context": self.pair_context,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "DetectorRecipe":
        """Rebuild a recipe from a saved config dict, tolerating pilkwang's superset of keys."""
        return cls(
            downsample=tuple(config["downsample"]),  # type: ignore[arg-type]
            pool_kernel_um=float(config.get("pool_kernel_um", 5.0)),  # type: ignore[arg-type]
            tta=bool(config.get("tta", True)),
            keep_per_frame=int(config.get("keep_per_frame", 2000)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class _VideoSource:
    """A raw video zarr with the per-video normalisation statistics the detector expects."""

    array: zarr.Array[ArrayV3Metadata]
    q_low: float
    q_high: float
    scale: tuple[float, float, float]


class TemporalUNetDetector(nn.Module):
    """A `TemporalUNet3D` backbone plus a detection head — the pilkwang detector, ours to train and run."""

    def __init__(self, out_channels: int = 32, layers: tuple[int, ...] = (32, 64, 128), norm: str = "batch") -> None:
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        self.out_channels = out_channels
        self.layers = list(layers)
        # "batch" keeps the published BatchNorm3d backbone byte-identical (a warm-started pilkwang pack carries
        # BN running stats, so its weights only load into a BN backbone). "group" swaps every BatchNorm3d for a
        # GroupNorm, whose per-sample stats store no source-domain running mean/var and so do not leak the train
        # microscope's intensity distribution across the domain gap at eval — a from-scratch-only swap (random
        # init licenses it; there is no BN checkpoint to honour). Persisted so a trained checkpoint rebuilds the
        # same norm before its weights load.
        self.norm = norm
        self.unet = self._backbone_cls()(in_channels=1, out_channels=out_channels, layers=list(layers))
        if norm == "group":
            self._groupnorm_backbone()
        self.detect_head = nn.Conv3d(out_channels, 1, kernel_size=1)
        # A dedicated per-voxel embedding head, PARALLEL to the detection head and tapping the SAME backbone
        # features. Fresh-init and unused at inference (the tracker reads only detection + edge logits); it exists
        # to carry the center-point contrastive gradient back into the backbone during training. A warm-start
        # checkpoint predates this head, so every loader below tolerates its absent key (strict=False).
        self.embed_head = nn.Conv3d(out_channels, _EMBED_CHANNELS, kernel_size=1)
        # Whether the backbone's per-voxel time attention carries a frame-position embedding (motion sight).
        # False = the published order-blind backbone. `install_temporal_position` flips it and rewrites the
        # attention blocks; the flag is persisted so a trained checkpoint rebuilds the same blocks before loading.
        self.temporal_position = False

    def install_temporal_position(self, max_positions: int = 8) -> None:
        """Give the backbone motion sight: wrap each temporal-attention block with a frame-position embedding.

        Called AFTER the backbone's weights are in place (a warm start's pack load, or the moment before a
        checkpoint load), so the reused `norm`/`attn` weights are the trained ones and only the zero-initialised
        position table is fresh. Identity blocks (the skipped full-resolution stage) carry no attention and are
        left alone. Idempotent guard: a second call would wrap the wrappers, so it refuses once installed.
        """
        from celltrack.models.temporal_position_attention import TemporalPositionAttention  # noqa: PLC0415

        if self.temporal_position:
            return
        blocks = cast(nn.ModuleList, self.unet.temporal_blocks)  # external backbone is untyped
        for index, block in enumerate(blocks):
            if hasattr(block, "attn"):  # a real _TemporalAttention (Identity, the skipped stage, has none)
                blocks[index] = TemporalPositionAttention(block, max_positions=max_positions)
        self.temporal_position = True

    def _groupnorm_backbone(self) -> None:
        """Replace every BatchNorm3d in the backbone with a same-width GroupNorm — the domain-robust swap.

        Module surgery on OUR wrapper, never the pinned external backbone. The published conv block normalises
        with BatchNorm3d, whose stored running mean/var are the TRAIN movies' intensity statistics, applied
        unchanged to val frames — the leak across the 44b6/6bba microscope gap. GroupNorm recomputes per sample
        at train and test alike, so no source-domain statistic is carried. Each width takes `gcd(width, groups)`
        groups so the count always divides it. Affine weight/bias survive (GroupNorm has them too); only the BN
        running buffers vanish, which is why the swap is from-scratch only — there is no BN checkpoint to honour.
        """
        for module in list(self.unet.modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm3d):
                    width = child.num_features
                    setattr(module, child_name, nn.GroupNorm(gcd(width, _NORM_GROUPS), width))

    @staticmethod
    def _backbone_cls() -> type[nn.Module]:
        """The `TemporalUNet3D` class, from whichever package the environment exposes it under.

        Local runs import it from the pinned `external/` checkout (`tracking_cellmot`); the Kaggle kernel
        gets the same architecture from the mounted support pack (`biohub_tracking`).
        """
        try:
            from tracking_cellmot.models import TemporalUNet3D  # type: ignore[missing-import]  # noqa: PLC0415
        except ImportError:
            if str(_EXT_SRC) not in sys.path:
                sys.path.insert(0, str(_EXT_SRC))
            try:
                from tracking_cellmot.models import TemporalUNet3D  # type: ignore[missing-import]  # noqa: PLC0415
            except ImportError:
                from biohub_tracking.models import TemporalUNet3D  # type: ignore[missing-import]  # noqa: PLC0415
        return TemporalUNet3D

    @override
    def forward(self, frame: Float[Tensor, "z y x"]) -> Float[Tensor, "z y x"]:
        """Detection logits for one already-normalised, already-downsampled frame (the batch-of-one path)."""
        return self.forward_batch(frame.unsqueeze(0))[0]  # (Z, Y', X')

    def forward_pair(self, frames: Float[Tensor, "two z y x"]) -> Float[Tensor, "two z y x"]:
        """Detection logits for BOTH frames of a real consecutive window — the pair the model was trained on.

        `forward_batch` duplicates a frame to satisfy the temporal window and then discards the second output.
        Here the window is genuine, so the attention has a real neighbour to attend to, and both outputs are
        kept: sliding the pair covers every frame, the last one from the trailing slot.
        """
        features = self.unet(frames.unsqueeze(0).unsqueeze(2))  # (1, 2, C, Z, Y', X')
        return self.detect_head(features[0])[:, 0]  # (2, Z, Y', X')

    def forward_batch(
        self, frames: Float[Tensor, "b z y x"], *, single_frame: bool = False
    ) -> Float[Tensor, "b z y x"]:
        """Detection logits for a batch of frames of one shape — the trainer's path, GPU fed in one pass.

        The temporal backbone needs a window per item. The shipped path duplicates each frame into a fake
        two-frame window (`single_frame=False`) — the exact path the model was trained/grafted through. For
        detection-only work `single_frame=True` sends a genuine one-frame window instead, halving the spatial
        convolutions the duplicate frame otherwise wastes; validated near-lossless on the dense movie (identical
        node count, 99.98% peak match) since only the first output frame is read either way.
        """
        window = frames.unsqueeze(1) if single_frame else torch.stack([frames, frames], dim=1)  # (B, T, Z, Y', X')
        features = self.unet(window.unsqueeze(2))  # (B, T, C, Z, Y', X')
        return self.detect_head(features[:, 0])[:, 0]  # (B, Z, Y', X')

    @classmethod
    def from_pack(cls, pack: Path, map_location: str = "cpu") -> tuple[Self, DetectorRecipe]:
        """Load pilkwang's published split — the `unet.*` + `detect_head.*` half of `edge_predictor_best.pth`."""
        config = json.loads((pack / "config.json").read_text())
        detector = cls(config["unet_out_channels"], tuple(config["unet_layers"]))
        state = torch.load(pack / "edge_predictor_best.pth", map_location=map_location, weights_only=True)
        # strict=False: the fresh embed_head has no counterpart in the pilkwang pack (unet + detect_head only).
        detector.load_state_dict(
            {k: v for k, v in state.items() if k.startswith(("unet.", "detect_head."))}, strict=False
        )
        return detector, DetectorRecipe.from_config(config)

    @classmethod
    def from_checkpoint(cls, path: Path, map_location: str = "cpu") -> tuple[Self, DetectorRecipe]:
        """Rebuild a detector we trained: our checkpoint carries both the weights and the recipe."""
        blob = torch.load(path, map_location=map_location, weights_only=True)
        detector = cls(int(blob["out_channels"]), tuple(blob["layers"]), norm=blob.get("norm", "batch"))
        # strict=False: a checkpoint written before the embed head existed carries no embed_head.* key.
        detector.load_state_dict(cls._uncompiled(blob["state_dict"]), strict=False)
        return detector, DetectorRecipe.from_config(blob["recipe"])

    @staticmethod
    def _uncompiled(state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Strip the `_orig_mod.` prefix torch.compile inserts, so a compiled run's checkpoint loads plainly."""
        return {key.replace("_orig_mod.", ""): value for key, value in state.items()}

    def save_checkpoint(self, path: Path, recipe: DetectorRecipe) -> None:
        """Persist the trained weights with the structure and recipe needed to reconstruct the detector."""
        torch.save(
            {
                "state_dict": self._uncompiled(self.state_dict()),
                "out_channels": self.out_channels,
                "layers": self.layers,
                "norm": self.norm,
                "recipe": recipe.as_config(),
            },
            path,
        )

    @staticmethod
    def _open_source(path: Path) -> _VideoSource:
        """Open a video zarr and read its intensity quantiles and voxel scale from the OME metadata."""
        attributes = zarr.open_group(path, mode="r").attrs
        transform = cast(list[Multiscale], attributes["multiscales"])[0]["datasets"][0]["coordinateTransformations"]
        scale = transform[0]["scale"]
        quantiles = cast(ImageStatistics, attributes["image_statistics"])["quantiles"]
        return _VideoSource(
            array=zarr.open_array(path / "0"),
            q_low=float(quantiles["0.001"]),
            q_high=float(quantiles["0.999"]),
            scale=(float(scale[1]), float(scale[2]), float(scale[3])),
        )

    @staticmethod
    def _read_frame(
        source: _VideoSource, timepoint: int, downsample: tuple[int, int, int], device: str
    ) -> Float[Tensor, "z y x"]:
        """One quantile-normalised frame, strided-read at the downsample and moved to the device."""
        dz, dy, dx = downsample
        raw = np.asarray(source.array[timepoint, ::dz, ::dy, ::dx], dtype=np.float32)
        normed = (torch.from_numpy(raw) - source.q_low) / (source.q_high - source.q_low + 1e-6)
        return normed.clamp(0.0).to(device)
