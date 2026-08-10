"""The temporal-U-Net detector: pilkwang's architecture, trained by us, wrapped as one detector strategy.

The detection half of the pilkwang model is a `TemporalUNet3D` backbone with a 1x1x1 detection head. It
reads a frame at a fixed spatial downsample (Z,Y,X = 1,4,4 by default — 16x fewer voxels, the input recipe
that makes the whole frame trainable and inference fast), quantile-normalises the intensities per video, and
emits a per-voxel cellness logit. Local maxima above a threshold are the detected cell centres.

This one class is the shared contract: the trainer optimises `.model`, the fold evaluator and the Kaggle
kernel both call `.detections(...)`. The architecture lives in the pinned `external/` checkout (their code,
not vendored); everything around it — the inference recipe, the peak read-out, the weight (de)serialisation
— is ours.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import numpy as np
import torch
import zarr
from jaxtyping import Float, Int
from torch import Tensor, nn
from zarr.core.metadata import ArrayV3Metadata

from celltrack.detectors.peaks import PeakExtractor
from core.data.tracks import TrackGraph
from core.data.video import ImageStatistics, Multiscale
from core.geometry import Spacing

_EXT_SRC = Path(__file__).parents[2] / "external" / "kaggle-cell-tracking-competition" / "src"


@dataclass(frozen=True)
class DetectorRecipe:
    """The fixed input pipeline around the network — downsample, peak-suppression radius, TTA, node cap."""

    downsample: tuple[int, int, int] = (1, 4, 4)
    pool_kernel_um: float = 5.0
    tta: bool = True
    # Hard cap on detected centres per frame. A well-trained detector emits far fewer, but an undertrained one
    # can fire on noise everywhere; without a cap that flood explodes the linker's per-frame O(N^3) assignment.
    keep_per_frame: int = 2000

    def fingerprint(self) -> str:
        """A short cache key of the inputs that change the forward response — the downsample and the TTA.

        The read-out params (`pool_kernel_um`, `keep_per_frame`) shape peaks, not the cached logit volumes, so
        they are excluded: a threshold or NMS-radius change must replay the same cache, not re-forward.
        """
        return f"ds{'x'.join(map(str, self.downsample))}_tta{int(self.tta)}"

    def as_config(self) -> dict[str, object]:
        """The recipe as a JSON-serialisable dict, saved beside the weights so inference is reproducible."""
        return {
            "downsample": list(self.downsample),
            "pool_kernel_um": self.pool_kernel_um,
            "tta": self.tta,
            "keep_per_frame": self.keep_per_frame,
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

    def __init__(self, out_channels: int = 32, layers: tuple[int, ...] = (32, 64, 128)) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.layers = list(layers)
        self.unet = self._backbone_cls()(in_channels=1, out_channels=out_channels, layers=list(layers))
        self.detect_head = nn.Conv3d(out_channels, 1, kernel_size=1)

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

    def forward_batch(self, frames: Float[Tensor, "b z y x"]) -> Float[Tensor, "b z y x"]:
        """Detection logits for a batch of frames of one shape — the trainer's path, GPU fed in one pass.

        The temporal backbone needs a pair per item, so each frame is duplicated into a fake two-frame window
        and the first output frame is read — the same path the model was trained through, run over the batch.
        """
        pair = torch.stack([frames, frames], dim=1).unsqueeze(2)  # (B, 2, 1, Z, Y', X')
        features = self.unet(pair)  # (B, 2, C, Z, Y', X')
        return self.detect_head(features[:, 0])[:, 0]  # (B, Z, Y', X')

    @classmethod
    def from_pack(cls, pack: Path, map_location: str = "cpu") -> tuple["TemporalUNetDetector", DetectorRecipe]:
        """Load pilkwang's published split — the `unet.*` + `detect_head.*` half of `edge_predictor_best.pth`."""
        config = json.loads((pack / "config.json").read_text())
        detector = cls(config["unet_out_channels"], tuple(config["unet_layers"]))
        state = torch.load(pack / "edge_predictor_best.pth", map_location=map_location, weights_only=True)
        detector.load_state_dict({k: v for k, v in state.items() if k.startswith(("unet.", "detect_head."))})
        return detector, DetectorRecipe.from_config(config)

    @classmethod
    def from_checkpoint(cls, path: Path, map_location: str = "cpu") -> tuple["TemporalUNetDetector", DetectorRecipe]:
        """Rebuild a detector we trained: our checkpoint carries both the weights and the recipe."""
        blob = torch.load(path, map_location=map_location, weights_only=True)
        detector = cls(int(blob["out_channels"]), tuple(blob["layers"]))
        detector.load_state_dict(blob["state_dict"])
        return detector, DetectorRecipe.from_config(blob["recipe"])

    def save_checkpoint(self, path: Path, recipe: DetectorRecipe) -> None:
        """Persist the trained weights with the structure and recipe needed to reconstruct the detector."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "out_channels": self.out_channels,
                "layers": self.layers,
                "recipe": recipe.as_config(),
            },
            path,
        )

    def detections(self, path: Path, threshold: float, recipe: DetectorRecipe, device: str) -> TrackGraph:
        """Detect cell centres in every frame of a video, returned as an edge-free `TrackGraph`.

        The forward pass and the peak read-out are split so the response can be cached (see
        `logit_volumes`): here they run back to back, the live path. Coordinates come back in the
        original full-resolution voxel grid (peaks scaled by the downsample), so the graph drops straight
        into a linker built for the video's native spacing.
        """
        volumes = self.logit_volumes(path, recipe, device)
        return self.graph_from_volumes(volumes, self.voxel_scale(path), threshold, recipe, device)

    def logit_volumes(self, path: Path, recipe: DetectorRecipe, device: str) -> list[Float[np.ndarray, "z y x"]]:
        """Each frame's per-voxel cellness **logit** on the downsampled grid — the cacheable response to blend.

        Logits, not probabilities, are the multi-seed blend space: they are unsaturated, so averaging seeds
        keeps the peak distinctions the equality-NMS reads. `probability_volumes` is the sigmoid of these.
        """
        return self._response(path, recipe, device, as_probability=False)

    def probability_volumes(self, path: Path, recipe: DetectorRecipe, device: str) -> list[Float[np.ndarray, "z y x"]]:
        """Each frame's per-voxel cellness probability — the single-seed cacheable response (sigmoid on device)."""
        return self._response(path, recipe, device, as_probability=True)

    @torch.no_grad()
    def _response(
        self, path: Path, recipe: DetectorRecipe, device: str, *, as_probability: bool
    ) -> list[Float[np.ndarray, "z y x"]]:
        """Forward the whole video into per-frame volumes, taking the sigmoid on the device before the host copy.

        The forward is the expensive, weights-and-video-determined half; a `ResponseCache` runs it once and
        every later read-out replays cheaply. Volumes are the downsampled shape (16x fewer voxels) — small.
        """
        self.eval()
        source = self._open_source(path)
        volumes: list[Float[np.ndarray, "z y x"]] = []
        for timepoint in range(int(source.array.shape[0])):
            logits = self._logits(self._read_frame(source, timepoint, recipe.downsample, device), recipe.tta)
            volumes.append((torch.sigmoid(logits) if as_probability else logits).cpu().numpy())
        return volumes

    @classmethod
    def graph_from_volumes(
        cls,
        volumes: list[Float[np.ndarray, "z y x"]],
        scale: tuple[float, float, float],
        threshold: float,
        recipe: DetectorRecipe,
        device: str,
    ) -> TrackGraph:
        """Read cell centres out of cached **logit** volumes — the cheap, replayable half of detection.

        Suppression runs on the logits, not their sigmoid: `sigmoid` saturates every confident logit (>~16)
        to an identical fp32 `1.0`, so the valley separating two touching nuclei collapses into the same flat
        plateau as their peaks and the connected-component labelling merges both cells into one centre. The
        logits keep that valley, so the reference read-out (`max_pool3d(logits)` equality, threshold on the
        sigmoid) recovers one centre per cell. The prob-space `threshold` becomes the equivalent logit floor.
        """
        extractor = cls._extractor(scale, recipe, device)
        floor = cls._logit_floor(threshold)
        stamped: list[np.ndarray] = []
        for timepoint, volume in enumerate(volumes):
            peaks = cls._capped_peaks(extractor, volume, floor, recipe.keep_per_frame)
            scaled = peaks * np.array(recipe.downsample)
            stamped.append(np.column_stack([np.full(len(scaled), timepoint), scaled]).astype(np.int64))
        nodes = np.concatenate(stamped) if stamped else np.empty((0, 4), dtype=np.int64)
        return TrackGraph(np.arange(len(nodes), dtype=np.int64), nodes, np.empty((0, 2), np.int64))

    def voxel_scale(self, path: Path) -> tuple[float, float, float]:
        """The video's `(z, y, x)` micrometre voxel scale — read cheaply from the OME metadata for the read-out."""
        return self._open_source(path).scale

    def _logits(self, frame: Float[Tensor, "z y x"], tta: bool) -> Float[Tensor, "z y x"]:  # noqa: FBT001
        """Detection logits for a frame, averaged over the flip-TTA ensemble when enabled."""
        logits = self(frame)
        if tta:
            for dims in [(-1,), (-2,), (-2, -1)]:
                logits = logits + self(frame.flip(dims)).flip(dims)
            logits = logits / 4
        return logits

    @staticmethod
    def _extractor(scale: tuple[float, float, float], recipe: DetectorRecipe, device: str) -> PeakExtractor:
        """A peak extractor on the downsampled voxel grid — its NMS collapses a saturated plateau to one centre."""
        spacing = Spacing(
            z=scale[0] * recipe.downsample[0],
            y=scale[1] * recipe.downsample[1],
            x=scale[2] * recipe.downsample[2],
        )
        return PeakExtractor(spacing, scale_um=recipe.pool_kernel_um / 2.0, device=device)

    @staticmethod
    def _logit_floor(threshold: float) -> float:
        """The logit-space floor for a probability `threshold` — total over `[0, 1]` (0 keeps all, 1 keeps none)."""
        if threshold <= 0.0:
            return -math.inf
        if threshold >= 1.0:
            return math.inf
        return math.log(threshold / (1.0 - threshold))

    @staticmethod
    def _capped_peaks(
        extractor: PeakExtractor, logits: Float[np.ndarray, "z y x"], floor: float, keep: int
    ) -> Int[np.ndarray, "k 3"]:
        """Suppressed logit maxima above the logit `floor`, capped to the `keep` strongest — bounds the cost matrix."""
        coordinates, values = extractor.maxima(logits, floor=floor)
        if len(coordinates) > keep:
            coordinates = coordinates[np.argsort(values)[::-1][:keep]]
        return coordinates

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
