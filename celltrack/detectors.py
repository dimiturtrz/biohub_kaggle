"""Building a detector from a name and its hyperparameters — the detector family's factory, mirroring linkers.py.

Every detector turns a video into detected cell centres, but the family is heterogeneous: the learned nets
(pilkwang's mounted pack, our own checkpoint) read a probability threshold, the classical LoG/DoG read a
node budget, and their construction pulls different resources (a pack directory, a checkpoint file, nothing).
`DetectorConfig` is the one place that knows the whole family — a validated name plus the union of
hyperparameters — and `build` resolves it against a `DetectorContext` (device, the resolved weight paths, the
response-cache root) into a `Detector`. The `Detector` contract is uniform even where construction is not:
`nodes(video_key, path)` returns the edge-free graph, the readout the config chose already baked in.

This does not replace `BlendDetectorScorer` inside `CellTracker` (the shipped tracker is coupled to the
pilkwang edge transformer, which only reads the pilkwang detector's features); it is the systematic registry a
kernel or an eval constructs a detector through, so a submission's `S1` token maps to one `DetectorConfig`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from celltrack.detection import CELL_SCALE_UM, HIGH, LOW
from celltrack.dog_detection import DoGDetector
from celltrack.inference import LearnedDetector
from celltrack.peaks import PeakExtractor
from celltrack.pipeline import BlendDetectorScorer
from celltrack.response_cache import ResponseCache
from celltrack.training.model import DetectionUNet
from celltrack.tunet import TemporalUNetDetector
from core.data.tracks import TrackGraph
from core.data.video import CellVideo

DETECTOR_NAMES = ("pilktunet", "ourstunet", "unetaug", "dog")


class Detector(Protocol):
    """A mounted detector: a video in, its detected cell centres out as an edge-free graph — readout baked in."""

    def nodes(self, video_key: str, path: Path) -> TrackGraph:
        """The detected centres for a video, replayed from the response cache after a first forward."""
        ...


@dataclass(frozen=True)
class DetectorContext:
    """The runtime a build needs — the device, the resolved weight paths, and the response-cache root."""

    device: str
    weights: tuple[Path, ...] = ()
    responses: Path = Path("cache")


@dataclass(frozen=True)
class _ScorerDetector:
    """A TemporalUNet scorer (single or blended) read out at a fixed probability threshold."""

    scorer: BlendDetectorScorer
    threshold: float

    def nodes(self, video_key: str, path: Path) -> TrackGraph:
        """The scorer's centres at the configured threshold."""
        return self.scorer.nodes(video_key, path, self.threshold)


@dataclass(frozen=True)
class _LearnedDetector:
    """Our own DetectionUNet, its peak-picker rebuilt per video from that video's spacing, read above a threshold."""

    model: DetectionUNet
    threshold: float
    scale_um: float
    window: int
    device: str

    def nodes(self, video_key: str, path: Path) -> TrackGraph:
        """Detect every centre above the threshold, the peak-picker at this video's physical spacing."""
        video = CellVideo.from_ome_zarr(path)
        detector = LearnedDetector(
            model=self.model,
            peaks=PeakExtractor(video.spacing, self.scale_um, self.device),
            window=self.window,
            device=self.device,
        )
        return detector.detect_above(video, self.threshold)


@dataclass(frozen=True)
class _DogDetector:
    """The classical multi-scale DoG detector, rebuilt per video from its spacing, read at a node budget."""

    scales_um: tuple[float, ...]
    keep_per_video: int
    device: str

    def nodes(self, video_key: str, path: Path) -> TrackGraph:
        """Detect the budget's strongest centres, the DoG band at this video's physical spacing."""
        video = CellVideo.from_ome_zarr(path)
        detector = (
            DoGDetector(spacing=video.spacing, scales_um=self.scales_um, device=self.device)
            if self.scales_um
            else DoGDetector(spacing=video.spacing, device=self.device)
        )
        return detector.detect(video.normalised_frames(LOW, HIGH), video.timepoint_count, self.keep_per_video)


class DetectorConfig(BaseModel):
    """A detector choice and the hyperparameters the family draws from, resolved to a concrete detector by `build`.
    Immutable + validated: `name` must be one the family knows, the threshold is a probability, budgets positive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "pilktunet"
    threshold: float = Field(0.99, gt=0, lt=1)
    keep_per_video: int = Field(24000, gt=0)
    scale_um: float = Field(CELL_SCALE_UM, gt=0)
    scales_um: tuple[float, ...] = ()

    @field_validator("name")
    @classmethod
    def _known_name(cls, name: str) -> str:
        if name not in DETECTOR_NAMES:
            raise ValueError(f"unknown detector {name!r}; choose from {DETECTOR_NAMES}")
        return name

    def build(self, context: DetectorContext) -> Detector:
        """The concrete detector named by this config, mounted against the runtime context."""
        return _BUILDERS[self.name](self, context)


def _blend(
    loader: Callable[[Path, str], tuple[TemporalUNetDetector, object]], config: DetectorConfig, context: DetectorContext
) -> _ScorerDetector:
    """A single- or multi-seed TemporalUNet scorer over `context.weights`, each seed loaded by `loader`."""
    seeds = []
    recipe: object = None
    for index, weight in enumerate(context.weights):
        detector, recipe = loader(weight, context.device)
        cache = ResponseCache(context.responses, f"{config.name}_{index}")
        seeds.append((detector.to(context.device).eval(), cache))
    return _ScorerDetector(
        BlendDetectorScorer(detectors=tuple(seeds), recipe=recipe, device=context.device), config.threshold
    )


def _unetaug(config: DetectorConfig, context: DetectorContext) -> _LearnedDetector:
    checkpoint = torch.load(context.weights[0], map_location=context.device, weights_only=True)
    model = DetectionUNet.from_checkpoint(checkpoint).to(context.device).eval()
    return _LearnedDetector(model, config.threshold, config.scale_um, int(checkpoint["window"]), context.device)


_BUILDERS: dict[str, Callable[[DetectorConfig, DetectorContext], Detector]] = {
    "pilktunet": lambda config, context: _blend(
        lambda weight, device: TemporalUNetDetector.from_pack(weight, map_location=device), config, context
    ),
    "ourstunet": lambda config, context: _blend(
        lambda weight, device: TemporalUNetDetector.from_checkpoint(weight, map_location=device), config, context
    ),
    "unetaug": _unetaug,
    "dog": lambda config, context: _DogDetector(config.scales_um, config.keep_per_video, context.device),
}
