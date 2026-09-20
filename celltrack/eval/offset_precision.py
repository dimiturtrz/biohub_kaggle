"""How precise the center-offset head actually is, on the voxels a clustering readout would actually use.

The training loss reports one number: mean L1 over every supervised voxel, out to a full cell radius. That
denominator is the wrong one for deciding whether the head can split a merged blob. A readout clusters the
voxels that FIRE — the ones the detection head already calls cell — and it has to resolve neighbours about
1.6um apart (one (1,4,4) voxel), so what matters is the error there, not averaged with the far shell where
the field is hardest and the readout never looks.

So this reports the error stratified two ways: by distance ring from the owning centre, and by the detection
head's own confidence at that voxel. A head whose whole-box mean sits above the separation it must resolve
can still be usable if the error concentrates in the outer rings; one that is flat across rings is not.
"""

import argparse
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.data.tunet_dataset import FrameDataset, FrameTarget
from celltrack.detectors.detection import CELL_SCALE_UM
from celltrack.detectors.tunet import TemporalUNetDetector
from celltrack.eval.kernel_reference import VOXEL_SCALE_UM
from celltrack.eval.proxy import TestMovieProxy
from celltrack.eval.stratification import PrecisionRow, Stratification
from celltrack.losses.offset_target import OffsetTarget
from core.obs import Obs
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "paths.yaml"
_DATASET = "biohub_cell_tracking"


@dataclass(frozen=True)
class FramePrediction:
    """One forwarded frame: both head outputs and the centres they were supposed to point at."""

    logits: Float[Tensor, "z y x"]
    offsets: Float[Tensor, "3 z y x"]
    centres: Int[Tensor, "n 3"]

    def shape(self) -> tuple[int, int, int]:
        return cast(tuple[int, int, int], tuple(self.logits.shape))

    @staticmethod
    def stream(
        detector: TemporalUNetDetector,
        targets: list[FrameTarget],
        downsample: tuple[int, int, int],
        device: str,
        frames: int,
    ) -> Iterator["FramePrediction"]:
        """Read and forward `frames` annotated frames one at a time — the IO half, kept out of the measurement."""
        dataset = FrameDataset(targets[:frames], frames, downsample, seed=0)
        for volumes, centres in dataset.batches(threads=4, prefetch=8, batch_size=1):
            heads = detector.forward_heads(volumes.to(device), single_frame=True)
            if heads.offsets is None:
                raise ValueError("this checkpoint has no offset head: nothing to measure")
            yield FramePrediction(heads.logits[0], heads.offsets[0], centres[0].to(device))


class OffsetPrecision:
    """The stratified offset error of a trained head, measured on the frames it was trained to index."""

    def __init__(self, rings: list[PrecisionRow], confidence: list[PrecisionRow], overall: PrecisionRow) -> None:
        self.rings = rings
        self.confidence = confidence
        self.overall = overall

    def report(self) -> None:
        for row in [self.overall, *self.rings, *self.confidence]:
            logger.info(
                "%-28s %6.3f um over %9d voxels%s",
                row.label,
                row.error_um,
                row.voxels,
                "  <- resolves 1.6um" if row.resolves_confusor() else "",
            )

    @classmethod
    def of(cls, predictions: Iterable[FramePrediction], downsample: tuple[int, int, int]) -> "OffsetPrecision":
        """Accumulate every frame's per-voxel offset error into both stratifications.

        Takes already-forwarded frames rather than a detector and a data root, so the measurement is separable
        from the reading — the CLI below owns the IO, and what is measured is checkable on tensors alone.
        """
        scale = cast(
            tuple[float, float, float],
            tuple(size * step for size, step in zip(VOXEL_SCALE_UM, downsample, strict=True)),
        )
        radius = cast(tuple[int, int, int], tuple(max(1, round(CELL_SCALE_UM / size)) for size in scale))
        ring_count = cls._ring_count(radius, scale)
        rings = Stratification([f"ring <={(r + 1) * min(scale):.1f}um" for r in range(ring_count)])
        confidence = Stratification(["cellness <0.1", "cellness 0.1-0.5", "cellness >=0.5"])
        total, counted = 0.0, 0
        for prediction in predictions:
            if not len(prediction.centres):
                continue
            target = OffsetTarget.of(prediction.centres, prediction.shape(), radius, scale)
            error = (prediction.offsets.float() - target.offsets).abs().sum(dim=0)
            supervised = target.supervised.bool()
            total += float(error[supervised].sum())
            counted += int(supervised.sum())
            rings.add(error, cls._ring_index(target, ring_count, min(scale)), supervised)
            confidence.add(error, cls._confidence_index(prediction.logits), supervised)
        overall = PrecisionRow("all supervised voxels", total / max(counted, 1), counted)
        return cls(rings.rows(), confidence.rows(), overall)

    @staticmethod
    def _ring_count(radius: tuple[int, int, int], scale: tuple[float, float, float]) -> int:
        """Enough rings to reach the box's far corner, so the outermost ring is a ring and not a catch-all."""
        corner = sum((r * size) ** 2 for r, size in zip(radius, scale, strict=True)) ** 0.5
        return max(1, int(corner / min(scale)) + 1)

    @staticmethod
    def _ring_index(target: OffsetTarget, rings: int, ring_um: float) -> Int[Tensor, "z y x"]:
        """Which distance ring each voxel sits in, from the micron offset it was assigned — 0 is the centre itself."""
        return (target.offsets.norm(dim=0) / ring_um).floor().long().clamp(max=rings - 1)

    @staticmethod
    def _confidence_index(logits: Float[Tensor, "z y x"]) -> Int[Tensor, "z y x"]:
        """Bucket each voxel by the detection head's own probability: what a readout would and would not look at."""
        probability = logits.sigmoid()
        return torch.bucketize(probability, torch.tensor([0.1, 0.5], device=probability.device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, default="detector_tunet_ours.pt")
    parser.add_argument("--frames", type=int, default=64, help="annotated frames to measure over")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    root = DataRoot.from_config(_CONFIG)
    checkpoint = root.processed(_DATASET) / args.weights
    Obs.setup(checkpoint.with_suffix(".precision.log"), truncate=True)
    detector, recipe = TemporalUNetDetector.from_checkpoint(checkpoint, map_location=args.device)
    detector = detector.to(args.device).eval()
    paths = TestMovieProxy.training_videos(root.videos("train"))[:1]
    targets = FrameTarget.index(root, paths, recipe.downsample)
    logger.info("measuring %s over %d of %d annotated frames", args.weights, args.frames, len(targets))
    with torch.no_grad():
        predictions = FramePrediction.stream(detector, targets, recipe.downsample, args.device, args.frames)
        OffsetPrecision.of(predictions, recipe.downsample).report()


if __name__ == "__main__":
    main()
