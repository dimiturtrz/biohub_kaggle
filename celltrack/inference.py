"""Running a trained detection U-Net over a whole video to recover cell centres.

The learned counterpart of the classical `BlobDetector`: instead of a Laplacian-of-Gaussian it evaluates
the trained network on each timepoint's temporal window, then recovers centres from the predicted response
with the *same* anisotropic non-maximum suppression the classical detector uses.

The whole read-out stays on the GPU. Timepoints are pushed through the network in batches, the sigmoid is
taken on the device, and each probability volume is handed to the peak-picker still on the device — nothing
round-trips to host memory until the final short list of coordinates. That keeps the RTX saturated instead
of stalling on per-frame transfers, and is what lets a full video be detected inside a competition kernel's
time cap.

Two readouts share that network. A fixed node budget (`detect`) keeps the strongest `keep` maxima — the
mode used to compare detectors at a matched count. A probability threshold (`detect_above`) keeps every
maximum whose predicted probability clears a high bar, letting the count *emerge*: this is how a
classification-trained detector is meant to run, trading recall for the precision that gives a linker
stable, confident centres and lands the node count in the metric's under-prediction sweet spot.
"""

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.peaks import PeakExtractor
from celltrack.training.model import DetectionUNet
from core.data.tracks import TrackGraph
from core.data.video import CellVideo

_LOW, _HIGH = 0.001, 0.999
_DEFAULT_BATCH = 8


@dataclass(frozen=True)
class LearnedDetector:
    """A trained detection network plus the shared peak-picker, run over full frames on the GPU."""

    model: DetectionUNet
    peaks: PeakExtractor
    window: int
    device: str = "cpu"
    batch: int = _DEFAULT_BATCH

    def detect(self, video: CellVideo, keep: int) -> TrackGraph:
        """Detect the `keep` strongest centres across a video, evenly split over its timepoints."""
        per_frame = max(1, round(keep / video.timepoint_count))
        return self._graph([self.peaks.centres(response, per_frame) for response in self.responses(video)])

    def detect_above(self, video: CellVideo, probability_threshold: float) -> TrackGraph:
        """Detect every centre whose predicted probability exceeds the threshold — the node count emerges."""
        return self._graph(
            [self.peaks.above_threshold(response, probability_threshold) for response in self.responses(video)]
        )

    def responses(self, video: CellVideo) -> Iterator[Float[Tensor, "z y x"]]:
        """Each timepoint's probability volume, on the device, from batched forward passes over the video.

        The forward runs under fp16 autocast in channels-last-3d memory format, so full-frame inference takes
        the tensor-core path instead of dense fp32 convolutions. fp16 (not the training loop's bf16) keeps the
        response fine-grained: the equality-based peak NMS collapses a plateau of *bit-identical* maxima to one
        centre, and bf16's 8-bit mantissa would round distinct neighbouring peaks into false plateaus and merge
        them away. Inference has no gradients to overflow, so bf16's wider range buys nothing here.
        """
        cuda = self.device.startswith("cuda")
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if cuda else nullcontext()
        stacks = self._stacks(video)
        for start in range(0, len(stacks), self.batch):
            batch = torch.from_numpy(np.stack(stacks[start : start + self.batch])).to(self.device)
            batch = batch.contiguous(memory_format=torch.channels_last_3d)
            with torch.no_grad(), autocast:
                probabilities = torch.sigmoid(self.model(batch)).float()
            for index in range(probabilities.shape[0]):
                yield probabilities[index]

    def _stacks(self, video: CellVideo) -> list[Float[np.ndarray, "t z y x"]]:
        """Every timepoint's `window`-frame temporal stack, clamped to the video's ends."""
        frames = list(video.normalised_frames(_LOW, _HIGH))
        return [self._window(frames, timepoint) for timepoint in range(len(frames))]

    def _graph(self, per_frame_centres: list[Int[np.ndarray, "k 3"]]) -> TrackGraph:
        """Stack per-timepoint `(z, y, x)` centres into a `(t, z, y, x)` edgeless track graph."""
        stamped = [
            np.column_stack([np.full(len(centres), timepoint), centres])
            for timepoint, centres in enumerate(per_frame_centres)
        ]
        stacked = np.concatenate(stamped).astype(np.int64) if stamped else np.empty((0, 4), np.int64)
        return TrackGraph(
            node_ids=np.arange(len(stacked), dtype=np.int64),
            coordinates=stacked,
            edges=np.empty((0, 2), dtype=np.int64),
        )

    def _window(self, frames: list[Float[np.ndarray, "z y x"]], timepoint: int) -> Float[np.ndarray, "t z y x"]:
        """A `window`-frame stack centred on a timepoint, clamped to the video's ends."""
        half = self.window // 2
        start = min(max(timepoint - half, 0), max(len(frames) - self.window, 0))
        return np.stack(frames[start : start + self.window])
