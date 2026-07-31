"""Running a trained detection U-Net over a whole video to recover cell centres.

The learned counterpart of the classical `BlobDetector`: instead of a Laplacian-of-Gaussian it evaluates
the trained network on each timepoint's temporal window, then recovers centres from the predicted response
with the *same* anisotropic non-maximum suppression the classical detector uses.

Two readouts share that network. A fixed node budget (`detect`) keeps the strongest `keep` maxima — the
mode used to compare detectors at a matched count. A probability threshold (`detect_above`) keeps every
maximum whose predicted probability clears a high bar, letting the count *emerge*: this is how a
classification-trained detector is meant to run, trading recall for the precision that gives a linker
stable, confident centres and lands the node count in the metric's under-prediction sweet spot.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int

from celltrack.peaks import PeakExtractor
from celltrack.training.model import DetectionUNet
from core.data.tracks import TrackGraph
from core.data.video import CellVideo

_LOW, _HIGH = 0.001, 0.999


@dataclass(frozen=True)
class LearnedDetector:
    """A trained detection network plus the shared peak-picker, run over full frames."""

    model: DetectionUNet
    peaks: PeakExtractor
    window: int
    device: str = "cpu"

    def detect(self, video: CellVideo, keep: int) -> TrackGraph:
        """Detect the `keep` strongest centres across a video, evenly split over its timepoints."""
        per_frame = max(1, round(keep / video.timepoint_count))
        frames = list(video.normalised_frames(_LOW, _HIGH))
        return self._graph(
            [self.peaks.centres(self._heatmap(frames, timepoint), per_frame) for timepoint in range(len(frames))]
        )

    def detect_above(self, video: CellVideo, probability_threshold: float) -> TrackGraph:
        """Detect every centre whose predicted probability exceeds the threshold — the node count emerges."""
        frames = list(video.normalised_frames(_LOW, _HIGH))
        return self._graph(
            [
                self.peaks.above_threshold(self._probability(frames, timepoint), probability_threshold)
                for timepoint in range(len(frames))
            ]
        )

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

    def _probability(self, frames: list[Float[np.ndarray, "z y x"]], timepoint: int) -> Float[np.ndarray, "z y x"]:
        """The network's per-voxel cell probability — its logits passed through a sigmoid."""
        return torch.sigmoid(torch.from_numpy(self._heatmap(frames, timepoint))).numpy()

    def _heatmap(self, frames: list[Float[np.ndarray, "z y x"]], timepoint: int) -> Float[np.ndarray, "z y x"]:
        """The network's cellness logits for one timepoint, from the temporal window centred on it."""
        window = self._window(frames, timepoint)
        batch = torch.from_numpy(window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(batch)
        return prediction.squeeze(0).cpu().numpy()

    def _window(self, frames: list[Float[np.ndarray, "z y x"]], timepoint: int) -> Float[np.ndarray, "t z y x"]:
        """A `window`-frame stack centred on a timepoint, clamped to the video's ends."""
        half = self.window // 2
        start = min(max(timepoint - half, 0), max(len(frames) - self.window, 0))
        return np.stack(frames[start : start + self.window])
