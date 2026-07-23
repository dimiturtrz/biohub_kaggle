"""Running a trained detection U-Net over a whole video to recover cell centres.

The learned counterpart of the classical `BlobDetector`: instead of a Laplacian-of-Gaussian it evaluates
the trained network on each timepoint's temporal window, then recovers centres from the predicted heatmap
with the *same* anisotropic non-maximum suppression the classical detector uses. Sharing the peak-picker is
what makes the two comparable at a matched node count — the only thing that differs between them is the
response volume, which is the whole point of the comparison.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float

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
        normalised = list(video.normalised_frames(_LOW, _HIGH))
        coordinates = [
            np.column_stack([np.full(len(centres), timepoint), centres])
            for timepoint in range(video.timepoint_count)
            for centres in [self.peaks.centres(self._heatmap(normalised, timepoint), per_frame)]
        ]
        stacked = np.concatenate(coordinates).astype(np.int64) if coordinates else np.empty((0, 4), np.int64)
        return TrackGraph(
            node_ids=np.arange(len(stacked), dtype=np.int64),
            coordinates=stacked,
            edges=np.empty((0, 2), dtype=np.int64),
        )

    def _heatmap(self, frames: list[Float[np.ndarray, "z y x"]], timepoint: int) -> Float[np.ndarray, "z y x"]:
        """The network's cellness map for one timepoint, from the temporal window centred on it."""
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
