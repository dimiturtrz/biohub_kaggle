"""Sampling supervised training crops from a video and its sparse annotation.

A crop is a short temporal window of the normalised volume — the model sees a few timepoints so it can use
motion, and predicts a detection heatmap for the middle one. Because annotation is sparse, a uniformly
random crop would usually contain no labelled cell and no positive signal, so crops are centred on an
annotated cell: that guarantees supervision while the surrounding tissue still supplies the dark negatives
and the excluded-ambiguous regions the mask needs.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from celltrack.training.targets import DetectionTarget
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing

_LOW, _HIGH = 0.001, 0.999


@dataclass(frozen=True)
class TrainingCrop:
    """One supervised example: a temporal input window and the masked target for its central frame."""

    frames: Float[np.ndarray, "t z y x"]
    heatmap: Float[np.ndarray, "z y x"]
    mask: Bool[np.ndarray, "z y x"]


@dataclass(frozen=True)
class CropSampler:
    """Samples fixed-size supervised crops centred on annotated cells."""

    window: int
    size: tuple[int, int, int]
    scale_um: float

    def sample(
        self, video: CellVideo, tracks: AnnotatedTracks, spacing: Spacing, rng: np.random.Generator
    ) -> TrainingCrop:
        """One crop centred on a randomly chosen annotated cell, with its masked detection target."""
        coordinates = tracks.graph.coordinates
        anchor = coordinates[rng.integers(len(coordinates))]
        timepoints = self._window_timepoints(int(anchor[0]), video.timepoint_count)
        origin = self._crop_origin(anchor[1:], video.volume_shape)
        frames = np.stack([self._crop(video.frame(t), origin) for t in timepoints])
        centre_frame = video.quantiles.normalise(video.frame(timepoints[len(timepoints) // 2]), _LOW, _HIGH)
        centre_volume = self._crop(centre_frame, origin)
        centres = self._centres_in_crop(coordinates, timepoints[len(timepoints) // 2], origin)
        target = DetectionTarget.build(centres, centre_volume, spacing, self.scale_um)
        normalised = np.stack([video.quantiles.normalise(frame, _LOW, _HIGH) for frame in frames])
        return TrainingCrop(frames=normalised, heatmap=target.heatmap, mask=target.mask)

    def _window_timepoints(self, centre_t: int, timepoint_count: int) -> list[int]:
        """A temporal window around a timepoint, clamped to the video and centred on `centre_t`."""
        half = self.window // 2
        start = min(max(centre_t - half, 0), max(timepoint_count - self.window, 0))
        return list(range(start, start + min(self.window, timepoint_count)))

    def _crop_origin(
        self, centre_zyx: Int[np.ndarray, "3"], volume_shape: tuple[int, int, int]
    ) -> Int[np.ndarray, "3"]:
        """The lower corner of a crop centred on a point, clamped so the crop stays inside the volume."""
        wanted = np.asarray(centre_zyx) - np.asarray(self.size) // 2
        upper = np.asarray(volume_shape) - np.asarray(self.size)
        return np.clip(wanted, 0, np.maximum(upper, 0))

    def _crop(self, volume: Float[np.ndarray, "z y x"], origin: Int[np.ndarray, "3"]) -> Float[np.ndarray, "z y x"]:
        """Extract the fixed-size box at an origin."""
        window = tuple(slice(int(o), int(o) + s) for o, s in zip(origin, self.size, strict=True))
        return volume[window]

    def _centres_in_crop(
        self,
        coordinates: Int[np.ndarray, "n 4"],
        centre_t: int,
        origin: Int[np.ndarray, "3"],
    ) -> Int[np.ndarray, "k 3"]:
        """The annotated centres at the middle timepoint that fall inside the crop, in crop-local voxels."""
        here = coordinates[coordinates[:, 0] == centre_t][:, 1:]
        local = here - origin
        inside = np.all((local >= 0) & (local < np.asarray(self.size)), axis=1)
        return local[inside]
