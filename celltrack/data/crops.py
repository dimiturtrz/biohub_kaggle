"""Sampling supervised training crops from a video and its sparse annotation.

A crop is a short temporal window of the normalised volume — the model sees a few timepoints so it can use
motion, and predicts a detection heatmap for the middle one. Because annotation is sparse, a uniformly
random crop would usually contain no labelled cell and no positive signal, so crops are *anchored* on an
annotated cell: that guarantees supervision while the surrounding tissue still supplies the dark negatives
and the excluded-ambiguous regions the mask needs.

Anchoring on a cell narrows what the model sees, though — if every crop had its cell dead-centre the model
could learn "fire in the middle" instead of "fire on a cell". Two augmentations widen the distribution back
out, turning the few thousand annotated anchors into far more effective examples: the anchor is *jittered*
so the cell lands anywhere in the crop, and the crop is randomly *flipped* along each spatial axis (with its
centres flipped to match). This is the standard cure for a detector that memorises its training crops.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from celltrack.data.targets import DetectionTarget
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing

_LOW, _HIGH = 0.001, 0.999
# The anchor is jittered by up to a quarter of the crop in each axis: enough to move the cell well off
# centre, not so far it leaves the crop and loses the supervision the anchor exists to guarantee.
_JITTER_FRACTION = 4
# Each spatial axis is mirrored with even odds — an unbiased coin per axis gives all eight reflections.
_FLIP_PROBABILITY = 0.5
# Intensity jitter for the model's input: acquisitions vary in brightness and contrast, so scale and shift
# each crop's normalised intensities, clipped back to [0, 1]. The supervision target is built from the clean
# crop, not this one, so the dark-background mask threshold is never moved by the augmentation.
_INTENSITY_SCALE = (0.8, 1.2)
_INTENSITY_SHIFT = 0.1


@dataclass(frozen=True)
class TrainingCrop:
    """One supervised example: a temporal input window and the masked target for its central frame."""

    frames: Float[np.ndarray, "t z y x"]
    heatmap: Float[np.ndarray, "z y x"]
    mask: Bool[np.ndarray, "z y x"]


@dataclass(frozen=True)
class CropSampler:
    """Samples fixed-size supervised crops anchored on annotated cells, jittered and flipped for variety.

    A fraction of crops are instead taken from a uniformly-random location — `background_fraction`. Cell-only
    anchoring teaches the model what a cell looks like but never what empty tissue looks like, so at full-frame
    inference it over-fires on the background it never trained against. Random crops (almost all empty in a
    sparsely-annotated volume) supply that confidently-negative signal; the target mask still excludes ambiguous
    bright regions, so an un-annotated cell that lands in a random crop carries no gradient.
    """

    window: int
    size: tuple[int, int, int]
    scale_um: float
    augment: bool = True
    background_fraction: float = 0.0

    def sample(
        self, video: CellVideo, tracks: AnnotatedTracks, spacing: Spacing, rng: np.random.Generator
    ) -> TrainingCrop:
        """One crop — usually anchored on an annotated cell, sometimes on random background — with its target."""
        coordinates = tracks.graph.coordinates
        anchor_t, spatial = self._anchor(video, coordinates, rng)
        timepoints = self._window_timepoints(anchor_t, video.timepoint_count)
        origin = self._crop_origin(spatial, video.volume_shape, rng)
        middle = timepoints[len(timepoints) // 2]
        box = (int(origin[0]), int(origin[1]), int(origin[2]))
        frames = np.stack([video.quantiles.normalise(video.window(t, box, self.size), _LOW, _HIGH) for t in timepoints])
        centres = self._centres_in_crop(coordinates, middle, origin)
        if self.augment:
            frames, centres = self._flip(frames, centres, rng)
        target = DetectionTarget.build(centres, frames[len(timepoints) // 2], spacing, self.scale_um)
        model_input = self._intensity(frames, rng) if self.augment else frames
        return TrainingCrop(frames=model_input, heatmap=target.heatmap, mask=target.mask)

    def _anchor(
        self, video: CellVideo, coordinates: Int[np.ndarray, "n 4"], rng: np.random.Generator
    ) -> tuple[int, Int[np.ndarray, "3"]]:
        """The `(timepoint, zyx)` a crop centres on — an annotated cell, or a random background point."""
        if self.augment and rng.random() < self.background_fraction:
            spatial = rng.integers(np.asarray(video.volume_shape))
            return int(rng.integers(video.timepoint_count)), spatial
        anchor = coordinates[rng.integers(len(coordinates))]
        return int(anchor[0]), anchor[1:]

    def _intensity(
        self, frames: Float[np.ndarray, "t z y x"], rng: np.random.Generator
    ) -> Float[np.ndarray, "t z y x"]:
        """Scale and shift a crop's normalised intensities, clipped to [0, 1] — brightness/contrast variety."""
        scale = rng.uniform(*_INTENSITY_SCALE)
        shift = rng.uniform(-_INTENSITY_SHIFT, _INTENSITY_SHIFT)
        return np.clip(frames * scale + shift, 0.0, 1.0).astype(np.float32)

    def _window_timepoints(self, centre_t: int, timepoint_count: int) -> list[int]:
        """A temporal window around a timepoint, clamped to the video and centred on `centre_t`."""
        half = self.window // 2
        start = min(max(centre_t - half, 0), max(timepoint_count - self.window, 0))
        return list(range(start, start + min(self.window, timepoint_count)))

    def _crop_origin(
        self, centre_zyx: Int[np.ndarray, "3"], volume_shape: tuple[int, int, int], rng: np.random.Generator
    ) -> Int[np.ndarray, "3"]:
        """The lower corner of a crop anchored on a point, jittered for variety and clamped inside the volume."""
        size = np.asarray(self.size)
        jitter = self._jitter(rng)
        wanted = np.asarray(centre_zyx) - size // 2 + jitter
        upper = np.asarray(volume_shape) - size
        return np.clip(wanted, 0, np.maximum(upper, 0))

    def _jitter(self, rng: np.random.Generator) -> Int[np.ndarray, "3"]:
        """A per-axis anchor offset of up to a quarter of the crop, or zero when augmentation is off."""
        if not self.augment:
            return np.zeros(3, dtype=np.int64)
        reach = np.asarray(self.size) // _JITTER_FRACTION
        return rng.integers(-reach, reach + 1)

    def _flip(
        self, frames: Float[np.ndarray, "t z y x"], centres: Int[np.ndarray, "k 3"], rng: np.random.Generator
    ) -> tuple[Float[np.ndarray, "t z y x"], Int[np.ndarray, "k 3"]]:
        """Randomly mirror each spatial axis, moving the centres to match so labels stay aligned to the volume."""
        flipped = centres.copy()
        for axis in range(3):
            if rng.random() < _FLIP_PROBABILITY:
                frames = np.flip(frames, axis=axis + 1)
                flipped[:, axis] = self.size[axis] - 1 - flipped[:, axis]
        return np.ascontiguousarray(frames), flipped

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
