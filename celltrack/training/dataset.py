"""A torch dataset of supervised crops drawn from the annotated videos.

Each item is one crop sampled on the fly, so an epoch is a number of *steps* rather than a fixed corpus —
the videos are large and sparsely annotated, and resampling gives fresh crops every step. Sampling is
seeded per index so a run is reproducible without materialising the crops to disk.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data import Dataset

from celltrack.training.crops import CropSampler
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing

_Item = tuple[Float[Tensor, "t z y x"], Float[Tensor, "z y x"], Bool[Tensor, "z y x"]]


@dataclass(frozen=True)
class AnnotatedVideo:
    """A video paired with its annotation and spacing — one source of training crops."""

    video: CellVideo
    tracks: AnnotatedTracks
    spacing: Spacing

    def cells_per_frame(self) -> float:
        """Mean annotated cells per timepoint — the video's cell density."""
        return len(self.tracks.graph.coordinates) / max(self.video.timepoint_count, 1)


@dataclass(frozen=True)
class DensitySampling:
    """The distribution over source videos a crop's source is drawn from — uniform, or density-stratified.

    Under uniform video draws a detector sees positive signal in proportion to cell count, so it over-fits the
    dense majority and loses recall on sparse videos. Stratifying rebalances that: density spans an order of
    magnitude (~1 to ~20 cells per frame), so the strata are octaves — each video binned by
    `floor(log2(cells_per_frame))` — and every octave is drawn with equal probability, its videos sharing that
    mass evenly.
    """

    stratified: bool

    def weights(self, sources: list[AnnotatedVideo]) -> Float[np.ndarray, "n"]:
        """Per-source sampling probabilities over the videos."""
        return self._stratified(sources) if self.stratified else np.full(len(sources), 1.0 / len(sources))

    def _stratified(self, sources: list[AnnotatedVideo]) -> Float[np.ndarray, "n"]:
        """Weights that make each cell-density octave equally likely, however many videos share it."""
        octave = np.floor(np.log2(np.maximum([source.cells_per_frame() for source in sources], 1.0))).astype(int)
        strata, counts = np.unique(octave, return_counts=True)
        per_stratum = dict(zip(strata, counts, strict=True))
        weights = np.array([1.0 / (len(strata) * per_stratum[value]) for value in octave])
        return weights / weights.sum()


class CropDataset(Dataset[_Item]):
    """Fixed-length stream of supervised crops, sampled fresh and reproducibly per index."""

    def __init__(
        self,
        sources: list[AnnotatedVideo],
        sampler: CropSampler,
        steps: int,
        seed: int,
        weights: Float[np.ndarray, "n"],
    ) -> None:
        self._sources = sources
        self._sampler = sampler
        self._steps = steps
        self._seed = seed
        self._weights = weights

    def __len__(self) -> int:
        """How many crops make one epoch."""
        return self._steps

    @override
    def __getitem__(self, index: int) -> _Item:
        """Sample one crop, seeded by its index so the same index always yields the same crop."""
        rng = np.random.default_rng(self._seed + index)
        source = self._sources[int(rng.choice(len(self._sources), p=self._weights))]
        crop = self._sampler.sample(source.video, source.tracks, source.spacing, rng)
        return (
            torch.from_numpy(crop.frames),
            torch.from_numpy(crop.heatmap),
            torch.from_numpy(crop.mask),
        )
