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


class CropDataset(Dataset[_Item]):
    """Fixed-length stream of supervised crops, sampled fresh and reproducibly per index."""

    def __init__(self, sources: list[AnnotatedVideo], sampler: CropSampler, steps: int, seed: int) -> None:
        self._sources = sources
        self._sampler = sampler
        self._steps = steps
        self._seed = seed

    def __len__(self) -> int:
        """How many crops make one epoch."""
        return self._steps

    @override
    def __getitem__(self, index: int) -> _Item:
        """Sample one crop, seeded by its index so the same index always yields the same crop."""
        rng = np.random.default_rng(self._seed + index)
        source = self._sources[rng.integers(len(self._sources))]
        crop = self._sampler.sample(source.video, source.tracks, source.spacing, rng)
        return (
            torch.from_numpy(crop.frames),
            torch.from_numpy(crop.heatmap),
            torch.from_numpy(crop.mask),
        )
