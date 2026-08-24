"""Cell-centred raw boxes drawn from the annotated corpus — the encoder's self-supervised diet.

The encoder learns identity from many cells seen twice under nuisance, so the data is exactly that: a box around
every annotated cell across the training videos, handed out in batches as two augmented views of itself. The
labels are used ONLY to centre the boxes on cells (a cell-free box teaches nothing about telling cells apart);
no track, parent or successor is read, so this stays self-supervised on identity and never touches the
association signal the gate later tests it on.

A batch is drawn from a SINGLE video so every box shares one voxel shape and stacks without resampling — the
competition's stores share a spacing, but sampling within a video makes that an observation rather than an
assumption. Epochs cover every video in turn, so the per-step locality costs diversity across a step, not across
training.

The two views are nuisance-only by construction: an in-plane flip, a brightness gain and offset, and read noise.
Each is something the imaging does to the SAME cell between frames — the invariances association actually wants —
and none of them is identity, which is what keeps the objective from being satisfied by discarding the cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from celltrack.data.raw_boxes import RawBoxes
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.paths import DataRoot

_FLIP_PROBABILITY = 0.5


@dataclass(frozen=True)
class AnnotatedVideo:
    """One training video paired with its ground-truth track graph — the boxes' source."""

    video: CellVideo
    graph: TrackGraph
    spacing: Spacing

    @property
    def cell_count(self) -> int:
        """How many annotated cells this video offers as box centres."""
        return len(self.graph.positions())


@dataclass(frozen=True)
class PatchCorpus:
    """Every annotated cell in a split, indexed by video, as a source of augmented box batches."""

    videos: tuple[AnnotatedVideo, ...]
    radius_um: float

    @classmethod
    def build(cls, root: DataRoot, split: str, radius_um: float) -> PatchCorpus:
        """Open each split video beside its GEFF annotation, keeping its graph as the box centres."""
        annotated = tuple(
            AnnotatedVideo(
                video=(video := CellVideo.from_ome_zarr(path)),
                graph=AnnotatedTracks.from_geff(root.track_store(path)).graph,
                spacing=video.spacing,
            )
            for path in root.videos(split)
        )
        return cls(videos=annotated, radius_um=radius_um)

    def sample(self, rng: np.random.Generator, batch_size: int) -> Float[np.ndarray, "b z y x"]:
        """A batch of raw boxes from one random video — fewer than asked if some fell off the volume edge."""
        chosen = self.videos[int(rng.integers(len(self.videos)))]
        rows = rng.choice(chosen.cell_count, size=min(batch_size, chosen.cell_count), replace=False)
        return RawBoxes.of(chosen.video, chosen.graph, np.asarray(rows), chosen.spacing, self.radius_um).boxes


class NuisanceViews:
    """The identity-preserving augmentation that turns one box into two views — the SAME cell, seen differently.

    Every step is something the imaging does to a cell between frames — a per-box standardisation, an in-plane
    flip, a brightness gain and offset, and read noise — and none of them is identity, which is what keeps the
    objective from being satisfied by discarding the cell. z is anisotropic and is never flipped.
    """

    @staticmethod
    def two(
        boxes: Float[Tensor, "b z y x"], generator: torch.Generator
    ) -> tuple[Float[Tensor, "b 1 z y x"], Float[Tensor, "b 1 z y x"]]:
        """Two independent nuisance augmentations of the same boxes — the SAME cells, seen differently."""
        return NuisanceViews._augment(boxes, generator), NuisanceViews._augment(boxes, generator)

    @staticmethod
    def _augment(boxes: Float[Tensor, "b z y x"], generator: torch.Generator) -> Float[Tensor, "b 1 z y x"]:
        """One identity-preserving view: normalise per box, in-plane flip, brightness gain and offset, read noise."""
        view = NuisanceViews._standardise(boxes)
        view = NuisanceViews._random_flip(view, generator)
        view = NuisanceViews._brightness(view, generator)
        return NuisanceViews._read_noise(view, generator).unsqueeze(1)

    @staticmethod
    def _standardise(boxes: Float[Tensor, "b z y x"]) -> Float[Tensor, "b z y x"]:
        """Zero-mean, unit-variance per box — the encoder sees shape, not the brightness that confounds it."""
        flat = boxes.flatten(1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        return ((flat - mean) / std).view_as(boxes)

    @staticmethod
    def _random_flip(boxes: Float[Tensor, "b z y x"], generator: torch.Generator) -> Float[Tensor, "b z y x"]:
        """Flip a random half of the batch in each in-plane axis — plane is isotropic, z is not, so z is left alone."""
        view = boxes
        for axis in (2, 3):
            mask = torch.rand(boxes.shape[0], device=boxes.device, generator=generator) < _FLIP_PROBABILITY
            flipped = torch.flip(view, dims=(axis,))
            view = torch.where(mask.view(-1, 1, 1, 1), flipped, view)
        return view

    @staticmethod
    def _brightness(boxes: Float[Tensor, "b z y x"], generator: torch.Generator) -> Float[Tensor, "b z y x"]:
        """Per-box gain in [0.8, 1.2] and a small offset — the acquisition's gain and pedestal, not identity."""
        shape = (boxes.shape[0], 1, 1, 1)
        gain = 0.8 + 0.4 * torch.rand(shape, device=boxes.device, generator=generator)
        offset = 0.2 * (torch.rand(shape, device=boxes.device, generator=generator) - 0.5)
        return boxes * gain + offset

    @staticmethod
    def _read_noise(boxes: Float[Tensor, "b z y x"], generator: torch.Generator) -> Float[Tensor, "b z y x"]:
        """Additive Gaussian read noise at the standardised scale — the sensor floor, blind to the cell."""
        noise = torch.randn(boxes.shape, device=boxes.device, generator=generator)
        return boxes + 0.1 * noise
