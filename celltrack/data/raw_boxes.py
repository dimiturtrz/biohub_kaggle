"""Full-resolution cell boxes read straight from the unpooled store — the raw-voxel primitive both the appearance
diagnosis and the patch encoder read from.

A box is a direct slice of the raw array around a cell centre: `TrackGraph` coordinates are already
full-resolution voxel indices, so no rescaling is involved, and `CellVideo.window` slices zarr rather than
materialising a timepoint. This is a data reader with no analysis or model dependency, so it sits at the data
layer where both the analysis diagnosis (top) and the encoder's dataset (data) can read it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from jaxtyping import Float, Int

from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing

# A box "just big enough to contain the cell", not its neighbourhood — the neighbourhood is what makes two cells
# look alike, so the collar around the cell is kept thin.
BOX_RADIUS_UM = 3.25


@dataclass(frozen=True)
class RawBoxes:
    """One full-resolution box per node, read straight from the unpooled store.

    Coordinates on a `TrackGraph` are already FULL-RESOLUTION voxel indices — the read-out multiplies its
    downsampled peaks back up — so a box is a direct slice of the raw array and no rescaling is involved.
    `CellVideo.window` slices zarr rather than materialising a timepoint, so only the chunks the box touches
    decompress.

    A node whose box would fall off the volume edge is dropped rather than padded: padding would fabricate
    voxels and make a boundary cell correlate with anything else that was padded the same way.
    """

    boxes: Float[np.ndarray, "k z y x"]
    kept: Int[np.ndarray, "k"]
    """Positions in the REQUESTED row list that survived — the handle that keeps a triple aligned.

    A box is dropped when it would fall off the volume, and the three columns of a triple drop independently,
    so a positional zip of two box arrays would silently pair a source with another triple's target. Carrying
    which requests survived lets the caller intersect and compare like with like.
    """

    @classmethod
    def of(
        cls,
        video: CellVideo,
        graph: TrackGraph,
        rows: Int[np.ndarray, "k"],
        spacing: Spacing,
        radius_um: float = BOX_RADIUS_UM,
    ) -> Self:
        """Read a box around each named node, keeping only those that fit entirely inside the volume."""
        size = cls._size(spacing, radius_um)
        volume = video.volume_shape
        kept: list[int] = []
        boxes: list[np.ndarray] = []
        for index, row in enumerate(rows.tolist()):
            corner = [int(c) - e // 2 for c, e in zip(graph.positions()[row], size, strict=True)]
            origin: tuple[int, int, int] = (corner[0], corner[1], corner[2])
            if any(s < 0 or s + e > lim for s, e, lim in zip(origin, size, volume, strict=True)):
                continue
            boxes.append(video.window(int(graph.timepoints()[row]), origin, size).astype(np.float32))
            kept.append(index)
        stacked = np.stack(boxes) if boxes else np.zeros((0, *size), dtype=np.float32)
        return cls(boxes=stacked, kept=np.asarray(kept, dtype=np.int64))

    def select(self, requested: Int[np.ndarray, "m"]) -> Float[np.ndarray, "m z y x"]:
        """This column's boxes for a chosen set of REQUEST positions, in the order asked for."""
        order = {int(position): index for index, position in enumerate(self.kept)}
        return self.boxes[[order[int(position)] for position in requested]]

    @staticmethod
    def _size(spacing: Spacing, radius_um: float) -> tuple[int, int, int]:
        """The box in full-resolution voxels — odd per axis, so the cell centre is the centre voxel."""
        return tuple(2 * round(radius_um / axis) + 1 for axis in spacing.as_array())  # type: ignore[return-value]

    def aligned(self) -> Self:
        """Each box re-centred on its own brightest voxel — the control for a miscentred detection.

        Mislink endpoints sit measurably further from their annotated centres than the population, so a null
        under detection centring is ambiguous between "these cells look alike" and "our boxes are off the
        cells". Rolling each box so its maximum lands at the centre removes the second reading without using
        any annotation, and it is what a matcher would do anyway.
        """
        rolled = np.empty_like(self.boxes)
        centre = np.asarray(self.boxes.shape[1:]) // 2
        for index, box in enumerate(self.boxes):
            peak = np.unravel_index(int(np.argmax(box)), box.shape)
            rolled[index] = np.roll(box, tuple(centre - np.asarray(peak)), axis=(0, 1, 2))
        return type(self)(boxes=rolled, kept=self.kept)
