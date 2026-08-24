"""Unit tests for the raw-box reader — the volume-edge drop, request-order selection, and peak re-centring.

The reader's contract is about which nodes survive and how their boxes line up, so the fixtures build a volume
with hand-placed content and read known nodes: an interior node keeps its box, a corner node is dropped, and a
peak-off-centre box rolls back to centre. No zarr and no data root — a one-frame array stands in for the store.
"""

import numpy as np

from celltrack.data.raw_boxes import RawBoxes
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)


def _blob(shape: tuple[int, int, int], centre: tuple[int, int, int], sigma: float = 2.0) -> np.ndarray:
    """A single Gaussian blob with a definite centre — a stand-in for one cell's intensity."""
    grids = np.meshgrid(*[np.arange(dim) for dim in shape], indexing="ij")
    squared = sum((axis - c) ** 2 for axis, c in zip(grids, centre, strict=True))
    return np.exp(-squared / (2 * sigma**2)).astype(np.float32)


class _FakeVideo:
    """A one-frame volume, so a box read has known content — no zarr, no data root."""

    def __init__(self, volume: np.ndarray) -> None:
        self._volume = volume

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        return self._volume.shape  # type: ignore[return-value]

    def window(self, timepoint: int, origin: tuple[int, int, int], size: tuple[int, int, int]) -> np.ndarray:
        z, y, x = origin
        dz, dy, dx = size
        return self._volume[z : z + dz, y : y + dy, x : x + dx]


def _graph(coordinates: list[list[int]]) -> TrackGraph:
    """A track graph over consecutively-numbered node ids at the given `(t, z, y, x)` coordinates."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.int64),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_of():
    """A box is read around each node, and a node too close to the edge to fit a full box is dropped."""
    video = _FakeVideo(_blob((20, 40, 40), (10, 20, 20)))
    graph = _graph([[0, 10, 20, 20], [0, 0, 0, 0]])  # second node sits in the corner, box would fall off

    boxes = RawBoxes.of(video, graph, np.array([0, 1]), _SPACING, radius_um=2.0)

    assert boxes.kept.tolist() == [0]  # only the interior node survived
    assert boxes.boxes.shape[0] == 1


def test_select():
    """Boxes come back for the requested REQUEST positions in the asked order, keeping a triple aligned."""
    boxes = RawBoxes(boxes=np.arange(3 * 8).reshape(3, 2, 2, 2).astype(np.float32), kept=np.array([0, 2, 5]))

    picked = boxes.select(np.array([5, 0]))

    assert np.array_equal(picked[0], boxes.boxes[2])  # request-position 5 is the third stored box
    assert np.array_equal(picked[1], boxes.boxes[0])


def test_aligned():
    """A box is rolled so its brightest voxel lands at the centre — the miscentring control, no annotation used."""
    boxes = RawBoxes(boxes=_blob((5, 9, 9), (2, 6, 3))[None], kept=np.array([0]))  # peak off-centre

    rolled = boxes.aligned().boxes[0]

    peak = np.unravel_index(int(np.argmax(rolled)), rolled.shape)
    assert peak == (2, 4, 4)  # the maximum now sits at the box centre
