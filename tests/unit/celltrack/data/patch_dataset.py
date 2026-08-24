"""Unit tests for the patch-encoder diet — the nuisance augmentation and the box-batch source.

`two_views` must add the channel axis, keep the batch and spatial shape, standardise each box to zero mean, flip
only in plane (z is anisotropic and left alone), and produce two DIFFERENT views of the same cells. `PatchCorpus`
counts, samples and builds over videos: the real-zarr IO of `build` is stubbed with a duck root so the wiring is
pinned without a store, and `sample`/`cell_count` run against a one-frame fake video. The fixtures are a small
random batch and a hand-placed blob whose statistics are known.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from celltrack.data.patch_dataset import AnnotatedVideo, NuisanceViews, PatchCorpus
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)


def _blob(shape: tuple[int, int, int], centre: tuple[int, int, int], sigma: float = 2.0) -> np.ndarray:
    """A single Gaussian blob with a definite centre — a stand-in for one cell's intensity."""
    grids = np.meshgrid(*[np.arange(dim) for dim in shape], indexing="ij")
    squared = sum((axis - c) ** 2 for axis, c in zip(grids, centre, strict=True))
    return np.exp(-squared / (2 * sigma**2)).astype(np.float32)


class _FakeVideo:
    """A one-frame volume with a spacing, so a box read has known content — no zarr, no data root."""

    spacing = _SPACING

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


def test_two():
    """Each view gains a channel axis, keeps the box extent, and the two are independent draws of the same cells."""
    boxes = torch.randn(8, 5, 17, 17)

    view_one, view_two = NuisanceViews.two(boxes, torch.Generator().manual_seed(0))

    assert view_one.shape == (8, 1, 5, 17, 17)
    assert view_two.shape == (8, 1, 5, 17, 17)
    assert not torch.allclose(view_one, view_two)  # different flips/brightness/noise per view


def test_two_views_z_not_flipped():
    """A z-asymmetric box keeps its z ordering — flips act only in the isotropic plane, never on the coarse axis.

    A box bright on one z-half and dark on the other would, if z were ever flipped, sometimes come back with the
    halves swapped. Averaging over the plane and comparing the two z-halves across many boxes shows z is untouched.
    """
    boxes = torch.zeros(64, 4, 8, 8)
    boxes[:, :2] = 1.0  # bright on the low-z half only

    view_one, _ = NuisanceViews.two(boxes, torch.Generator().manual_seed(0))
    low_half = view_one[:, 0, :2].mean()
    high_half = view_one[:, 0, 2:].mean()

    assert low_half > high_half  # standardisation preserves sign; a z-flip on any box would erode the gap


def test_cell_count():
    """A video offers as many box centres as its graph has annotated nodes — no video read is needed to count."""
    annotated = AnnotatedVideo(video=None, graph=_graph([[0, 5, 5, 5], [1, 6, 6, 6], [1, 7, 7, 7]]), spacing=_SPACING)  # type: ignore[arg-type]

    assert annotated.cell_count == 3


def test_sample():
    """A batch of boxes is drawn from one video, at most as many as the video has cells."""
    video = _FakeVideo(_blob((20, 40, 40), (10, 20, 20)))
    graph = _graph([[0, 10, 20, 20], [0, 12, 22, 22]])
    corpus = PatchCorpus(videos=(AnnotatedVideo(video=video, graph=graph, spacing=_SPACING),), radius_um=2.0)  # type: ignore[arg-type]

    boxes = corpus.sample(np.random.default_rng(0), batch_size=8)

    assert boxes.ndim == 4  # (b, z, y, x), the channel axis is added later by two_views
    assert boxes.shape[0] <= 2  # never more boxes than the video has cells


def test_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`build` opens each split video beside its GEFF annotation — the IO is stubbed so only the wiring is tested."""
    from celltrack.data import patch_dataset as module  # noqa: PLC0415 — test-local, to reach the IO helpers

    graph = _graph([[0, 10, 20, 20], [0, 12, 22, 22]])
    video = _FakeVideo(_blob((20, 40, 40), (10, 20, 20)))
    monkeypatch.setattr(module.CellVideo, "from_ome_zarr", classmethod(lambda cls, path: video))
    monkeypatch.setattr(
        module.AnnotatedTracks, "from_geff", classmethod(lambda cls, store: type("_T", (), {"graph": graph})())
    )

    class _Root:
        def videos(self, split: str) -> list[Path]:
            return [tmp_path / "a.zarr"]

        def track_store(self, path: Path) -> Path:
            return path

    corpus = PatchCorpus.build(_Root(), "train", radius_um=2.0)  # type: ignore[arg-type]

    assert len(corpus.videos) == 1
    assert corpus.videos[0].cell_count == 2  # both annotated nodes carried through
