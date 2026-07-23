from pathlib import Path

from celltrack.training.crops import CropSampler
from celltrack.training.dataset import AnnotatedVideo, CropDataset
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing

SAMPLER = CropSampler(window=1, size=(2, 4, 4), scale_um=2.0)


def a_source(video_store: Path, tracks: AnnotatedTracks) -> AnnotatedVideo:
    return AnnotatedVideo(
        video=CellVideo.from_ome_zarr(video_store),
        tracks=tracks,
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
    )


def test_len(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    dataset = CropDataset([a_source(video_store, in_bounds_tracks)], SAMPLER, steps=5, seed=0)
    assert len(dataset) == 5


def test_getitem(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """An item is a (frames, heatmap, mask) tensor triple, reproducible from its index."""
    dataset = CropDataset([a_source(video_store, in_bounds_tracks)], SAMPLER, steps=5, seed=0)
    frames, heatmap, mask = dataset[0]
    assert frames.shape == (1, 2, 4, 4)
    assert heatmap.shape == (2, 4, 4)
    assert bool((dataset[0][0] == frames).all())
