from pathlib import Path

import numpy as np

from celltrack.training.crops import CropSampler
from celltrack.training.dataset import AnnotatedVideo, CropDataset, DensitySampling
from core.data.tracks import AnnotatedTracks, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing

SAMPLER = CropSampler(window=1, size=(2, 4, 4), scale_um=2.0)


def a_source(video_store: Path, tracks: AnnotatedTracks) -> AnnotatedVideo:
    return AnnotatedVideo(
        video=CellVideo.from_ome_zarr(video_store),
        tracks=tracks,
        spacing=Spacing(z=1.0, y=1.0, x=1.0),
    )


def a_source_of_density(video_store: Path, cells_per_frame: int) -> AnnotatedVideo:
    """A source whose annotation carries `cells_per_frame` cells at every timepoint — a chosen cell density."""
    video = CellVideo.from_ome_zarr(video_store)
    count = cells_per_frame * video.timepoint_count
    graph = TrackGraph(
        node_ids=np.arange(count),
        coordinates=np.zeros((count, 4), dtype=np.int64),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    tracks = AnnotatedTracks(graph=graph, estimated_node_count=count)
    return AnnotatedVideo(video=video, tracks=tracks, spacing=Spacing(z=1.0, y=1.0, x=1.0))


def uniform_over(sources: list[AnnotatedVideo]) -> np.ndarray:
    return DensitySampling(stratified=False).weights(sources)


def test_len(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    source = a_source(video_store, in_bounds_tracks)
    dataset = CropDataset([source], SAMPLER, steps=5, seed=0, weights=uniform_over([source]))
    assert len(dataset) == 5


def test_getitem(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """An item is a (frames, heatmap, mask) tensor triple, reproducible from its index."""
    source = a_source(video_store, in_bounds_tracks)
    dataset = CropDataset([source], SAMPLER, steps=5, seed=0, weights=uniform_over([source]))
    frames, heatmap, mask = dataset[0]
    assert frames.shape == (1, 2, 4, 4)
    assert heatmap.shape == (2, 4, 4)
    assert bool((dataset[0][0] == frames).all())


def test_cells_per_frame(video_store: Path):
    """A video's density is its annotated cells divided by its timepoints."""
    assert a_source_of_density(video_store, cells_per_frame=3).cells_per_frame() == 3.0


def test_weights(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """Uniform sampling makes every video equally likely, as a distribution."""
    sources = [a_source(video_store, in_bounds_tracks) for _ in range(3)]
    weights = DensitySampling(stratified=False).weights(sources)
    assert np.allclose(weights, 1 / 3)
    assert np.isclose(weights.sum(), 1.0)


def test_weights_stratify_each_octave_equally(video_store: Path):
    """Stratified sampling gives each density octave equal mass, however many videos share it."""
    sparse = [a_source_of_density(video_store, 2), a_source_of_density(video_store, 3)]  # octave floor(log2)=1
    dense = [a_source_of_density(video_store, 10)]  # octave floor(log2)=3
    weights = DensitySampling(stratified=True).weights([*sparse, *dense])
    assert np.isclose(weights[:2].sum(), 0.5)
    assert np.isclose(weights[2], 0.5)
