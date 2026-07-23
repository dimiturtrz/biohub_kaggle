from pathlib import Path

import numpy as np

from celltrack.training.crops import CropSampler
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo
from core.geometry import Spacing

SAMPLER = CropSampler(window=1, size=(2, 4, 4), scale_um=2.0)
SPACING = Spacing(z=1.0, y=1.0, x=1.0)


def test_sample(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """A crop carries a temporal window of frames plus a matching heatmap and mask for its centre frame."""
    video = CellVideo.from_ome_zarr(video_store)
    crop = SAMPLER.sample(video, in_bounds_tracks, SPACING, np.random.default_rng(0))
    assert crop.frames.shape == (1, 2, 4, 4)
    assert crop.heatmap.shape == (2, 4, 4)
    assert crop.mask.shape == (2, 4, 4)


def test_sample_is_centred_on_an_annotated_cell(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """Crops are anchored on annotated cells, so every crop contains at least one supervised peak."""
    video = CellVideo.from_ome_zarr(video_store)
    crop = SAMPLER.sample(video, in_bounds_tracks, SPACING, np.random.default_rng(1))
    assert crop.heatmap.max() == 1.0
