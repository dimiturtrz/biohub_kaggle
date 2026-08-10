"""Unit tests for the joint-trainer pair dataset — the edge matrix read off the GT graph is the load-bearing bit."""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.data.joint_dataset import PairDataset, PairTarget
from core.data.tracks import TrackGraph
from core.data.video import ImageStatistics


def _graph() -> TrackGraph:
    """Two nodes at t=0, two at t=1; one annotated link 0->2 (row 0 -> row 2)."""
    node_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    coordinates = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 2, 2, 2]], dtype=np.int64)
    edges = np.array([[0, 2]], dtype=np.int64)  # node 0 (t=0) -> node 2 (t=1)
    return TrackGraph(node_ids=node_ids, coordinates=coordinates, edges=edges)


def test_enumerate():
    """One pair at t=0 with the two-source, two-target edge matrix marking only the annotated link 0->2."""
    targets = PairTarget.enumerate(_graph(), Path("v.zarr"), q_low=0.0, q_high=1.0)
    assert len(targets) == 1  # only t=0 has nodes at both t and t+1
    target = targets[0]
    assert target.timepoint == 0
    assert target.source_centres.shape == (2, 3) and target.target_centres.shape == (2, 3)
    # sources [row0,row1], targets [row2,row3]; only 0->2 links -> matrix[0,0]=1, rest 0
    assert target.edge_matrix.tolist() == [[1.0, 0.0], [0.0, 0.0]]


def _dataset(video_store: Path) -> PairDataset:
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    target = PairTarget(
        zarr_path=video_store,
        timepoint=0,
        q_low=float(quantiles["0.001"]),
        q_high=float(quantiles["0.999"]),
        source_centres=np.array([[1, 2, 2]], dtype=np.int64),
        target_centres=np.array([[1, 2, 2], [0, 1, 1]], dtype=np.int64),
        edge_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    return PairDataset([target], steps=4, downsample=(1, 1, 1), seed=0)


def test_len(video_store: Path):
    """The stream length is the configured step count, not the corpus size."""
    assert len(_dataset(video_store)) == 4


def test_getitem(video_store: Path):
    """An item is (frame_t, frame_t1, source_centres, target_centres, edge_matrix) with matching shapes."""
    frame_t, frame_t1, sources, targets, matrix = _dataset(video_store)[0]
    assert frame_t.ndim == 3 and frame_t1.ndim == 3
    assert torch.all(frame_t >= 0.0) and torch.all(frame_t1 >= 0.0)
    assert sources.shape == (1, 3) and targets.shape == (2, 3)
    assert matrix.shape == (1, 2)


def test_target_count(video_store: Path):
    """The corpus size is the number of GT pairs, independent of how many steps the stream serves."""
    assert _dataset(video_store).target_count == 1


def test_pair(video_store: Path):
    """A chosen pair's tensors are the same ones the uniform draw would have produced for that corpus index."""
    dataset = _dataset(video_store)
    chosen, drawn = dataset.pair(0), dataset[0]
    assert all(torch.equal(a, b) for a, b in zip(chosen, drawn, strict=True))


def test_stream(video_store: Path):
    """The stream yields every item once, in index order, matching direct indexing."""
    dataset = _dataset(video_store)
    streamed = list(dataset.stream())
    assert len(streamed) == 4
    assert torch.equal(streamed[0][0], dataset[0][0])
