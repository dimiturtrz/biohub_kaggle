"""Unit tests for the joint-trainer pair dataset — the edge matrix read off the GT graph is the load-bearing bit."""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr

from celltrack.data.frame_source import ZarrFrames
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
    targets = PairTarget.enumerate(_graph(), ZarrFrames(Path("v.zarr"), q_low=0.0, q_high=1.0))
    assert len(targets) == 1  # only t=0 has nodes at both t and t+1
    target = targets[0]
    assert target.timepoint == 0
    assert target.source_centres.shape == (2, 3) and target.target_centres.shape == (2, 3)
    # sources [row0,row1], targets [row2,row3]; only 0->2 links -> matrix[0,0]=1, rest 0
    assert target.edge_matrix.tolist() == [[1.0, 0.0], [0.0, 0.0]]


def _three_frame_graph() -> TrackGraph:
    """One node per timepoint at t=0,1,2, linked in a chain — so the pair at t=1 has a real predecessor."""
    node_ids = np.array([0, 1, 2], dtype=np.int64)
    coordinates = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 1, 2, 2]], dtype=np.int64)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    return TrackGraph(node_ids=node_ids, coordinates=coordinates, edges=edges)


def test_enumerate_carries_the_predecessors_centres():
    """The pair at t=1 knows t=0's annotated centres; the video's FIRST pair has no predecessor and says so."""
    first, second = PairTarget.enumerate(_three_frame_graph(), ZarrFrames(Path("v.zarr"), q_low=0.0, q_high=1.0))

    assert first.previous_centres is None
    assert second.previous_centres is not None
    assert second.previous_centres.tolist() == [[0, 0, 0]]


def _target(video_store: Path, timepoint: int = 0, previous: np.ndarray | None = None) -> PairTarget:
    """One synthetic pair over the tiny video fixture, optionally carrying a predecessor's centres."""
    statistics = cast(ImageStatistics, zarr.open_group(video_store, mode="r").attrs["image_statistics"])
    quantiles = statistics["quantiles"]
    return PairTarget(
        frames=ZarrFrames(video_store, float(quantiles["0.001"]), float(quantiles["0.999"])),
        timepoint=timepoint,
        source_centres=np.array([[1, 2, 2]], dtype=np.int64),
        target_centres=np.array([[1, 2, 2], [0, 1, 1]], dtype=np.int64),
        edge_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
        previous_centres=previous,
    )


def _dataset(video_store: Path) -> PairDataset:
    return PairDataset([_target(video_store)], steps=4, downsample=(1, 1, 1), seed=0)


def test_len(video_store: Path):
    """The stream length is the configured step count, not the corpus size."""
    assert len(_dataset(video_store)) == 4


def test_getitem(video_store: Path):
    """An item carries both frames, both centre sets and the edge matrix — and no predecessor unless asked."""
    sample = _dataset(video_store)[0]
    assert sample.frame_t.ndim == 3 and sample.frame_t1.ndim == 3
    assert torch.all(sample.frame_t >= 0.0) and torch.all(sample.frame_t1 >= 0.0)
    assert sample.source_centres.shape == (1, 3) and sample.target_centres.shape == (2, 3)
    assert sample.edge_matrix.shape == (1, 2)
    assert sample.has_previous is False


def test_has_previous(video_store: Path):
    """A pair at t=1 whose t=0 is annotated serves that frame and its centres — the prior gap the velocity needs."""
    target = _target(video_store, timepoint=1, previous=np.array([[0, 1, 1]], dtype=np.int64))
    dataset = PairDataset([target], steps=1, downsample=(1, 1, 1), seed=0, with_previous=True)

    sample = dataset.pair(0)

    assert sample.has_previous
    assert sample.previous_frame is not None and sample.previous_centres is not None
    assert sample.previous_frame.shape == sample.frame_t.shape
    assert sample.previous_centres.tolist() == [[0, 1, 1]]
    frame_zero = PairDataset([_target(video_store)], steps=1, downsample=(1, 1, 1), seed=0).pair(0).frame_t
    assert torch.equal(sample.previous_frame, frame_zero)  # the pair at t=1 reads frame 0 as its predecessor


def test_pair_has_no_predecessor_for_a_videos_first_pair(video_store: Path):
    """Asked for history a first pair does not have, the sample is explicitly empty rather than silently zeroed."""
    dataset = PairDataset([_target(video_store)], steps=1, downsample=(1, 1, 1), seed=0, with_previous=True)

    sample = dataset.pair(0)

    assert sample.previous_frame is None and sample.previous_centres is None
    assert sample.has_previous is False


def test_to(video_store: Path):
    """Moving a sample without history leaves the absent halves absent — no zero tensor invented on the way."""
    moved = _dataset(video_store)[0].to("cpu")
    assert moved.previous_frame is None
    assert moved.edge_matrix.device.type == "cpu"


def test_pair(video_store: Path):
    """A chosen pair's tensors are the same ones the uniform draw would have produced for that corpus index."""
    dataset = _dataset(video_store)
    chosen, drawn = dataset.pair(0), dataset[0]
    assert torch.equal(chosen.frame_t, drawn.frame_t)
    assert torch.equal(chosen.edge_matrix, drawn.edge_matrix)


def test_stream(video_store: Path):
    """The stream yields every item once, in index order, matching direct indexing."""
    dataset = _dataset(video_store)
    streamed = list(dataset.stream())
    assert len(streamed) == 4
    assert torch.equal(streamed[0].frame_t, dataset[0].frame_t)
