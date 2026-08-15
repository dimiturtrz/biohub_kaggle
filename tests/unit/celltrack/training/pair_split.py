"""Unit tests for the run's pair-set assembly — the corpus geometry, and the synthetic half's precondition."""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import zarr

from celltrack.data.frame_source import DOWNSAMPLE_ATTR, PooledFrames, ZarrFrames
from celltrack.data.joint_dataset import PairTarget
from celltrack.eval.proxy import TestMovieProxy
from celltrack.training.joint_config import DataCfg, JointTrainConfig
from celltrack.training.pair_split import PairSplit
from core.data.tracks import AnnotatedTracks
from core.geometry import Spacing
from core.paths import DataRoot

_LOG = logging.getLogger("tests.pair_split")


def _target(video_store: Path) -> PairTarget:
    return PairTarget(
        frames=ZarrFrames(video_store, 0.0, 1.0),
        timepoint=0,
        source_centres=np.zeros((1, 3), dtype=np.int64),
        target_centres=np.zeros((1, 3), dtype=np.int64),
        edge_matrix=np.ones((1, 1), dtype=np.float32),
    )


def test_corpus(video_store: Path):
    """The draw index space is the real pairs first, then the synthetic ones — the order a mixture assumes."""
    real, synthetic = [_target(video_store)] * 3, [_target(video_store)] * 2

    corpus = PairSplit(real, [], synthetic).corpus()

    assert len(corpus) == 5
    assert corpus[:3] == real and corpus[3:] == synthetic


def test_of_video(video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """One video's GT pairs are enumerated against ITS OWN intensity window, read from the OME metadata."""
    targets = PairSplit.of_video(video_store, in_bounds_tracks)

    assert len(targets) == 1  # the fixture annotates t=0 and t=1
    assert isinstance(targets[0].frames, ZarrFrames)
    assert targets[0].frames.q_low == 10.0 and targets[0].frames.q_high == 210.0


def test_assemble(monkeypatch: pytest.MonkeyPatch, video_store: Path, in_bounds_tracks: AnnotatedTracks):
    """One video in, its pairs out of BOTH halves — and no synthetic corpus decoded when none is asked for."""
    root = DataRoot(video_store.parent)
    monkeypatch.setattr(DataRoot, "videos", lambda self, split: [video_store])
    monkeypatch.setattr(TestMovieProxy, "training_videos", staticmethod(lambda videos, held_out=None: list(videos)))
    monkeypatch.setattr(AnnotatedTracks, "from_geff", staticmethod(lambda path: in_bounds_tracks))
    proxy = TestMovieProxy(paths=(video_store,), truths=(in_bounds_tracks,), spacing=Spacing(z=1.0, y=1.0, x=1.0))

    pairs, split = PairSplit.assemble(root, JointTrainConfig(), proxy, _LOG, val_videos=1)

    assert split.train == 1
    assert len(pairs.train) == 1 and len(pairs.val) == 1
    assert pairs.synthetic == []  # unasked for, so the corpus is never even listed


def test_of_synthetic(tmp_path: Path):
    """Unasked, nothing is decoded at all; asked at the wrong grid, the run fails BEFORE its first step."""
    root, log = DataRoot(tmp_path), _LOG

    assert PairSplit.of_synthetic(root, JointTrainConfig(), log) == []

    wrong_grid = JointTrainConfig(data=DataCfg(downsample=(1, 1, 1), synthetic_fraction=0.5))
    with pytest.raises(ValueError, match="pooled by"):
        PairSplit.of_synthetic(root, wrong_grid, log)


def test_across_acquisitions():
    """The detected corpus is drawn across acquisitions, not off the head of a sorted list.

    Taking the first N of sorted paths took 20 of 20 videos from `44b6` alone — it sorts before `6bba` — so a
    corpus built to supply CROWDED candidate sets was drawn entirely from the SPARSE acquisition, while the
    score is dominated by the dense one. The pin is that no acquisition is starved at any size, since that is
    the property the bug violated and the logs did not show.
    """
    paths = [Path(f"44b6_{index:02d}.zarr") for index in range(6)] + [
        Path(f"6bba_{index:02d}.zarr") for index in range(9)
    ]

    for size in (2, 4, 10):
        chosen = PairSplit._across_acquisitions(paths, size)
        prefixes = {path.stem.split("_")[0] for path in chosen}
        assert len(chosen) == size
        assert len(set(chosen)) == size  # no video counted twice
        assert prefixes == {"44b6", "6bba"}

    # Asking for more than one acquisition can supply still fills the request from what remains.
    assert len(PairSplit._across_acquisitions(paths, 15)) == 15


def test_of_detected():
    """The detected corpus is opt-in: at 0 videos it returns empty WITHOUT mounting a detector.

    That is the whole reason the guard comes first — mounting the two packs to then discard them would make
    every annotated-corpus run pay for a feature it did not ask for, and the mount is the expensive half.
    A `None` data root is the assertion: touching it at all would raise.
    """
    config = JointTrainConfig()

    assert config.data.detected_videos == 0  # off by default, so the corpus of yesterday is unchanged
    assert PairSplit.of_detected(cast(DataRoot, None), config, [], logging.getLogger(__name__)) == []


def test_frames(video_store: Path, tmp_path: Path):
    """Frames come from the POOLED store when one exists at the run's grid, and from the raw zarr otherwise.

    Auto-detection is the whole point — a flag would be a way to forget — so what is pinned is that the
    resolver PICKS the pooled source only when a store for THAT grid is present, and silently falls back
    when it is not. The two sources return identical voxels, so choosing wrongly would cost speed silently
    rather than fail loudly, which is exactly the kind of thing that needs a test rather than a comment.
    """
    assert isinstance(PairSplit.frames(video_store), ZarrFrames)
    assert isinstance(PairSplit.frames(video_store, tmp_path, (1, 4, 4)), ZarrFrames)  # no store written yet

    store = PooledFrames.store(tmp_path, (1, 4, 4), video_store.stem)
    store.parent.mkdir(parents=True, exist_ok=True)
    array = zarr.create_array(store=str(store), shape=(1, 1, 1, 1), chunks=(1, 1, 1, 1), dtype=np.uint16)
    array.attrs[DOWNSAMPLE_ATTR] = [1, 4, 4]

    assert isinstance(PairSplit.frames(video_store, tmp_path, (1, 4, 4)), PooledFrames)
    assert isinstance(PairSplit.frames(video_store, tmp_path, (1, 2, 2)), ZarrFrames)  # a different grid: raw
