"""Unit tests for the crowding measurement — the number that decides whether complete labelling buys anything."""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from celltrack.data.candidate_density import CandidateDensity
from celltrack.data.frame_source import FrameSource
from celltrack.data.joint_dataset import PairTarget
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)


def _graph(coordinates: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.array(coordinates, dtype=np.int64),
        edges=np.zeros((0, 2), dtype=np.int64),
    )


def test_mean():
    """One source, one target within the gate: the sparse-annotation regime, where nothing is a rival."""
    density = CandidateDensity.of([_graph([[0, 0, 0, 0], [1, 0, 0, 1]])], _ISOTROPIC, gate_um=10.0)

    assert density.mean() == pytest.approx(1.0)
    assert density.summary()["share_with_a_rival"] == 0.0


def test_summary():
    """Three targets inside the gate and one outside it — the gate, not the frame, defines the choice."""
    coordinates = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 2], [1, 0, 0, 3], [1, 0, 0, 50]]
    density = CandidateDensity.of([_graph(coordinates)], _ISOTROPIC, gate_um=10.0)

    assert density.mean() == pytest.approx(3.0)
    assert density.summary()["share_with_a_rival"] == 1.0
    assert density.summary()["candidates_per_frame"] == pytest.approx(4.0)


def test_of():
    """Anisotropy is the whole reason this converts: 5 z-voxels is 8.1 um in, 20.3 um out."""
    coordinates = [[0, 0, 0, 0], [1, 5, 0, 0]]
    spacing = Spacing(z=1.625, y=0.40625, x=0.40625)

    assert CandidateDensity.of([_graph(coordinates)], spacing, gate_um=10.0).mean() == pytest.approx(1.0)
    assert CandidateDensity.of([_graph(coordinates)], spacing, gate_um=5.0).mean() == pytest.approx(0.0)


def test_a_gap_with_no_successor_frame_contributes_nothing():
    """A frame whose successor holds no cells is not a source of zero-candidate rows — there was no choice."""
    density = CandidateDensity.of([_graph([[0, 0, 0, 0], [0, 0, 0, 1]])], _ISOTROPIC, gate_um=10.0)

    assert np.isnan(density.mean())
    assert density.summary() == {"sources": 0.0}


def test_report(caplog: pytest.LogCaptureFixture):
    """One population's row carries the mean the whole argument turns on, and the per-frame counts beside it."""
    density = CandidateDensity(
        counts=np.array([1, 3, 4, 4], dtype=np.int64), sources_per_frame=4.0, candidates_per_frame=5.0
    )

    with caplog.at_level(logging.INFO, logger="celltrack.data.candidate_density"):
        density.report("synthetic GT")

    line = caplog.text
    assert "synthetic GT" in line
    assert "3.00" in line  # the mean of 1, 3, 4, 4
    assert "5.0 candidates/frame" in line


def test_synthetic_spacing(tmp_path: Path):
    """The corpus records a POOLED voxel size while its nodes stay in the raw grid — the in-plane axes un-pool."""
    sequence = tmp_path / "seq.npz"
    np.savez(sequence, voxel_um_pooled=np.array([1.6, 1.6, 1.6], dtype=np.float64))

    spacing = CandidateDensity.synthetic_spacing(sequence)

    assert spacing.z == pytest.approx(1.6)  # z is never pooled, so it passes through
    assert spacing.y == pytest.approx(0.4)  # 1.6 / 4, the in-plane pooling undone
    assert spacing.x == pytest.approx(0.4)


def test_of_pairs():
    """The corpus form measures the rows a STEP sees, which need not be the population they came from.

    Two pairs, each with one source and two targets one of which is inside the 5um gate: the mean in-gate
    count is 1 even though the frames hold two candidates, because crowding is per SOURCE, not per frame.
    """
    pair = PairTarget(
        frames=cast(FrameSource, None),
        timepoint=0,
        source_centres=np.array([[0, 0, 0]], dtype=np.int64),
        target_centres=np.array([[0, 0, 2], [0, 0, 40]], dtype=np.int64),
        edge_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    density = CandidateDensity.of_pairs([pair, pair], _ISOTROPIC, gate_um=5.0)

    assert density.summary()["mean_in_gate"] == 1.0  # the far target is outside the gate
    assert density.summary()["sources_per_frame"] == 1.0
    assert density.summary()["candidates_per_frame"] == 2.0
