"""Unit tests for the fully-labelled pair corpus.

The load-bearing test is the coordinate one: the corpus stores POOLED volumes and RAW-grid nodes, and a
probe that missed that measured a spurious recall of 0.025. Everything else here is shape and completeness.
"""

from pathlib import Path

import numpy as np
import pytest

from celltrack.data.synthetic_pairs import POOLED_BY, SyntheticFrames, SyntheticPairs

_FRAMES, _POOLED = 3, 8
_RAW_SHAPE = (_POOLED, _POOLED * POOLED_BY[1], _POOLED * POOLED_BY[2])
_CENTRES = ((2, 12, 20), (5, 28, 4))  # raw-grid (z, y, x); pooled they are (2, 3, 5) and (5, 7, 1)


@pytest.fixture
def sequence(tmp_path: Path) -> Path:
    """A corpus-shaped sequence: two cells per frame, each a bright voxel of the POOLED volume, chained in time.

    The volume is written on the pooled grid and the nodes on the raw grid, exactly as the corpus ships them,
    so a reader that strides the volume a second time reads darkness where a cell is.
    """
    volumes = np.zeros((_FRAMES, _POOLED, _POOLED, _POOLED), dtype=np.uint16)
    nodes: list[list[float]] = []
    edges: list[list[int]] = []
    for timepoint in range(_FRAMES):
        for cell, (z, y, x) in enumerate(_CENTRES):
            volumes[timepoint, z, y // POOLED_BY[1], x // POOLED_BY[2]] = 4095
            nodes.append([timepoint, z, y, x, cell])
            if timepoint:
                edges.append([(timepoint - 1) * len(_CENTRES) + cell, timepoint * len(_CENTRES) + cell])
    path = tmp_path / "seq_0000.npz"
    np.savez(
        path,
        volumes=volumes,
        nodes=np.array(nodes, dtype=np.float32),
        edges=np.array(edges, dtype=np.int32),
        divisions=np.zeros(0, dtype=np.int64),
        voxel_um_pooled=np.array([1.625, 1.625, 1.625], dtype=np.float32),
    )
    return path


def test_frame(sequence: Path):
    """THE coordinate contract: raw centres divided by the run's downsample index the bright voxels of the frame.

    This is the exact arithmetic the trainer performs (`JointTrainer._losses` divides the full-resolution
    centres by the downsample before the detection loss), so a pooling mismatch fails here rather than as a
    quietly terrible recall number a week later.
    """
    frame = SyntheticFrames(sequence).frame(0, POOLED_BY)
    grid = np.rint(np.array(_CENTRES, dtype=np.float64) / np.array(POOLED_BY)).astype(int)

    assert tuple(frame.shape) == (_POOLED, _POOLED, _POOLED)
    for z, y, x in grid:
        assert frame[z, y, x] > 0.0
    assert float(frame.max()) == pytest.approx(frame[tuple(grid[0])].item())


def test_frame_is_never_strided_twice(sequence: Path):
    """A downsample other than the pooling already applied is refused, not silently resampled to a 2x2 slab."""
    with pytest.raises(ValueError, match="raw grid"):
        SyntheticFrames(sequence).frame(0, (1, 1, 1))


def test_frame_is_normalised_and_non_negative(sequence: Path):
    """Synthetic frames reach the backbone on the same scale competition frames do — quantile-windowed, clamped."""
    frame = SyntheticFrames(sequence).frame(0, POOLED_BY)

    assert float(frame.min()) >= 0.0
    assert float(frame.max()) > 0.0


def test_graph(sequence: Path):
    """The corpus stores edges as ROWS, so naming the rows as the ids is what makes the shared enumeration work."""
    graph = SyntheticPairs.graph(sequence)

    assert graph.coordinates.shape == (_FRAMES * len(_CENTRES), 4)
    assert graph.edge_rows().tolist() == graph.edges.tolist()
    assert set(np.diff(graph.timepoints()[graph.edge_rows()], axis=1).ravel().tolist()) == {1}


def test_enumerate(sequence: Path):
    """Every cell of the frame is a column of the edge matrix — the one property the real corpus cannot have."""
    targets = SyntheticPairs.enumerate([sequence])

    assert len(targets) == _FRAMES - 1
    for target in targets:
        assert target.source_centres.shape == (len(_CENTRES), 3)
        assert target.edge_matrix.shape == (len(_CENTRES), len(_CENTRES))
        assert target.edge_matrix.sum() == len(_CENTRES)  # every source has its true successor IN the matrix


def test_sequences(tmp_path: Path):
    """A run takes a reproducible subset of the corpus, and asking for more than exists takes all of it."""
    paths = [tmp_path / f"seq_{index:04d}.npz" for index in range(8)]
    for path in paths:
        path.touch()

    chosen = SyntheticPairs.sequences(tmp_path, 3, seed=0)

    assert len(chosen) == 3
    assert chosen == SyntheticPairs.sequences(tmp_path, 3, seed=0)
    assert SyntheticPairs.sequences(tmp_path, 99) == sorted(paths)
