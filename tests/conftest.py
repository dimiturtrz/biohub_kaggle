"""Synthetic OME-Zarr and GEFF stores, so the unit suite never touches the real dataset."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from core.data.tracks import AnnotatedTracks, TrackGraph

SPACING = (1.0, 1.625, 0.40625, 0.40625)
QUANTILES = {"0.001": 10.0, "0.999": 210.0}
SHAPE = (3, 2, 4, 4)


def _write_video(store: Path) -> Path:
    """Create one tiny OME-Zarr with the metadata layout the competition ships."""
    group = zarr.open_group(store, mode="w")
    frames = group.create_array("0", shape=SHAPE, dtype="uint16", chunks=(1, *SHAPE[1:]))
    frames[:] = np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)
    group.attrs["multiscales"] = [
        {"datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": list(SPACING)}]}]},
    ]
    group.attrs["image_statistics"] = {"quantiles": QUANTILES}
    return store


def _write_geff(store: Path) -> Path:
    """Create one GEFF store: node 100 at t=0 splits into 200 and 300 at t=1."""
    group = zarr.open_group(store, mode="w")
    group.create_array("nodes/ids", shape=(3,), dtype="int64")[:] = [100, 200, 300]
    for axis, values in (("t", [0, 1, 1]), ("z", [0, 0, 1]), ("y", [0, 4, 0]), ("x", [0, 0, 4])):
        group.create_array(f"nodes/props/{axis}/values", shape=(3,), dtype="int64")[:] = values
    group.create_array("edges/ids", shape=(2, 2), dtype="int64")[:] = [[100, 200], [100, 300]]
    group.attrs["geff"] = {"extra": {"estimated_number_of_nodes": 30}}
    return store


@pytest.fixture
def video_store(tmp_path: Path) -> Path:
    """A tiny `(T, Z, Y, X)` OME-Zarr with the same metadata layout the competition ships."""
    return _write_video(tmp_path / "aaaa_00000001.zarr")


@pytest.fixture
def geff_store(tmp_path: Path) -> Path:
    """A GEFF store holding one dividing lineage: node 100 at t=0 splits into 200 and 300 at t=1."""
    return _write_geff(tmp_path / "aaaa_00000001.geff")


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A minimal data root: one video of each acquisition in train (with GEFF) and test, laid out for DataRoot."""
    competition = tmp_path / "raw" / "biohub_cell_tracking"
    train, test = competition / "train", competition / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    for prefix in ("44b6", "6bba"):
        _write_video(train / f"{prefix}_0000abcd.zarr")
        _write_geff(train / f"{prefix}_0000abcd.geff")
        _write_video(test / f"{prefix}_0000ef01.zarr")
    return tmp_path


@pytest.fixture
def in_bounds_tracks() -> AnnotatedTracks:
    """A GEFF-shaped annotation whose cells sit inside the tiny `video_store` volume (Z=2, Y=4, X=4)."""
    graph = TrackGraph(
        node_ids=np.array([1, 2]),
        coordinates=np.array([[0, 1, 2, 2], [1, 1, 2, 2]]),
        edges=np.array([[1, 2]]),
    )
    return AnnotatedTracks(graph=graph, estimated_node_count=20)


@pytest.fixture
def graph() -> TrackGraph:
    """A two-timepoint chain: node 7 at the origin links to node 9 four voxels away in y."""
    return TrackGraph(
        node_ids=np.array([7, 9]),
        coordinates=np.array([[0, 0, 0, 0], [1, 0, 4, 0]]),
        edges=np.array([[7, 9]]),
    )
