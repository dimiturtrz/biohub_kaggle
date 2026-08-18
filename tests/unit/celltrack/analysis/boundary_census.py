"""Unit tests for the track-boundary census — births and deaths classified against the imaged volume."""

import logging

import numpy as np
import pytest

from celltrack.analysis.boundary_census import MovieBoundaries
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_SPACING = Spacing(z=1.0, y=1.0, x=1.0)
# Thick on every axis so an interior cell (50 um from each face) is beyond any plausible gate and a face cell
# (2 um in) is inside it — the classification then does not ride on the shipped gate's exact value.
_VOLUME = (100, 100, 100)


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph over consecutively-numbered node ids at the given `(t, z, y, x)` coordinates."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_of():
    """Births split by what explains them, and division daughters and dividing mothers are neither birth nor death."""
    births = _graph(
        [[0, 50, 50, 50], [1, 50, 50, 50], [1, 2, 50, 50], [1, 50, 50, 50]],
        [[0, 1]],  # only the first-frame birth has a child; the other two are lone parentless nodes
    )
    census = MovieBoundaries.of("m", births, _SPACING, _VOLUME)
    assert census.births == 3  # the three in_degree-0 nodes; the child (n1) is not a birth
    assert census.birth_explained == 2 / 3  # first-frame + face; the interior one is charged full
    assert census.birth_first_frame == 1 / 3  # only the t=0 node
    assert census.birth_interior_depth == (49.0 - 10.0) / 10.0  # sole interior birth at (50,50,50): 49 um from a face

    fork = _graph([[0, 50, 50, 50], [1, 50, 50, 50], [1, 51, 50, 50]], [[0, 1], [0, 2]])
    divided = MovieBoundaries.of("m", fork, _SPACING, _VOLUME)
    assert divided.births == 1  # the mother (in_degree 0); daughters have a parent
    assert divided.deaths == 2  # both daughters (out_degree 0); the mother has children
    assert divided.death_last_frame == 1.0  # both daughters sit in the last frame


def test_report(caplog: pytest.LogCaptureFixture):
    """The report logs both boundary sets with their explained shares — stake first."""
    graph = _graph([[0, 50, 50, 50], [1, 50, 50, 50]], [[0, 1]])
    census = MovieBoundaries.of("movie-x", graph, _SPACING, _VOLUME)
    with caplog.at_level(logging.INFO, logger="celltrack.analysis.boundary_census"):
        census.report()
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "movie-x" in logged
    assert "births" in logged and "deaths" in logged


def test_band_fraction():
    """The discounted shell is a union over faces, and a volume thinner than two margins is discounted whole."""
    band = MovieBoundaries.band_fraction(_SPACING, 10.0, _VOLUME)  # 99 um cube, 10 um margin each side
    assert band == pytest.approx(1.0 - (79.0 / 99.0) ** 3)
    thin = MovieBoundaries.band_fraction(_SPACING, 10.0, (10, 100, 100))  # 9 um z < 2*margin -> no interior in z
    assert thin == 1.0


def test_share_of_an_empty_set_is_nan_not_zero():
    """A share over an empty set is undefined, not 0 — an absent boundary class must read so, not as fully-charged."""
    assert np.isnan(MovieBoundaries._share(np.zeros(0, dtype=bool)))
    assert MovieBoundaries._share(np.array([True, False])) == 0.5
