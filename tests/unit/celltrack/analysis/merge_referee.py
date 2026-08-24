"""Equivalence-class tests for the GT-free merge referee.

Pure-compute only: `analyse` takes loaded logit volumes (no disk, no card), so these build synthetic
(T, Z, Y, X) volumes with known blob layouts and check the basin/peak/band readout. Classes: an isolated
single blob (one component, single band), two separated blobs (two components), and an empty field.
"""

from __future__ import annotations

import numpy as np

from celltrack.analysis.merge_referee import MergeReferee

_HI, _LO = 6.0, -6.0  # sigmoid(6) ~ 0.998 (above floor), sigmoid(-6) ~ 0.002 (below)
_SHAPE = (1, 3, 24, 24)  # T, Z, Y, X


def _field(centres: list[tuple[int, int, int]], radius: int = 1) -> np.ndarray:
    """A logit volume that is _HI within `radius` (Chebyshev) of each (z, y, x) centre, else _LO."""
    vol = np.full(_SHAPE, _LO, dtype=np.float64)
    for z, y, x in centres:
        vol[
            0,
            max(0, z - radius) : z + radius + 1,
            max(0, y - radius) : y + radius + 1,
            max(0, x - radius) : x + radius + 1,
        ] = _HI
    return vol


def _both_grids(centres: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    # same physical field seen by both decode grids (the test doesn't model resolution, only the readout)
    field = _field(centres)
    return {"ds1x4x4": field, "ds1x2x2": field}


def test_analyse() -> None:  # single isolated blob -> one component in the single band

    out = MergeReferee.analyse(_both_grids([(1, 12, 12)]), floor=0.5)
    assert out["ds1x4x4"].n_components == 1
    assert out["ds1x2x2"].n_components == 1
    # a lone blob sized at V0 lands in the single band, not fragment or merged
    assert out["ds1x4x4"].single.n == 1
    assert out["ds1x4x4"].merged.n == 0


def test_analyse_two_separated_blobs_are_two_components() -> None:
    out = MergeReferee.analyse(_both_grids([(1, 6, 6), (1, 18, 18)]), floor=0.5)
    assert out["ds1x4x4"].n_components == 2
    assert out["ds1x2x2"].peaks_native == 2


def test_analyse_empty_field_has_no_components() -> None:
    out = MergeReferee.analyse(_both_grids([]), floor=0.5)
    assert out["ds1x4x4"].n_components == 0
    assert out["ds1x4x4"].peaks_native == 0
