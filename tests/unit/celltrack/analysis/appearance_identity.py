"""Unit tests for the full-resolution appearance-identity gate — pinned on constructed appearance.

The module's claims are about whether raw boxes can separate cells, so the fixtures build boxes whose answer
is known: a box that IS another cell, a box that is a rotated copy, a triple whose boundary members drop. A
test that fed it real data would not be a unit test and could not pin the arithmetic.
"""

import numpy as np
import pytest

from celltrack.analysis.appearance_identity import (
    IdentitySimilarity,
    InGateRanking,
    RotationGain,
)
from celltrack.data.raw_boxes import RawBoxes
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)


def _blob(shape: tuple[int, int, int], centre: tuple[int, int, int], sigma: float = 2.0) -> np.ndarray:
    """A single Gaussian blob — a stand-in for one cell's intensity, with a definite centre to correlate on."""
    grids = np.meshgrid(*[np.arange(dim) for dim in shape], indexing="ij")
    squared = sum((axis - c) ** 2 for axis, c in zip(grids, centre, strict=True))
    return np.exp(-squared / (2 * sigma**2)).astype(np.float32)


def test_correlate():
    """A box correlates 1.0 with itself, is blind to a brightness offset, and drops when it is shifted."""
    box = _blob((5, 9, 9), (2, 4, 4))

    assert IdentitySimilarity.correlate(box, box) == pytest.approx(1.0)
    assert IdentitySimilarity.correlate(box, box + 100.0) == pytest.approx(1.0)  # mean removed → offset-blind
    assert IdentitySimilarity.correlate(box, _blob((5, 9, 9), (2, 4, 7))) < 0.9  # a shifted cell is less alike


def test_raw_correlation_scorer():
    """The default scorer ranks each candidate by correlation — the source's own copy scores highest."""
    source = _blob((5, 9, 9), (2, 4, 4))
    candidates = np.stack([source, _blob((5, 9, 9), (2, 4, 7)), _blob((5, 9, 9), (2, 4, 8))])

    scores = IdentitySimilarity.raw_correlation_scorer(source, candidates)

    assert scores.shape == (3,)
    assert int(np.argmax(scores)) == 0  # the identical box is the most similar


def test_identity_similarity_of():
    """When a boundary box drops from one column, the metric compares only complete triples, not shifted rows.

    A source whose true box was dropped must not be silently paired with a different triple's rival — the
    intersection of surviving requests is what prevents that, and it is the whole reason `kept` is carried.
    """
    source = RawBoxes(boxes=np.stack([_blob((5, 9, 9), (2, 4, 4)), _blob((5, 9, 9), (2, 4, 4))]), kept=np.array([0, 1]))
    true_target = RawBoxes(boxes=_blob((5, 9, 9), (2, 4, 4))[None], kept=np.array([0]))  # triple 1's true box dropped
    rival = RawBoxes(boxes=np.stack([_blob((5, 9, 9), (2, 4, 7)), _blob((5, 9, 9), (2, 4, 7))]), kept=np.array([0, 1]))

    similarity = IdentitySimilarity.of(source, true_target, rival)

    assert similarity.count() == 1  # only triple 0 had all three boxes
    assert similarity.prefers_true() == pytest.approx(1.0)  # its source matches its own true box over the shifted rival


def test_prefers_true():
    """The share of triples whose source matches the true box better than the rival — the two-way choice."""
    identical = _blob((5, 9, 9), (2, 4, 4))
    source = RawBoxes(boxes=identical[None], kept=np.array([0]))
    true_target = RawBoxes(boxes=identical[None], kept=np.array([0]))
    rival = RawBoxes(boxes=_blob((5, 9, 9), (2, 7, 7))[None], kept=np.array([0]))

    assert IdentitySimilarity.of(source, true_target, rival).prefers_true() == pytest.approx(1.0)


def test_rotation_gain_of():
    """A box rotated from its partner is recovered by the rotation search, and an unrotated pair gains ~nothing.

    The gain is exactly the in-plane turn a fixed-orientation correlation is blind to — near zero when the two
    boxes already share an orientation, positive when one is turned. An elongated blob makes the rotation
    visible; a symmetric one would gain nothing from any angle, which is why the fixture is off-centre.
    """
    # A bar: sharp across y, broad along x, so an in-plane turn genuinely changes which voxels are lit.
    axis = np.arange(15)
    across = np.exp(-((axis - 7) ** 2) / (2 * 1.2**2))
    along = np.exp(-((axis - 7) ** 2) / (2 * 5.0**2))
    bar = (across[:, None] * along[None, :])[None].repeat(5, axis=0).astype(np.float32)
    from scipy.ndimage import rotate  # noqa: PLC0415 — test-local, mirrors the module's own rotation

    turned = rotate(bar, 30.0, axes=(1, 2), reshape=False, order=1, mode="nearest")
    source = RawBoxes(boxes=np.stack([bar, bar]), kept=np.array([0, 1]))
    partner = RawBoxes(boxes=np.stack([bar, turned]), kept=np.array([0, 1]))

    gain = RotationGain.of(source, partner, angles_deg=(0.0, -30.0, -15.0, 15.0, 30.0))

    assert gain.count() == 2
    assert gain.best_rotated[0] == pytest.approx(gain.identity[0], abs=1e-6)  # already aligned → no gain
    assert gain.identity[1] < 0.9  # a 30-degree turn of a bar drops the fixed-orientation correlation
    assert gain.best_rotated[1] > gain.identity[1] + 0.05  # rotating it back recovers the match


def test_chance():
    """Chance is the mean of 1/candidates, not 0.5 — the correction that makes the ranking honest.

    Two sources, one facing two candidates and one facing four, give a chance of (1/2 + 1/4) / 2 = 0.375. A
    flat 0.5 would understate how easy the two-candidate source is and overstate the four-candidate one.
    """
    ranking = InGateRanking(true_rank=np.array([1, 2]), candidates=np.array([2, 4]))

    assert ranking.chance() == pytest.approx((0.5 + 0.25) / 2)
    assert ranking.top_one() == pytest.approx(0.5)  # the truth was first in one of the two


def test_top_one():
    """Top-1 is the share ranked first — the linker's actual question, put the right candidate first."""
    assert InGateRanking(true_rank=np.array([1, 1, 3]), candidates=np.array([4, 4, 4])).top_one() == pytest.approx(
        2 / 3
    )


def test_in_gate_ranking_of():
    """A source with a single in-gate candidate poses no ranking question and is left out, not scored as a win.

    Counting it as a trivial top-1 would inflate the score with cases the linker never had to decide, so the
    measurement runs only where there is a real choice.
    """
    stub = _RankingStub(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 50.0]]),
        timepoints=np.array([0, 1, 1]),
    )
    from celltrack.analysis.localisation import CandidatePairs  # noqa: PLC0415 — test-local import of the pair type

    pairs = CandidatePairs(source=np.array([0]), true_target=np.array([1]), rival=np.array([2]))

    def _unused_reader(_rows: np.ndarray) -> RawBoxes:  # must never run: nothing to rank
        raise AssertionError("box reader ran on a source with no real choice")

    ranking = InGateRanking.of(stub.prediction, stub.spacing, _unused_reader, pairs, gate_um=10.0)

    assert ranking.count() == 0  # only one candidate within 10 um, so nothing to rank


class _RankingStub:
    """The minimal surface `InGateRanking.of` reads — a prediction graph and a spacing, no video access."""

    def __init__(self, positions: np.ndarray, timepoints: np.ndarray) -> None:
        self.spacing = _SPACING
        self.prediction = TrackGraph(
            node_ids=np.arange(len(positions), dtype=np.int64),
            coordinates=np.column_stack([timepoints, positions]).astype(np.int64),
            edges=np.empty((0, 2), dtype=np.int64),
        )


def test_median_gain():
    """Median rotation gain is near zero for already-aligned pairs and positive when a partner is turned."""
    axis = np.arange(15)
    bar = np.exp(-((axis - 7) ** 2) / (2 * 1.2**2))[:, None] * np.exp(-((axis - 7) ** 2) / (2 * 5.0**2))[None, :]
    bar = bar[None].repeat(5, axis=0).astype(np.float32)
    from scipy.ndimage import rotate  # noqa: PLC0415 — test-local, mirrors the module's own rotation

    aligned = RotationGain.of(RawBoxes(bar[None], np.array([0])), RawBoxes(bar[None], np.array([0])), (0.0, 30.0))
    turned = rotate(bar, 30.0, axes=(1, 2), reshape=False, order=1, mode="nearest")
    rotated = RotationGain.of(RawBoxes(bar[None], np.array([0])), RawBoxes(turned[None], np.array([0])), (0.0, -30.0))

    assert aligned.median_gain() == pytest.approx(0.0, abs=1e-6)
    assert rotated.median_gain() > 0.05


def test_identity_similarity_count():
    """Count is the number of complete triples the metric ran on."""
    box = _blob((5, 9, 9), (2, 4, 4))
    similarity = IdentitySimilarity.of(
        RawBoxes(box[None], np.array([0])), RawBoxes(box[None], np.array([0])), RawBoxes(box[None], np.array([0]))
    )
    assert similarity.count() == 1


def test_rotation_gain_count():
    """Count is the number of same-cell pairs the rotation probe ran on."""
    box = _blob((5, 9, 9), (2, 4, 4))
    gain = RotationGain.of(RawBoxes(box[None], np.array([0])), RawBoxes(box[None], np.array([0])), (0.0,))
    assert gain.count() == 1


def test_in_gate_ranking_count():
    """Count is the number of contested sources ranked."""
    assert InGateRanking(true_rank=np.array([1, 2]), candidates=np.array([3, 4])).count() == 2
