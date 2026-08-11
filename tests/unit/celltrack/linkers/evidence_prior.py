"""Unit tests for pricing a birth by the head's own belief that the detection already has a parent."""

import numpy as np
from jaxtyping import Float

from celltrack.linkers.evidence_prior import EvidencePrior
from core.geometry import Spacing

_ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)
_GATE_UM = 10.0


class FakeAffinity:
    """An `EdgeAffinity` from an explicit per-gap matrix, rows the t sources, columns the t+1 targets."""

    def __init__(self, by_timepoint: dict[int, Float[np.ndarray, "s t"]]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> Float[np.ndarray, "s t"] | None:
        return self._by_timepoint.get(timepoint)


def _positions(rows: list[float]) -> Float[np.ndarray, "n 3"]:
    """Detections on one spatial axis — a single axis is enough to place them for or outside a gate."""
    return np.array([[0.0, 0.0, x] for x in rows])


def test_factors():
    """A target whose parent probability sits OUTSIDE the gate is cheap to appear; one with a parent is dear.

    Two targets share a frame. The first sits beside a source holding 0.9 of its column; the second sits far
    from every source, so no admissible parent holds any of its mass. Normalised to mean 1 within the frame,
    the well-parented target pays 2x the flat charge to be a birth and the orphan pays nothing.
    """
    positions = _positions([0.0, 100.0, 1.0, 50.0])  # sources at 0 and 100; one target beside 0, one adrift
    timepoints = np.array([0, 0, 1, 1])
    affinity = FakeAffinity({0: np.array([[0.9, 0.5], [0.1, 0.5]])})

    factors = EvidencePrior(affinity=affinity).factors(_ISOTROPIC, _GATE_UM, positions, timepoints)

    assert np.allclose(factors.appearance[[2, 3]], [2.0, 0.0])
    assert np.allclose(factors.disappearance, 1.0)  # a source-normalised softmax says nothing about children


def test_factors_leaves_the_first_frame_flat():
    """The first observed frame has no incoming gap, so there is no evidence and the flat charge stands."""
    positions = _positions([0.0, 1.0])
    timepoints = np.array([0, 1])
    affinity = FakeAffinity({0: np.array([[0.4]])})

    factors = EvidencePrior(affinity=affinity).factors(_ISOTROPIC, _GATE_UM, positions, timepoints)

    assert factors.appearance[0] == 1.0


def test_factors_preserves_the_appearance_budget():
    """Factors mean 1 WITHIN each frame, so an arm moves where the charge falls, never how much there is.

    That is what makes the flat cost its own control: without it, a sweep would be confounded with simply
    making every birth cheaper.
    """
    positions = _positions([0.0, 2.0, 4.0, 1.0, 3.0, 5.0])
    timepoints = np.array([0, 0, 0, 1, 1, 1])
    affinity = FakeAffinity({0: np.array([[0.5, 0.1, 0.0], [0.2, 0.6, 0.1], [0.1, 0.1, 0.7]])})

    factors = EvidencePrior(affinity=affinity).factors(_ISOTROPIC, _GATE_UM, positions, timepoints)

    assert np.isclose(factors.appearance[timepoints == 1].mean(), 1.0)


def test_factors_is_flat_where_the_scorer_scored_nothing():
    """A gap with no learned matrix leaves its targets on the flat charge rather than a fabricated one."""
    positions = _positions([0.0, 1.0, 2.0])
    timepoints = np.array([0, 1, 2])

    factors = EvidencePrior(affinity=FakeAffinity({})).factors(_ISOTROPIC, _GATE_UM, positions, timepoints)

    assert np.allclose(factors.appearance, 1.0)
