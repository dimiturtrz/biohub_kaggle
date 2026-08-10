import numpy as np

from celltrack.linkers.agreement_gating import AgreementGate


class _Mutual:
    """The bidirectionally fused probabilities for one gap — the `EdgeAffinity` shape the gate consumes."""

    def __init__(self, by_timepoint: dict[int, np.ndarray]) -> None:
        self._by_timepoint = by_timepoint

    def probabilities(self, timepoint: int) -> np.ndarray | None:
        return self._by_timepoint.get(timepoint)


def test_narrow():
    """A pair survives only where it is already in-gate AND its fused probability clears the floor."""
    gate = AgreementGate(mutual=_Mutual({0: np.array([[0.9, 0.2], [0.4, 0.7]])}), floor=0.5)
    within_gate = np.array([[True, True], [False, True]])

    assert gate.narrow(0, within_gate).tolist() == [[True, False], [False, True]]


def test_narrow_leaves_an_unscored_gap_to_geometry():
    """A gap the fused affinity does not score is passed through untouched — the cost's own unscored rule."""
    gate = AgreementGate(mutual=_Mutual({}), floor=0.99)
    within_gate = np.array([[True, False]])

    assert gate.narrow(3, within_gate).tolist() == within_gate.tolist()
