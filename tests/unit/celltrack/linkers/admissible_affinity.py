"""Unit tests for conditioning the parent probability on the parents the solver may pick."""

import numpy as np

from celltrack.linkers.admissible_affinity import AdmissibleAffinity


def test_conditioned():
    """Each target's column is renormalised over its in-gate sources, so admissible mass sums to one.

    The far source holds 0.6 of the column and the gate forbids it; conditioning gives the two admissible
    parents the whole column in the ratio the head scored them, 0.3 to 0.1.
    """
    probability = np.array([[0.6], [0.3], [0.1]])
    within_gate = np.array([[False], [True], [True]])

    conditioned = AdmissibleAffinity.conditioned(probability, within_gate)

    assert np.isclose(conditioned.sum(), 1.0)
    assert np.allclose(conditioned, [[0.0], [0.75], [0.25]])


def test_conditioned_leaves_a_column_with_no_admissible_parent_untouched():
    """Nothing to condition on, and the gate is about to price every pair at infinity anyway."""
    probability = np.array([[0.6], [0.4]])
    within_gate = np.array([[False], [False]])

    assert np.allclose(AdmissibleAffinity.conditioned(probability, within_gate), probability)


def test_conditioned_is_identity_when_every_parent_is_admissible():
    """A column whose mass is already entirely in-gate is unchanged — the correction is exactly the leak."""
    probability = np.array([[0.7], [0.3]])
    within_gate = np.array([[True], [True]])

    assert np.allclose(AdmissibleAffinity.conditioned(probability, within_gate), probability)


def test_applied():
    """Off returns the very same array, so a linker that does not ask prices exactly what it priced before."""
    probability = np.array([[0.6], [0.4]])
    within_gate = np.array([[False], [True]])

    assert AdmissibleAffinity.applied(False, probability, within_gate) is probability
    assert np.allclose(AdmissibleAffinity.applied(True, probability, within_gate), [[0.0], [1.0]])
