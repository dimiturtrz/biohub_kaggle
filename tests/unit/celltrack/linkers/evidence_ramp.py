import numpy as np

from celltrack.linkers.evidence_ramp import EvidenceRamp, evidence_weights

ADMITTED = np.array([[True, True], [True, True]])


def test_weights():
    """The identity dose: at strength 0 every weight is the literal 1.0, so `bonus*weight*P` is the shipped product."""
    distance = np.array([[1.0, 9.0], [4.0, 6.0]])

    assert EvidenceRamp(strength=0.0).weights(distance, ADMITTED).tobytes() == np.ones_like(distance).tobytes()


def test_weights_at_unit_strength_are_proportional_to_distance():
    """At `strength=1` the ramp is `d / mean`, so the evidence weight IS the distance in units of the mean."""
    distance = np.array([[1.0, 9.0], [4.0, 6.0]])  # mean 5

    weights = EvidenceRamp(strength=1.0).weights(distance, ADMITTED)

    assert np.allclose(weights, distance / 5.0)


def test_the_mean_weight_over_admitted_pairs_is_one_at_every_strength():
    """The budget is REWEIGHTED, not raised — the sweep varies where the bonus is spent, not how much of it."""
    distance = np.array([[3.0, 9.0], [4.0, 8.0]])

    for strength in (0.0, 0.25, 0.5, 1.0):
        weights = EvidenceRamp(strength=strength).weights(distance, ADMITTED)
        assert np.isclose(float(weights.mean()), 1.0)


def test_weights_are_clamped_at_zero_so_evidence_never_becomes_a_penalty():
    """A very short candidate's weight floors at 0: the head stops helping it, it is never punished by it."""
    distance = np.array([[0.5, 9.5]])  # mean 5, so an unclamped strength-2 ramp would give -0.8

    weights = EvidenceRamp(strength=2.0).weights(distance, ADMITTED[:1])

    assert weights.tolist() == [[0.0, 2.8]]


def test_the_normalising_mean_reads_only_the_admitted_pairs():
    """The scale is the gate's own — a pair the gate excluded cannot move the weight of the ones it admitted."""
    distance = np.array([[2.0, 6.0, 100.0]])
    admitted = np.array([[True, True, False]])

    weights = EvidenceRamp(strength=1.0).weights(distance, admitted)

    assert weights[0, 0] == 0.5  # mean of the admitted pair = 4, not of all three
    assert weights[0, 1] == 1.5


def test_weights_fall_back_to_one_when_the_gate_admits_no_scale():
    """No admitted pair (or only coincident ones) leaves no scale to normalise by — the flat shipped weight."""
    distance = np.array([[0.0, 0.0]])

    assert EvidenceRamp(strength=1.0).weights(distance, np.array([[False, False]])).tolist() == [[1.0, 1.0]]
    assert EvidenceRamp(strength=1.0).weights(distance, np.array([[True, True]])).tolist() == [[1.0, 1.0]]


def test_evidence_weights_without_a_ramp_is_the_scalar_one():
    """Off is an exact float multiply, not an array of ones — one seam, so the off state cannot drift per linker."""
    assert evidence_weights(None, np.array([[1.0, 9.0]]), ADMITTED[:1]) == 1.0


def test_evidence_weights_with_a_ramp_delegates():
    """Wired, the seam is the ramp itself — nothing between the option and the weight it computes."""
    distance = np.array([[1.0, 9.0]])
    ramp = EvidenceRamp(strength=1.0)

    assert evidence_weights(ramp, distance, ADMITTED[:1]).tolist() == ramp.weights(distance, ADMITTED[:1]).tolist()
