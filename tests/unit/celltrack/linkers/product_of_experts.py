import numpy as np

from celltrack.linkers.product_of_experts import MotionDiffusionPrior, ProductOfExperts


def _entropy(logits: np.ndarray) -> float:
    """Shannon entropy (nats) of the per-source softmax over targets — the ambiguity the fusion leaves."""
    probabilities = np.exp(logits - logits.max())
    probabilities = probabilities / probabilities.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def test_log_likelihood():
    """The Gaussian expert is `-d^2 / 2sigma^2`: zero at the prediction, more negative as the residual grows."""
    prior = MotionDiffusionPrior(sigma_um=2.0)
    distances = np.array([[0.0, 2.0, 4.0]])

    likelihood = prior.log_likelihood(distances)

    np.testing.assert_allclose(likelihood, -(distances**2) / (2.0 * 2.0**2))
    assert likelihood[0, 0] == 0.0  # a target exactly at the prediction pays nothing
    assert likelihood[0, 2] < likelihood[0, 1] < 0.0  # farther residual, lower log-likelihood


def test_from_one_step_msd():
    """sigma is derived as sqrt(MSD at lag 1) — the fitted one-step spread, no tuned constant."""
    prior = MotionDiffusionPrior.from_one_step_msd(9.0)

    assert prior.sigma_um == 3.0  # sqrt(9)
    # a residual one sigma out costs exactly -1/2 (the Gaussian half-width), tying width to the fit
    np.testing.assert_allclose(prior.log_likelihood(np.array([[3.0]])), -0.5)


def test_as_log_probability():
    """Appearance turns to log-space, with a zero floored to a finite cost rather than -inf (priced by motion)."""
    poe = ProductOfExperts()
    probabilities = np.array([[1.0, 0.5, 0.0]])

    log_probabilities = poe.as_log_probability(probabilities)

    assert log_probabilities[0, 0] == 0.0  # P=1 -> log 0
    np.testing.assert_allclose(log_probabilities[0, 1], np.log(0.5))
    assert np.isfinite(log_probabilities[0, 2])  # P=0 floored, not -inf
    assert log_probabilities[0, 2] < log_probabilities[0, 1]


def test_log_fused():
    """The product of experts is the SUM of the two log-probabilities, pair for pair."""
    poe = ProductOfExperts()
    log_appearance = np.array([[-0.1, -1.2, -3.0]])
    log_motion = np.array([[-0.5, -0.2, -4.0]])

    fused = poe.log_fused(log_appearance, log_motion)

    np.testing.assert_allclose(fused, log_appearance + log_motion)


def test_cost():
    """Cost is `-log P_fused`; agreeing experts sharpen the choice, flat ones leave it soft (the mechanism)."""
    poe = ProductOfExperts()
    prior = MotionDiffusionPrior(sigma_um=2.0)

    # AGREEMENT: appearance and motion both favour target 0 -> the fused cost is minimal there AND the fused
    # distribution is sharper (lower entropy) than either expert alone.
    log_appearance = poe.as_log_probability(np.array([[0.7, 0.2, 0.1]]))
    log_motion = prior.log_likelihood(np.array([[0.5, 3.0, 5.0]]))
    cost = poe.cost(log_appearance, log_motion)

    np.testing.assert_allclose(cost, -(log_appearance + log_motion))  # exactness
    assert int(cost.argmin()) == 0  # both experts point at target 0
    fused_entropy = _entropy(-cost)
    assert fused_entropy < _entropy(log_appearance)  # product is sharper than appearance alone
    assert fused_entropy < _entropy(log_motion)  # ... and than motion alone

    # CONFLICT + FLAT: two near-uniform experts -> the product stays near-uniform, softness preserved.
    flat_appearance = poe.as_log_probability(np.array([[0.34, 0.33, 0.33]]))
    flat_motion = prior.log_likelihood(np.array([[1.0, 1.0, 1.0]]))
    flat_fused = poe.log_fused(flat_appearance, flat_motion)
    assert _entropy(flat_fused) > 0.99 * np.log(3)  # within 1% of maximum ambiguity
