"""Unit tests for the preprocessing layer — the strategies, and the invariance that bounds what they can do."""

import numpy as np
import pytest

from celltrack.data.normalisation import (
    NORMALISATIONS,
    GlobalWindow,
    NormalisationConfig,
    PerSliceWindow,
    QuantileWindow,
)

_QUANTILES = {"0.001": 26.0, "0.999": 2145.0}


def _frame() -> np.ndarray:
    """A frame whose z-slices are deliberately at DIFFERENT brightnesses — a stand-in for depth attenuation."""
    slices = [np.full((4, 4), level, dtype=np.float32) for level in (100.0, 400.0, 1600.0)]
    frame = np.stack(slices)
    frame[:, 2, 2] = frame[:, 0, 0] * 3.0  # one bright "cell" per slice, scaling with its slice
    return frame


def test_quantile_window_apply():
    """The reference's own expression, unchanged: window by the video's quantiles and clamp below at zero.

    Pinned against the literal formula rather than a golden number, because this default is what the MOUNTED
    weights were trained under — a drift here silently feeds published weights a distribution they never saw.
    """
    frame = _frame()
    expected = np.clip((frame - 26.0) / (2145.0 - 26.0 + 1e-6), 0.0, None)

    assert np.allclose(QuantileWindow().apply(frame, _QUANTILES), expected)
    assert QuantileWindow().apply(np.full((1, 1, 1), 5.0, dtype=np.float32), _QUANTILES).min() == 0.0  # clamped


def test_global_window_apply():
    """A global window ignores the video's own quantiles — that is exactly what keeps the scale absolute.

    Peak normalisation stretches every video to its own extremes, so a value means a different brightness in
    each. Here the same input maps to the same number whatever the video reports about itself.
    """
    frame = _frame()
    window = GlobalWindow(19.0, 1674.0)

    assert np.allclose(window.apply(frame, _QUANTILES), window.apply(frame, {"0.001": 0.0, "0.999": 9e9}))


def test_per_slice_window_apply():
    """Each z-slice is windowed by its OWN percentiles, so a depth gradient is removed rather than learned.

    The fixture's slices sit at 100/400/1600 with a proportionally brighter cell in each. Per-video windowing
    would leave the deep slice three times brighter than the shallow one; per-slice makes them comparable,
    which is the whole reason a light sheet wants this.
    """
    normalised = PerSliceWindow().apply(_frame(), _QUANTILES)
    per_slice_cell = [float(normalised[index, 2, 2]) for index in range(3)]

    assert max(per_slice_cell) - min(per_slice_cell) < 1e-3  # the gradient is gone
    assert normalised.shape == (3, 4, 4)


def test_build():
    """Every registered name builds, and an unregistered one is a construction error rather than a fallback."""
    for name in NORMALISATIONS:
        assert NormalisationConfig(name=name).build().apply(_frame(), _QUANTILES).shape == (3, 4, 4)

    with pytest.raises(ValueError, match="unknown normalisation"):
        NormalisationConfig(name="peak")


def test_build_defaults_to_the_reference_window():
    """Unasked, a run gets the expression the mounted weights were trained under — not a better idea."""
    frame = _frame()

    assert np.allclose(
        NormalisationConfig().build().apply(frame, _QUANTILES), QuantileWindow().apply(frame, _QUANTILES)
    )


def test_a_window_cannot_change_within_frame_contrast():
    """THE BOUND on this whole layer: any affine window leaves cell-vs-background separability untouched.

    d' = (mean_cell - mean_background) / sigma is invariant under x -> (x - a) / b, since numerator and
    denominator scale together. So no choice of quantiles can make cells stand out more — a window only sets
    SCALE ACROSS videos, and making cells jump out is a filter's job. This is pinned because it is the fact
    that stops the next person sweeping windows in search of contrast they cannot get here.
    """
    frame = _frame()
    cell, background = (slice(None), 2, 2), (slice(None), 0, 0)

    def separability(values: np.ndarray) -> float:
        near, far = values[cell], values[background]
        return float((near.mean() - far.mean()) / np.sqrt(0.5 * (near.var() + far.var()) + 1e-12))

    narrow = separability(QuantileWindow().apply(frame, {"0.001": 90.0, "0.999": 500.0}))
    wide = separability(QuantileWindow().apply(frame, {"0.001": 0.0, "0.999": 9000.0}))
    globally = separability(GlobalWindow(19.0, 1674.0).apply(frame, _QUANTILES))

    assert narrow == pytest.approx(wide, rel=1e-6)
    assert narrow == pytest.approx(globally, rel=1e-6)
