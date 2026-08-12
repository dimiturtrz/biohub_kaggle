"""How raw intensities become model input — the preprocessing layer, as strategies rather than a formula.

The normalisation was an expression inlined into every frame reader, copied from the reference. That is fine
while there is exactly one, and wrong as soon as there is a question about it: the choice is a real design
axis with measured trade-offs, so it belongs behind an interface a run can select, not in two places that
must be kept in sync by hand.

WHAT THE CHOICE CAN AND CANNOT DO, measured, because it bounds what any strategy here is for. d-prime
between cell centres and background is INVARIANT under an affine map, so no window — per video, global, any
quantile pair — can change contrast WITHIN a frame. Measured across three very different windows on the same
videos, d-prime moves by at most 5%, and that residual is the one-sided clamp rather than the window. A
window therefore only sets SCALE ACROSS videos; making cells stand out is a filter's job, not a window's.

THE SHIPPED DEFAULT IS PEAK NORMALISATION, and its cost is real: stretching each video to its own quantiles
means 0.5 denotes a different physical brightness in every video (measured: the same annotated cell centre
lands anywhere in 0.32-0.85). That is the audio critique of peak normalisation — low and high stop meaning
anything absolute. `GlobalWindow` is the alternative that keeps the scale, and the two are INDISTINGUISHABLE
by d-prime (3.37 and CV 0.408 for both), because both are affine: the criterion is blind to exactly the
property that separates them. That question is open, not settled, and the interface exists so it can be
asked.
"""

from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, field_validator

# The organisers ship these eight; a strategy names the two it wants rather than hard-coding a formula.
LOW_QUANTILE = "0.001"
HIGH_QUANTILE = "0.999"
_EPSILON = 1e-6  # the reference's own guard against a degenerate window; kept so the default stays faithful


@runtime_checkable
class Normalisation(Protocol):
    """One frame's raw voxels to model input, given the video's own shipped intensity quantiles."""

    def apply(self, frame: Float[np.ndarray, "z y x"], quantiles: Mapping[str, float]) -> Float[np.ndarray, "z y x"]:
        """The normalised frame — same shape, float32."""
        ...


class QuantileWindow:
    """The reference's own normalisation: window by the video's quantiles, clamp below at zero.

    THE DEFAULT, and deliberately so: the shipped pipeline MOUNTS pilkwang's weights, which were trained
    under exactly this expression, so anything else feeds them a distribution they never saw. Copied line
    for line from both of their scripts, epsilon and one-sided clamp included — the clamp is the only
    nonlinearity here, and leaving the top unbounded is theirs, not an oversight (measured: 0.00-0.06% of
    voxels exceed 1.0).

    It is PEAK normalisation, with the cost that implies — each video is stretched to its own extremes, so a
    value carries no absolute meaning across videos. `GlobalWindow` is the alternative that keeps one.
    """

    def __init__(self, low: str = LOW_QUANTILE, high: str = HIGH_QUANTILE) -> None:
        self._low, self._high = low, high

    def apply(self, frame: Float[np.ndarray, "z y x"], quantiles: Mapping[str, float]) -> Float[np.ndarray, "z y x"]:
        """Window by this video's own quantiles — the expression the mounted weights were trained under."""
        low, high = float(quantiles[self._low]), float(quantiles[self._high])
        return np.clip((frame.astype(np.float32) - low) / (high - low + _EPSILON), 0.0, None)


class GlobalWindow:
    """One window for every video, so a value means the same brightness everywhere — scale, not peak.

    Peak normalisation removes absolute scale: stretching each video to its own quantiles makes 0.5 denote a
    different brightness per video, and the same annotated cell centre was measured landing anywhere in
    0.32-0.85. A single corpus-wide window keeps that meaning, at the price of letting genuinely dim videos
    stay dim — which is information rather than noise.

    NOT MEASURABLY WORSE, and not measurably better: against per-video windowing it scores an IDENTICAL
    d-prime of 3.37 and CV 0.408, because both maps are affine and the criterion cannot see the difference.
    Only a trained model can answer it, which is why this is an option rather than a recommendation.

    The bounds are the corpus MEDIAN of the per-video quantiles, so the default window is the middle video's
    rather than an invented constant.
    """

    def __init__(self, low: float, high: float) -> None:
        self._low, self._high = low, high

    def apply(self, frame: Float[np.ndarray, "z y x"], quantiles: Mapping[str, float]) -> Float[np.ndarray, "z y x"]:
        """Window by the CORPUS bounds, ignoring this video's own — that is the point of it."""
        return np.clip((frame.astype(np.float32) - self._low) / (self._high - self._low + _EPSILON), 0.0, None)


class PerSliceWindow:
    """A window per Z-SLICE, correcting the depth gradient a light sheet actually has.

    Illumination attenuates and scatters with depth, so intensity carries a real z-gradient that a per-video
    window cannot see. Correcting it per slice is published practice for this modality (Intensify3D does
    per-plane quantile matching for exactly this reason), and it is the only variant measured to improve BOTH
    of the things that matter: cell-vs-background separability 2.90 -> 3.28, and its consistency across
    videos 0.436 -> 0.378.

    It is NOT affine over the frame — different slices get different maps — which is why it can move
    separability at all where a window cannot. The quantiles are computed from the slice itself, so the
    shipped per-video statistics are unused and a store needs nothing extra.
    """

    def __init__(self, low_percentile: float = 0.1, high_percentile: float = 99.9) -> None:
        self._low, self._high = low_percentile, high_percentile

    def apply(self, frame: Float[np.ndarray, "z y x"], quantiles: Mapping[str, float]) -> Float[np.ndarray, "z y x"]:
        """Window each z-slice by its own percentiles, so depth attenuation is removed rather than learned."""
        planes = frame.astype(np.float32).reshape(frame.shape[0], -1)
        low = np.percentile(planes, self._low, axis=1)[:, None, None]
        high = np.percentile(planes, self._high, axis=1)[:, None, None]
        return np.clip((frame.astype(np.float32) - low) / (high - low + _EPSILON), 0.0, None)


class NormalisationConfig(BaseModel):
    """Which preprocessing a run uses, validated — the selectable half of the strategies above.

    Defaults to the reference's own window, so a run that does not ask sees the numbers every arm before it
    saw, and the MOUNTED weights keep the distribution they were trained under. Anything else is an
    own-weights decision by construction: pilkwang's weights cannot be fed a normalisation they never saw.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "quantile"
    # Only `global` reads these, and they DEFAULT to the corpus median of the per-video quantiles (19 / 1674
    # over a stratified sample) rather than to an invented pair — the middle video's window, not a guess.
    global_low: float = 19.0
    global_high: float = 1674.0
    # Only `per_slice` reads these; they mirror the shipped 0.001/0.999 as percentages of the slice.
    slice_low_percentile: float = Field(0.1, ge=0.0, lt=100.0)
    slice_high_percentile: float = Field(99.9, gt=0.0, le=100.0)

    @field_validator("name")
    @classmethod
    def _known(cls, name: str) -> str:
        """A misspelled strategy is a construction error, not a silent fallback to the default."""
        if name not in NORMALISATIONS:
            raise ValueError(f"unknown normalisation {name!r}, expected one of {sorted(NORMALISATIONS)}")
        return name

    def build(self) -> Normalisation:
        """The strategy this config names, ready to apply to a frame."""
        return NORMALISATIONS[self.name](self)


# Declaration-only dispatch: one row per strategy, so adding one cannot mean editing a chain of ifs.
NORMALISATIONS: dict[str, "Callable[[NormalisationConfig], Normalisation]"] = {
    "quantile": lambda config: QuantileWindow(),
    "global": lambda config: GlobalWindow(config.global_low, config.global_high),
    "per_slice": lambda config: PerSliceWindow(config.slice_low_percentile, config.slice_high_percentile),
}
