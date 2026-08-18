"""Label-preserving frame perturbations shared by the detector-only and the joint trainers.

A crop centred on a cell teaches "fire here"; without variation the detector memorises the exact intensity
and orientation it saw. Multiplicative + additive intensity jitter breaks the brightness dependence, and a
random flip along each eligible axis multiplies the effective orientations — coordinates are mirrored with
the volume so a flipped frame stays correctly labelled. The pilkwang recipe augments and ours did not, which
is the from-scratch recall gap: warm weights recover cells our own weights miss from identical pixels.

A single frame draws its own transform (`apply`); a `(t-1, t, t+1)` pair must share ONE transform across all
of its frames (`for_pair`), because the edge matrix and the prior-velocity gap index node correspondences that
a per-frame flip would desynchronise, and independent intensity jitter would manufacture an appearance change
across `t -> t+1` that the association head would read as real motion. Defaults are the identity, so a run that
does not ask for augmentation is byte-for-byte the un-augmented baseline.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class FrameTransform:
    """One drawn augmentation — a fixed gain, bias and set of flip axes — applied identically wherever it lands."""

    gain: float
    bias: float
    flips: tuple[int, ...]

    @property
    def is_identity(self) -> bool:
        """Whether this transform changes nothing, so a caller can keep the exact input tensor it already has."""
        return self.gain == 1.0 and self.bias == 0.0 and not self.flips

    def frame(self, frame: Float[Tensor, "z y x"]) -> Float[Tensor, "z y x"]:
        """The intensity-jittered, flipped volume — non-negative, since a normalised frame's floor is zero."""
        out = (frame * self.gain + self.bias).clamp(0.0)
        for axis in self.flips:
            out = torch.flip(out, dims=(axis,))
        return out

    def coords(self, coords: Int[Tensor, "n 3"], shape: tuple[int, ...]) -> Int[Tensor, "n 3"]:
        """The centres mirrored to match `frame` — each flipped axis reflected about that axis's extent."""
        if not self.flips:
            return coords
        coords = coords.clone()
        for axis in self.flips:
            coords[:, axis] = shape[axis] - 1 - coords[:, axis]
        return coords


@dataclass(frozen=True)
class Augmentation:
    """The augmentation policy: how far to jitter intensity and which axes may flip. Identity by default."""

    brightness: float = 0.0  # multiplicative jitter half-range: gain in [1 - b, 1 + b]
    offset: float = 0.0  # additive jitter half-range: bias in [-o, o]
    flip_axes: tuple[int, ...] = ()  # frame axes (0=z, 1=y, 2=x) each flipped independently

    _FLIP_PROBABILITY = 0.5

    def for_pair(self, rng: np.random.Generator) -> FrameTransform:
        """Draw ONE transform to share across every frame of a pair — the geometry the edge matrix rides on."""
        gain = 1.0 + rng.uniform(-self.brightness, self.brightness)
        bias = rng.uniform(-self.offset, self.offset)
        flips = tuple(axis for axis in self.flip_axes if rng.random() < self._FLIP_PROBABILITY)
        return FrameTransform(gain, bias, flips)

    def apply(
        self, frame: Float[Tensor, "z y x"], coords: Int[Tensor, "n 3"], rng: np.random.Generator
    ) -> tuple[Float[Tensor, "z y x"], Int[Tensor, "n 3"]]:
        """Draw and apply a transform to one standalone frame and its centres — the single-frame detector path."""
        transform = self.for_pair(rng)
        return transform.frame(frame), transform.coords(coords, frame.shape)


_NO_AUGMENTATION = Augmentation()  # frozen identity default; a module singleton avoids a call in arg defaults
