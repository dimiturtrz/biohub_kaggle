"""Unit tests for label-preserving augmentation — the single-frame draw and the pair-shared transform.

The maths a flip must satisfy is one equality: a centre mirrors to `extent - 1 - centre` along the flipped
axis, and nothing moves along an unflipped one. The pair path adds the property the joint trainer turns on —
ONE transform reused across every frame, so a flip that moves `t` moves `t+1` and `t-1` the identical way.
"""

import numpy as np
import torch

from celltrack.data.augmentation import Augmentation, FrameTransform


def test_apply():
    """A flip mirrors both the frame and its centres along the axis; identity draws leave both untouched."""
    aug = Augmentation(flip_axes=(1,))
    frame = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    coords = torch.tensor([[0, 0, 2]])
    saw_flip = saw_identity = False
    for seed in range(8):
        out, out_coords = aug.apply(frame, coords, np.random.default_rng(seed))
        if torch.equal(out, frame):
            assert out_coords.tolist() == [[0, 0, 2]]
            saw_identity = True
        else:
            assert torch.equal(out, torch.flip(frame, dims=(1,)))
            assert out_coords.tolist() == [[0, 1, 2]]  # y=0 mirrors to shape[1]-1-0 = 1
            saw_flip = True
    assert saw_flip and saw_identity


def test_apply_jitter_clamps_to_non_negative():
    """Intensity jitter can drive a value below zero; the clamp keeps the frame a valid non-negative input."""
    aug = Augmentation(brightness=0.5, offset=1.0)
    out, _coords = aug.apply(torch.zeros(1, 2, 2), torch.tensor([[0, 0, 0]]), np.random.default_rng(3))
    assert torch.all(out >= 0.0)


def test_for_pair():
    """`for_pair` draws one transform whose gain, bias and flips fall in range — the draw a whole pair then shares.

    Over many seeds an all-eligible-axes policy must produce both a flip and no-flip on each axis, and the gain
    and bias must stay inside the half-ranges asked for; the transform is a value, so the pair reuses it verbatim.
    """
    aug = Augmentation(brightness=0.25, offset=2.0, flip_axes=(0, 2))
    seen_flips: set[tuple[int, ...]] = set()
    for seed in range(32):
        transform = aug.for_pair(np.random.default_rng(seed))
        assert 0.75 <= transform.gain <= 1.25
        assert -2.0 <= transform.bias <= 2.0
        assert set(transform.flips) <= {0, 2}
        seen_flips.add(transform.flips)
    assert () in seen_flips and (0, 2) in seen_flips  # both extremes are reachable


def test_is_identity():
    """The identity transform is exactly gain 1, bias 0 and no flips; any drawn perturbation is not identity."""
    assert FrameTransform(gain=1.0, bias=0.0, flips=()).is_identity
    assert not FrameTransform(gain=1.1, bias=0.0, flips=()).is_identity
    assert not FrameTransform(gain=1.0, bias=0.5, flips=()).is_identity
    assert not FrameTransform(gain=1.0, bias=0.0, flips=(1,)).is_identity


def test_frame():
    """`frame` applies gain then bias then the flips, clamped non-negative — the same op wherever the pair lands."""
    transform = FrameTransform(gain=2.0, bias=-1.0, flips=(2,))
    frame = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    out = transform.frame(frame)
    assert torch.equal(out, torch.flip((frame * 2.0 - 1.0).clamp(0.0), dims=(2,)))
    assert torch.all(out >= 0.0)


def test_coords():
    """`coords` mirrors only the flipped axes about their extent and leaves the rest of each centre in place."""
    transform = FrameTransform(gain=1.0, bias=0.0, flips=(0, 2))
    mirrored = transform.coords(torch.tensor([[0, 1, 3]]), shape=(4, 5, 6))
    assert mirrored.tolist() == [[3, 1, 2]]  # z: 4-1-0=3, y untouched, x: 6-1-3=2
    assert FrameTransform(gain=1.0, bias=0.0, flips=()).coords(torch.tensor([[0, 1, 3]]), (4, 5, 6)).tolist() == [
        [0, 1, 3]
    ]
