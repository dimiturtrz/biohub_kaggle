"""Unit tests for the attention bias itself — the zero start, the physical units, the basis over the gate.

The three facts the bias has to hold: it is exactly zero before training (so a pretrained head cannot move),
its distance is in MICROMETRES (z voxels are four times deeper than y/x are wide), and its basis spans zero
to the linker's gate, the campaign's one statement of how far a cell travels across a frame gap.
"""

import torch

from celltrack.models.relative_position_bias import (
    DirectionalAttentionBias,
    DistanceAttentionBias,
    RelativePositionConfig,
)
from core.geometry import Spacing

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)
_GATE_UM = 10.0
_HEADS = 2


def _config() -> RelativePositionConfig:
    return RelativePositionConfig(spacing=_SPACING, gate_um=_GATE_UM, enabled=True)


def _directional_config() -> RelativePositionConfig:
    return RelativePositionConfig(spacing=_SPACING, gate_um=_GATE_UM, enabled=True, directional=True)


def test_forward():
    """Untrained, the bias is the exact zero matrix whatever geometry it is handed — the warm start's guarantee."""
    bias = DistanceAttentionBias(_HEADS, _config())
    geometry = torch.Generator().manual_seed(0)
    coords_q = torch.rand(2, 3, 3, generator=geometry) * 20.0
    coords_kv = torch.rand(2, 5, 3, generator=geometry) * 20.0

    scores = bias.forward(coords_q, coords_kv)

    assert scores.shape == (2 * _HEADS, 3, 5)  # (B·heads, Nq, Nkv), the shape attention takes
    assert torch.equal(scores, torch.zeros_like(scores))


def test_forward_once_trained():
    """With non-zero weights the bias is a per-head function of separation — near and far get different offsets."""
    bias = DistanceAttentionBias(_HEADS, _config())
    with torch.no_grad():
        bias.weight.normal_(std=1.0)
    coords_q = torch.zeros(1, 1, 3)
    near_then_far = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 60.0, 0.0]]])  # 0.4 um and 24 um apart in y

    scores = bias(coords_q, near_then_far)

    assert not torch.isclose(scores[0, 0, 0], scores[0, 0, 1])
    assert not torch.isclose(scores[0, 0, 0], scores[1, 0, 0])  # the two heads weigh the same distance apart


def test_separation_um():
    """Distance is physical: one voxel of z is 1.625 um and one of x is 0.40625 um, four times shorter."""
    bias = DistanceAttentionBias(_HEADS, _config())
    origin = torch.zeros(1, 1, 3)
    one_voxel = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

    separation = bias.separation_um(origin, one_voxel)

    torch.testing.assert_close(separation, torch.tensor([[[_SPACING.z, _SPACING.x]]]))


def test_radial_basis():
    """A pair at a centre saturates that basis function; the bank tiles zero to the gate at even spacing."""
    bias = DistanceAttentionBias(_HEADS, _config())

    at_zero = bias.radial_basis(torch.zeros(1, 1, 1))
    at_gate = bias.radial_basis(torch.full((1, 1, 1), _GATE_UM))

    assert at_zero[0, 0, 0, 0] == 1.0  # the first centre is contact, zero separation
    assert at_gate[0, 0, 0, -1] == 1.0  # the last centre is the gate itself
    assert at_zero[0, 0, 0, -1] < at_zero[0, 0, 0, 0]


def test_directional_forward_zero_start():
    """Directional too starts at the exact zero matrix — both weight banks zero, so a warm start cannot move."""
    bias = DirectionalAttentionBias(_HEADS, _directional_config())
    geometry = torch.Generator().manual_seed(0)
    coords_q = torch.rand(2, 3, 3, generator=geometry) * 20.0
    coords_kv = torch.rand(2, 5, 3, generator=geometry) * 20.0

    scores = bias.forward(coords_q, coords_kv)

    assert scores.shape == (2 * _HEADS, 3, 5)
    assert torch.equal(scores, torch.zeros_like(scores))


def test_directional_separates_hemispheres():
    """The term the radial parent cannot express: two pairs at the SAME distance, opposite directions, differ."""
    bias = DirectionalAttentionBias(_HEADS, _directional_config())
    with torch.no_grad():
        bias.direction_weight.normal_(std=1.0)
    coords_q = torch.zeros(1, 1, 3)
    toward_then_away = torch.tensor([[[0.0, 10.0, 0.0], [0.0, -10.0, 0.0]]])  # same |sep|, +y vs -y

    scores = bias(coords_q, toward_then_away)

    radial = DistanceAttentionBias(_HEADS, _config())
    with torch.no_grad():
        radial.weight.normal_(std=1.0)
    radial_scores = radial(coords_q, toward_then_away)

    assert not torch.isclose(scores[0, 0, 0], scores[0, 0, 1])  # direction splits equal-distance pairs
    assert torch.isclose(radial_scores[0, 0, 0], radial_scores[0, 0, 1])  # the radial parent cannot


def test_direction_basis():
    """The basis is the offset's cosine against fixed signed axes: reversing the offset negates every anchor."""
    bias = DirectionalAttentionBias(_HEADS, _directional_config())
    origin = torch.zeros(1, 1, 3)
    toward = torch.tensor([[[0.0, 10.0, 0.0]]])  # +y
    away = torch.tensor([[[0.0, -10.0, 0.0]]])  # -y, same |sep|

    basis_toward = bias.direction_basis(origin, toward)
    basis_away = bias.direction_basis(origin, away)

    torch.testing.assert_close(basis_away, -basis_toward)  # reversing direction flips every cosine
    assert basis_toward.abs().max() <= 1.0 + 1e-6  # cosines are bounded by the unit axes
    torch.testing.assert_close(basis_toward.max(), torch.tensor(1.0))  # one signed axis aligns the offset


def test_directional_coincident_is_nan_safe():
    """A coincident pair has no direction — the unit offset is set to zero, so the term is silent, not a nan."""
    bias = DirectionalAttentionBias(_HEADS, _directional_config())
    with torch.no_grad():
        bias.direction_weight.normal_(std=1.0)
    origin = torch.zeros(1, 2, 3)

    scores = bias(origin, origin)

    assert torch.isfinite(scores).all()
    diagonal = scores[:, torch.arange(2), torch.arange(2)]
    assert torch.equal(diagonal, torch.zeros_like(diagonal))
