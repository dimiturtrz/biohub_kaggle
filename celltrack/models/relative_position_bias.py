"""The relative-geometry term itself: a per-head attention bias learned over a pair's separation in micrometres.

Applied to the pack's head by `celltrack.models.relative_position_transformer`; this module is the physics —
what the bias is a function of, in what units, and why it starts at exactly zero.

The pack's head is not geometry-blind at the END: `SimpleNodeTransformer` concatenates the candidate
displacement `rel = (coords_t - coords_t1) / 100.0` onto every expanded pair before `pair_mlp`. The layers
BEFORE it are. A node's representation is built by attending over the other frame's nodes with no notion of
how far away any of them is, so "which of my neighbours should inform me" is decided on appearance alone —
and appearance is exactly what a field of near-identical nuclei does not disambiguate. Trackastra's verified
mechanism is the opposite: relative position injected into EVERY attention layer (RoPE over space and time)
plus a mask beyond a distance `d_max` (`research/deep_dives/2026-08-10_association-and-motion-sota-corrective.md`).

This adds the spatial half of that, in the one form a warm start survives: a per-head ADDITIVE bias on the
attention scores, a learned function of the pair's separation in MICROMETRES, whose parameters start at exact
zero. RoPE proper is parameter-free but it is not identity-preserving — rotating q and k changes every score
the moment it is switched on, which is fine for Trackastra (trained with it from scratch) and wrong for us
(a published head we must not move at step 0). A zero-initialised bias is the same trade `PriorVelocity.
widen_linear_inputs` makes for the input projection: the pretrained weights load unchanged, the model's
output is identical on the first step, and the new signal is learned from zero gain upward.

The TEMPORAL half of Trackastra's encoding is deliberately NOT built here. Our window is T=2: every scored
pair spans exactly one frame gap, so a relative temporal offset is the same constant for every pair and any
encoding of it is a constant added to all scores — an identity under softmax. It becomes live only if the
window ever widens past two frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from core.geometry import Spacing


@dataclass(frozen=True)
class RelativePositionConfig:
    """What the attention bias needs to know: the voxel geometry, the admissible reach, and its resolution.

    `gate_um` is REQUIRED rather than defaulted, and it is meant to be the linker's own `gate_um` (10 um for
    this campaign) — the statement of the furthest a cell travels across one frame gap. It is the campaign's
    only argued distance scale, so the bias reuses it instead of introducing a second one. The pack's `/100.0`
    divisor on `rel` is the other candidate and is disqualified for a DISTANCE: it divides raw voxels, and a
    voxel is 1.625 um deep against 0.40625 um wide, so a norm taken over it measures a direction-dependent
    quantity that is not length. `models` sits below `linkers` in the import layering, so the value is passed
    in rather than imported — the linker's config stays the one home of the number.
    """

    spacing: Spacing
    gate_um: float
    basis_count: int = 4
    enabled: bool = False
    directional: bool = False


class DistanceAttentionBias(nn.Module):
    """A per-head additive term on cross-attention scores, learned over the pair's separation in micrometres.

    The bias is a per-head weighting of a FIXED radial basis: Gaussians tiling `[0, gate_um]` at even spacing,
    each as wide as that spacing. Nothing about the shape of the distance dependence is assumed — a head may
    learn to prefer the near set, to suppress it, or to favour a ring at one cell's displacement — but every
    weight starts at zero, so the bias is exactly the zero matrix until training moves it. The basis stops at
    the gate because the linker refuses every longer transition anyway; resolution outside the admissible set
    would be capacity spent on pairs no assignment can take.
    """

    def __init__(self, head_count: int, config: RelativePositionConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(head_count, config.basis_count))
        spread = config.gate_um / max(config.basis_count - 1, 1)
        self.register_buffer("centres_um", torch.arange(config.basis_count, dtype=torch.float32) * spread)
        self.register_buffer("spacing_um", torch.tensor(config.spacing.as_array(), dtype=torch.float32))
        self.width_um = spread

    @override
    def forward(
        self, coords_q: Float[Tensor, "b nq 3"], coords_kv: Float[Tensor, "b nkv 3"]
    ) -> Float[Tensor, "bh nq nkv"]:
        """The additive score bias per (query, key) pair, flattened to the `(B·heads, Nq, Nkv)` attention takes."""
        basis = self.radial_basis(self.separation_um(coords_q, coords_kv))
        bias = torch.einsum("hk,bqvk->bhqv", self.weight.to(basis.dtype), basis)
        return bias.reshape(-1, bias.shape[2], bias.shape[3])

    def separation_um(
        self, coords_q: Float[Tensor, "b nq 3"], coords_kv: Float[Tensor, "b nkv 3"]
    ) -> Float[Tensor, "b nq nkv"]:
        """Euclidean distance between every pair, in MICROMETRES — voxel offsets scaled by the anisotropic spacing.

        The same conversion `BoundaryPrior` makes for the same reason: z voxels are four times deeper than y/x
        are wide, so a distance taken on raw indices is not a physical length and a gate expressed in
        micrometres cannot be applied to it.
        """
        spacing = cast(Tensor, self.spacing_um).to(coords_q.dtype)
        offset_um = (coords_q.unsqueeze(2) - coords_kv.unsqueeze(1)) * spacing
        return offset_um.norm(dim=-1)

    def radial_basis(self, separation_um: Float[Tensor, "b nq nkv"]) -> Float[Tensor, "b nq nkv k"]:
        """Each pair's separation as its activation on the fixed Gaussian bank tiling the admissible range."""
        centres = cast(Tensor, self.centres_um).to(separation_um.dtype)
        deviation = (separation_um.unsqueeze(-1) - centres) / self.width_um
        return torch.exp(-0.5 * deviation.square())


# The six signed unit axes in the anisotropy-corrected (micrometre) frame — a fixed directional anchor set.
# A pair's unit offset is scored against each by cosine, so a head can learn to prefer any one hemisphere
# (or, combining anchors, any single preferred direction with a learned width). Six is the minimal set that
# spans all three axes in both senses; more anchors would refine the angular resolution the confusor does
# not need (its inversion is a coarse hemisphere effect, not a fine cone).
_DIRECTION_ANCHORS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
_ZERO_NORM_EPS_UM = 1e-6


class DirectionalAttentionBias(DistanceAttentionBias):
    """The radial bias plus a per-head term over the pair's offset DIRECTION — the axis `.norm()` throws away.

    The dense-movie confusor is a DIRECTIONAL inversion: the true successor is not the nearest node but the one
    the cell drifts TOWARD, and the field's common-mode drift makes that a consistent lab-frame direction. The
    radial parent collapses every offset to a scalar separation, so a head can prefer the near set or a ring at
    one displacement but can never prefer a HEMISPHERE — exactly the discrimination the confusor needs. This adds
    that: each pair's unit offset (in the anisotropy-corrected micrometre frame, so a physical direction and not a
    voxel-index one) is scored against `_DIRECTION_ANCHORS` by cosine, and a per-head weight over those six
    activations is summed into the score alongside the radial term.

    Both weight banks start at exact zero, so the total bias is the zero matrix until training moves it — the same
    warm-start trade the parent makes, extended to the new directional parameters. A coincident pair has no
    direction; its unit offset is set to zero, contributing zero to every anchor, so the term is silent exactly
    where direction is undefined rather than dividing by a vanishing norm.
    """

    def __init__(self, head_count: int, config: RelativePositionConfig) -> None:
        super().__init__(head_count, config)
        anchors = torch.tensor(_DIRECTION_ANCHORS, dtype=torch.float32)
        self.register_buffer("anchors", anchors)
        self.direction_weight = nn.Parameter(torch.zeros(head_count, anchors.shape[0]))

    @override
    def forward(
        self, coords_q: Float[Tensor, "b nq 3"], coords_kv: Float[Tensor, "b nkv 3"]
    ) -> Float[Tensor, "bh nq nkv"]:
        """The radial score bias plus the directional one, flattened to the `(B·heads, Nq, Nkv)` attention takes."""
        radial = torch.einsum(
            "hk,bqvk->bhqv", self.weight.to(coords_q.dtype), self.radial_basis(self.separation_um(coords_q, coords_kv))
        )
        directional = torch.einsum(
            "ha,bqva->bhqv", self.direction_weight.to(coords_q.dtype), self.direction_basis(coords_q, coords_kv)
        )
        bias = radial + directional
        return bias.reshape(-1, bias.shape[2], bias.shape[3])

    def direction_basis(
        self, coords_q: Float[Tensor, "b nq 3"], coords_kv: Float[Tensor, "b nkv 3"]
    ) -> Float[Tensor, "b nq nkv a"]:
        """Each pair's unit offset scored against every fixed anchor by cosine — zero where the pair is coincident.

        The offset is taken in micrometres (voxel indices scaled by the anisotropic spacing) so its direction is
        physical: a raw-index unit vector would point 'more in z' purely because z voxels are four times deeper.
        """
        spacing = cast(Tensor, self.spacing_um).to(coords_q.dtype)
        offset_um = (coords_q.unsqueeze(2) - coords_kv.unsqueeze(1)) * spacing
        norm = offset_um.norm(dim=-1, keepdim=True)
        unit = torch.where(norm > _ZERO_NORM_EPS_UM, offset_um / norm.clamp_min(_ZERO_NORM_EPS_UM), 0.0)
        return torch.einsum("bqvd,ad->bqva", unit, cast(Tensor, self.anchors).to(coords_q.dtype))
