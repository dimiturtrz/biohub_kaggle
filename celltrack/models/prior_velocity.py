"""A node's PRIOR displacement — where the cell was already heading — and the warm start that makes it free.

The pack's pair head already reads the CANDIDATE displacement: `SimpleNodeTransformer` concatenates
`rel = (coords_t - coords_t1) / 100.0` onto every expanded (source, target) pair before `pair_mlp`. So the
head is not geometry-blind; what it cannot see is HISTORY. Without the previous step, "is this candidate
consistent with where this cell was already heading?" is unanswerable, which is exactly the documented
failure — a fast mover's true successor lands FAR while a slower neighbour sits NEAR, and both raw distance
and the learned probability pick the near-wrong cell.

The prior step has to be computed the SAME WAY at training and at inference, so it cannot come from ground
truth edges (a train/inference asymmetry we have already paid for once). It comes from the model's own
affinity on the PREVIOUS gap: an affinity-weighted EXPECTED incoming displacement, soft rather than a hard
argmax link — differentiable, identical in both regimes, and shrinking toward zero exactly when the previous
gap was ambiguous or when a node has no incoming mass at all.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn


class PriorVelocity:
    """The expected incoming displacement of each node, and the zero-init widening that keeps a warm start exact."""

    DIM = 3
    # The pack's own undocumented divisor on `rel` inside `SimpleNodeTransformer.forward`. Reused rather than
    # replaced by a second invented scale: the velocity then lives in the head's displacement units up to the
    # constant per-axis downsample factor (velocity arrives in `grid_positions` space, `rel` in full-resolution
    # voxels), a fixed gain the zero-initialised input columns absorb while learning.
    SCALE = 100.0

    @staticmethod
    def expected_incoming(
        coordinates: Float[Tensor, "n 3"],
        previous_coordinates: Float[Tensor, "p 3"] | None = None,
        previous_probabilities: Float[Tensor, "p n"] | None = None,
    ) -> Float[Tensor, "n 3"]:
        """Each node's affinity-weighted displacement from the previous frame: `Σ_i P[i, j] · (c_j − c_i)`.

        `previous_probabilities` is the previous gap's source→target matrix in the scorer's own normalisation
        (softmax over the sources, so each target column carries at most unit incoming mass). Written as
        `mass_j · c_j − (Pᵀ c_prev)_j`, a column whose mass falls short of one — an entering cell, or a gap
        the head found ambiguous — yields a proportionally SHRUNKEN velocity rather than an extrapolated
        guess, degrading toward the zero vector instead of toward a confident wrong direction.

        Nodes in the first frame have no previous gap at all: passing neither argument returns exact zeros,
        the same value an unmatched node degrades to, so "no history" and "no usable history" read alike to
        the head instead of needing a separate flag column.
        """
        if previous_coordinates is None or previous_probabilities is None:
            return torch.zeros_like(coordinates)
        weights = previous_probabilities.to(coordinates.dtype)
        incoming_mass = weights.sum(dim=0).unsqueeze(-1)
        return incoming_mass * coordinates - weights.T @ previous_coordinates.to(coordinates.dtype)

    @staticmethod
    def widen_linear_inputs(layer: nn.Linear, extra_inputs: int) -> nn.Linear:
        """A copy of `layer` taking `extra_inputs` more TRAILING inputs, whose new columns are exactly zero.

        Appending a feature widens the head's input projection, which would otherwise make the published
        pretrained weights unloadable. Copying them into the leading columns and zeroing the new ones keeps
        the layer's output bitwise identical to the original at step 0 for any value of the appended feature,
        so the warm start is preserved and the feature is learned from zero gain upward rather than from a
        random projection that perturbs a trained head.
        """
        widened = nn.Linear(
            layer.in_features + extra_inputs,
            layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        with torch.no_grad():
            widened.weight.zero_()
            widened.weight[:, : layer.in_features].copy_(layer.weight)
            if widened.bias is not None and layer.bias is not None:
                widened.bias.copy_(layer.bias)
        return widened
