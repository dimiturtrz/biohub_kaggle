"""The contrastive objective's OWN head — a small projection off the node appearance block.

Run directly on the shared node features, InfoNCE degraded the model: at weight 0.1 over a full epoch the
proxy fell 0.8414 -> 0.7292 and node recall 0.966 -> 0.867 while the contrastive term itself CONVERGED
(0.877 -> 0.428) and the edge loss fell with it. That is not a warm-up dip, it is a TRADE — one
representation cannot be maximally invariant (every cell should look like a cell, which is what detection
asks of it) and maximally discriminative (this cell must differ from the one 3um away, which is what
association asks) at once, so buying the second with a term that reshapes the features themselves is paid
for out of the first.

The fix is PLACEMENT, not weight and not patience. The contrastive gradient gets an embedding of its own to
shape: a projection whose output nothing else reads, so what the detect head consumes is the trunk's
features, and what the objective sharpens is a view of them. This is the same reason SimCLR contrasts
`g(h)` rather than `h` and then throws `g` away — the invariances a contrastive loss imposes are destructive
to the representation it is imposed on, and a head absorbs them.

`stop_gradient` is the strict form of that separation: with it the contrastive term shapes ONLY the head and
reaches no backbone weight at all, so the two regimes — shaping trunk-plus-head, or head alone — are one
field apart and can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from jaxtyping import Float
from torch import Tensor, nn


@dataclass(frozen=True)
class ContrastiveProjectionConfig:
    """What the head reads, and whether its gradient is allowed back into the trunk that produced it."""

    appearance_dim: int
    # Detach the head's input, so the contrastive term updates the projection and NOTHING upstream of it.
    stop_gradient: bool = False

    @property
    def embedding_dim(self) -> int:
        """The contrasted width — DERIVED from the appearance block, never a chosen round number.

        Bounded ABOVE by the appearance width: the head reads `appearance_dim` channels, so no wider output
        can carry information the input did not, and a wider embedding would only be a rank-deficient lift.

        Bounded BELOW by what the objective needs. One step contrasts one frame's candidates — order 1e2-1e3
        nodes — under cosine similarity, and the Johnson-Lindenstrauss bound for preserving their pairwise
        angles to distortion e is ~8*ln(n)/e^2, which at n=1e3 and e=0.5 is already ~220 dimensions, far
        above the 32-channel appearance block we have.

        The two bounds cross, so the only width that satisfies both is the appearance width itself: the
        projection is a learned RE-BASIS of the appearance block, not a compression of it. Nothing here is
        picked — the number follows the backbone's feature width whatever that is set to.
        """
        return self.appearance_dim


class ContrastiveProjection(nn.Module):
    """A two-layer MLP from a node's appearance block to the embedding InfoNCE is computed on.

    An MLP rather than a 1x1 conv stack because the nodes arrive ALREADY SAMPLED — a (n, d) table of point
    features, not a spatial map — so a 1x1 conv over it is a `Linear` with reshaping ceremony around it.
    Two layers with a nonlinearity between them rather than one: a single linear map composed with the
    cosine similarity is still a bilinear form on the features, which is close enough to contrasting them
    directly that the head would absorb little, and it is exactly the nonlinear head that SimCLR measured as
    carrying the effect.
    """

    def __init__(self, config: ContrastiveProjectionConfig) -> None:
        super().__init__()
        self.config = config
        width = config.embedding_dim
        self.mlp = nn.Sequential(nn.Linear(config.appearance_dim, width), nn.GELU(), nn.Linear(width, width))

    @property
    def embedding_dim(self) -> int:
        """The width of what `forward` returns — what InfoNCE must be told the contrasted block is."""
        return self.config.embedding_dim

    @override
    def forward(self, features: Float[Tensor, "n d"]) -> Float[Tensor, "n e"]:
        """Project the LEADING appearance channels; detached from the trunk when the config says so.

        The slice is the same contract InfoNCE enforces one level down and for the same reason: node features
        are appearance followed by a sinusoidal position embedding (and, with kinematics on, a prior-velocity
        block), and a contrastive objective handed those trailing blocks can satisfy itself through POSITION.
        Slicing here means the whole projected vector is appearance-derived, so the width InfoNCE contrasts
        is the full embedding — the leading-width contract is kept, not worked around.
        """
        appearance = features[:, : self.config.appearance_dim]
        return self.mlp(appearance.detach() if self.config.stop_gradient else appearance)
