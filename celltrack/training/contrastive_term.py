"""The contrastive objective as the trainer sees it — one loss, and the PLACEMENT the configuration chose.

`InfoNCE` is the objective and knows nothing about where it acts; `ContrastiveProjection` is the head and
knows nothing about the loss. This joins them, because the two live in sibling packages that may not import
each other, and because the choice between them is one decision — on the shared features (measured: proxy
0.8414 -> 0.7292, node recall 0.966 -> 0.867, while the term itself converged) or through a head of its own.

The site owns which vectors are contrasted AND the width of the contrasted block, together. They cannot be
chosen apart: `InfoNCE.of` takes the LEADING width deliberately (an earlier version subtracted a trailing
position block and silently mis-sliced the moment a third block was appended), so whatever is handed to it
must arrive with its own true width, not a width assumed elsewhere.
"""

from __future__ import annotations

from typing import override

from jaxtyping import Float
from torch import Tensor, nn

from celltrack.losses.info_nce import InfoNCE
from celltrack.models.contrastive_projection import ContrastiveProjection, ContrastiveProjectionConfig
from celltrack.training.joint_config import LossCfg


class ContrastiveTerm(nn.Module):
    """InfoNCE over whichever vectors the configured site says it acts on — the features, or a projection.

    A module, not a function, because the projecting sites carry trained weights: the head is optimised
    alongside the two model heads (also in the detached regime — detaching spares the TRUNK, not the head).
    At the `FEATURES` site it holds no parameters at all, so a run that did not ask for a head builds none
    and draws nothing from the RNG.
    """

    def __init__(self, loss: LossCfg, appearance_dim: int) -> None:
        super().__init__()
        self.appearance_dim = appearance_dim
        self.temperature = loss.temperature
        site = loss.contrastive_site
        self.projection = (
            ContrastiveProjection(ContrastiveProjectionConfig(appearance_dim, stop_gradient=site.stop_gradient))
            if site.projects
            else None
        )

    @override
    def forward(
        self,
        source_features: Float[Tensor, "s d"],
        target_features: Float[Tensor, "u d"],
        edge_matrix: Float[Tensor, "s u"],
    ) -> Float[Tensor, ""]:
        """The contrastive loss for one pair, UNWEIGHTED — the trainer scales and logs it separately."""
        source, width = self._contrasted(source_features)
        target, _ = self._contrasted(target_features)
        return InfoNCE.of(source, target, edge_matrix, self.temperature, width)

    def _contrasted(self, features: Float[Tensor, "n d"]) -> tuple[Float[Tensor, "n a"], int]:
        """The vectors to contrast and the TRUE leading width of the appearance block inside them.

        Unprojected, that is the backbone's feature width and the position (and velocity) blocks trail it.
        Projected, the head has already taken the appearance slice, so the whole embedding is appearance and
        its full width is contrasted — the leading-width contract holds either way, with nothing assumed.
        """
        if self.projection is None:
            return features, self.appearance_dim
        return self.projection.forward(features), self.projection.embedding_dim
