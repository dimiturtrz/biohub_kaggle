"""InfoNCE over node appearance — a contrastive objective named by its form, not its use.

Detection trains a shared representation toward INVARIANCE (every cell should look like a cell); association
needs DISCRIMINATION (this cell must differ from the one 3um away). Nothing in a detection+edge objective asks
for the second: the edge loss only asks the HEAD to rank pairs given whatever features it is handed, so a
backbone that collapses neighbouring cells onto one point is never penalised. `InfoNCE` asks the FEATURES
directly — each annotated source must rank its true successor above every other candidate in the next frame
under a plain cosine similarity, with no head in between.

Sources with no annotated successor contribute nothing. Our labels are ~1-2% dense, so an unlabelled source is
not a negative — it is a source whose successor was never written down, and treating it as one would train the
features to push apart pairs that are in fact the same cell.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class InfoNCE:
    """Cross-entropy over temperature-scaled cosine similarities — an annotated source must pick its successor."""

    @staticmethod
    def of(
        source_features: Float[Tensor, "s d"],
        target_features: Float[Tensor, "u d"],
        edge_matrix: Float[Tensor, "s u"],
        temperature: float,
        position_dim: int,
    ) -> Float[Tensor, ""]:
        """Softmax over targets at `temperature`, cross-entropy against the annotated successor(s), row-meaned.

        `position_dim` is the width of the TRAILING position block the caller's node features carry, and
        dropping it is the whole point rather than a detail: node features are an appearance vector
        concatenated with a sinusoidal embedding of the node's coordinates, so contrasting the full vector
        lets the objective be satisfied through POSITION — nearby cells get similar embeddings, distant ones
        different — which is exactly the lesson that produces the mislinks this term exists to fix. Only the
        leading appearance channels are compared; do not "simplify" the slice away.

        A DIVISION (one source, two annotated successors) is scored as cross-entropy against the uniform
        distribution over its positives — the mean of `-log p` across them. That asks the mother to rank both
        daughters highly without forcing a choice between them, which a one-hot target would.

        Rows with no annotated successor are dropped, not counted as negatives; a pair with no such row
        returns a defined zero, never a NaN from an empty mean.
        """
        positives = edge_matrix > 0
        annotated = positives.any(dim=1)
        if not bool(annotated.any()):
            return source_features.new_zeros(())
        with torch.autocast(device_type=source_features.device.type, enabled=False):
            source = InfoNCE._appearance(source_features, position_dim)
            target = InfoNCE._appearance(target_features, position_dim)
            log_probability = torch.log_softmax(source @ target.T / temperature, dim=1)[annotated]
            weight = positives[annotated].to(log_probability.dtype)
            return -((log_probability * weight).sum(dim=1) / weight.sum(dim=1)).mean()

    @staticmethod
    def _appearance(features: Float[Tensor, "n d"], position_dim: int) -> Float[Tensor, "n a"]:
        """The L2-normalised appearance block — the leading channels, with the trailing position embed dropped."""
        return F.normalize(features[:, : features.shape[1] - position_dim].float(), dim=1)
