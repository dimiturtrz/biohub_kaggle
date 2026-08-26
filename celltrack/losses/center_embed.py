"""The CELLECT-style center-point contrastive term — a dedicated embedding, contrasted and confusor-separated.

`InfoNCE` (its sibling) already asks the FEATURES to rank a source's true successor above every candidate, but on
the SHARED node features the edge head also consumes — where the objective was measured to trade detection recall
away (`joint_config.LossCfg.contrastive_site`). This term acts instead on `TemporalUNetDetector.embed_head`, a
head whose ONLY job is to carry this gradient: a wide per-cell vector the backbone is free to reshape for
discrimination without disturbing the one-channel detection logit beside it.

Two reused primitives, summed, under ONE weight:
- `InfoNCE.of` over the full-width embedding — the global cosine cross-entropy (each source picks its successor).
- `HardNegativeMargin.of` on the embedding's own cosine similarities, using the SAME proximity-mined near-decoys
  the edge term mines — a local ranking that holds each true pair's embedding-similarity above its nearest
  confusors', which is exactly the dense mislink this head exists to separate.

They are summed 1:1, not with a tuned mix: both are the negative log-likelihood of the SAME event — the true
successor beats the wrong ones — under two negative samplings (all targets by softmax, nearest-K by margin), so
equal footing is the natural, number-free choice. The single `center_embed_weight` scales the pair against the
edge and detection terms.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.losses.hard_negative_margin import HardNegativeMargin
from celltrack.losses.info_nce import InfoNCE


class CenterEmbedContrastive:
    """The center-point contrastive objective over the dedicated embedding — global contrast plus a decoy margin."""

    @staticmethod
    def of(
        source_embed: Float[Tensor, "s e"],
        target_embed: Float[Tensor, "u e"],
        edge_matrix: Float[Tensor, "s u"],
        decoys: Int[Tensor, "h 2"],
        temperature: float,
    ) -> Float[Tensor, ""]:
        """`InfoNCE` over the full embedding plus the near-decoy margin on its cosine similarities, summed.

        `decoys` are the proximity-mined `(source, decoy)` rows the caller already built for the edge term
        (`HardNegativeMargin.mine`), reused here so both terms rank against the SAME hard set. A pair with no
        annotated successor returns a defined zero (both primitives do), never a NaN.
        """
        contrastive = InfoNCE.of(source_embed, target_embed, edge_matrix, temperature, source_embed.shape[-1])
        cosine = CenterEmbedContrastive._cosine(source_embed, target_embed, temperature)
        margin = HardNegativeMargin.of(cosine, edge_matrix, decoys)
        return contrastive + margin

    @staticmethod
    def _cosine(
        source_embed: Float[Tensor, "s e"], target_embed: Float[Tensor, "u e"], temperature: float
    ) -> Float[Tensor, "s u"]:
        """Temperature-scaled cosine similarities of the embeddings — the margin's "logits", in fp32.

        Computed with autocast disabled for the same reason `HardNegativeMargin.of` is: the margin reads a small
        difference of two of these, which a bf16 rounding would quantise away.
        """
        with torch.autocast(device_type=source_embed.device.type, enabled=False):
            source = F.normalize(source_embed.float(), dim=1)
            target = F.normalize(target_embed.float(), dim=1)
            return source @ target.T / temperature
