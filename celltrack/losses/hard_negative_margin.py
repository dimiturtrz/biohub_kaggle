"""Rank a true link ABOVE its nearest wrong candidates — the margin form of hard-negative mining.

The dense mislinks are affinity inversions: a wrong *near* neighbour is scored above the true *far*
successor (`dense_diagnosis.MislinkSignal`). Mining those decoys is the right diagnosis; the ABSOLUTE form of
the objective was the refuted one. `edge_finetune` (bead zni) pushed the mined decoy's softmax probability
toward zero, and the head satisfied it by FLATTENING — `P_true` collapsed 0.134 -> 0.017 with `P_chosen`, and
the arm lost 174 correct links to repair 4-6. Wrong-near and true-far candidates share the features, so a term
that names only the decoy drags the truth down with it.

THE FORM CHOSEN HERE, and why it cannot fail that way: the penalty is the pairwise-logistic (RankNet) loss
`softplus(logit_decoy - logit_true)` on the RAW logits of one source's row. It reads only the DIFFERENCE, so
adding any constant to a whole row leaves it exactly unchanged — depressing the decoy and the truth together
buys nothing, and the only way to reduce it is to re-rank. Logistic rather than a hinge because a hinge needs
a margin constant nobody here can derive (the logit scale is whatever the pretrained head happens to use),
while `softplus` is that same hinge smoothed at its own natural scale and introduces no number at all: it is
exactly the log-likelihood that the true link beats the decoy.

Per source the comparison uses the source's LOWEST true logit, so a dividing cell's every daughter has to beat
every decoy, not just its best-scored one.

Mining is by PROXIMITY in voxel-index space — a source's `keep` nearest WRONG targets — the same definition
the zni probe mined (`GapSupervision`, which now delegates here). Proximity, not the model's own top-scoring
wrong candidate: a decoy set defined by the model is a set the model can empty by changing its mind, whereas
the geometry that makes these cases hard does not move during a fine-tune.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class HardNegativeMargin:
    """The mined near-decoys of each annotated source, and the ranking penalty that keeps the truth above them."""

    @staticmethod
    def mine(
        source_positions: Float[Tensor, "s 3"],
        target_positions: Float[Tensor, "u 3"],
        links: Float[Tensor, "s u"],
        keep: int,
    ) -> Int[Tensor, "h 2"]:
        """`(source, decoy)` rows: for each source WITH a true successor, its `keep` nearest wrong targets.

        Sources the annotation does not link are skipped — nothing about them is known to be wrong. A frame
        offering fewer wrong targets than `keep` contributes only the ones it has, rather than padding with
        a repeated or a true column.
        """
        annotated = links.sum(dim=1) > 0
        wanted = min(keep, links.shape[1])
        if not bool(annotated.any()) or wanted == 0:
            return links.new_zeros((0, 2), dtype=torch.int64)
        distance = torch.cdist(source_positions.float(), target_positions.float())
        wrong_only = distance.masked_fill(links > 0, float("inf"))
        nearest = wrong_only.topk(wanted, dim=1, largest=False)
        rows = torch.arange(links.shape[0], device=links.device).unsqueeze(1).expand_as(nearest.indices)
        minable = annotated.unsqueeze(1) & torch.isfinite(nearest.values)
        return torch.stack((rows[minable], nearest.indices[minable]), dim=1)

    @staticmethod
    def of(logits: Float[Tensor, "s u"], links: Float[Tensor, "s u"], decoys: Int[Tensor, "h 2"]) -> Float[Tensor, ""]:
        """Mean `softplus(logit_decoy - logit_true)` over the mined pairs — zero when nothing was mined.

        Computed in fp32 with autocast disabled: the quantity IS a small difference of two logits, and a bf16
        subtraction quantises exactly the margin the term exists to move.
        """
        if decoys.numel() == 0:
            return logits.new_zeros(())
        with torch.autocast(device_type=logits.device.type, enabled=False):
            scores = logits.float()
            true_logit = scores.masked_fill(links <= 0, float("inf")).min(dim=1).values
            rows, columns = decoys[:, 0], decoys[:, 1]
            # softplus of a raw-logit DIFFERENCE is unbounded and grows linearly, unlike the sigmoid-bounded
            # BCE/NCE terms that absorb an outlier into their mean. A single exploded true_logit therefore
            # drives this term to millions and dominates the loss (avl8: hn 2.18M vs edge 17, early-stopped
            # epoch 5). The head's logits are O(1-10) (healthy edge BCE), so a true-vs-decoy gap above 50 is
            # already a maximally-confident inversion whose gradient is sigmoid(50)~=1 -- identical to any
            # larger gap. Clamping the difference at 50 preserves full ranking gradient across the genuine
            # hard range (0-50) and bounds only pathological/numerical outliers, capping the term near 50.
            margin = (scores[rows, columns] - true_logit[rows]).clamp(max=50.0)
            return F.softplus(margin).mean()
