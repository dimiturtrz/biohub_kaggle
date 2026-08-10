"""The joint detector+edge training objective — pilkwang's recipe composed from the two shipped losses.

A joint step supervises a frame pair (t, t+1): both frames' detection heads fire on their ground-truth
cell centres, and the edge transformer scores the true parent→child links between them. This module is a
pure composition — it reuses the count-normalised detection BCE and the frontier focal-BCE verbatim and
only decides how they combine. The edge term is primary (weight 1); detection is scaled by ``det_weight``
so the shared UNet substrate keeps serving the linker without letting the two per-frame detection terms
swamp the single edge term.

No model and no GPU live here: it takes logits and targets already on some device and returns a scalar,
so it is CPU-testable in isolation from the trainer that produces those tensors.
"""

from jaxtyping import Float, Int
from torch import Tensor

from celltrack.training.edge_finetune import EdgeHardNegativeFinetuner
from celltrack.training.tunet_detector import TUNetDetectorTrainer


class JointLoss:
    """Composes the reused detection and edge losses into the single scalar a joint step descends."""

    @staticmethod
    def compute(  # noqa: PLR0913 — a composition boundary carries both losses' tensors, not a decomposable arg list
        detection_t: Float[Tensor, "z y x"],
        detection_t1: Float[Tensor, "z y x"],
        centres_t: Int[Tensor, "s 3"],
        centres_t1: Int[Tensor, "u 3"],
        edge_logits: Float[Tensor, "s u"],
        edge_matrix: Float[Tensor, "s u"],
        neg_weight: float,
        det_weight: float,
    ) -> Float[Tensor, ""]:
        """Edge focal-BCE plus the down-weighted detection BCE of both frames — pilkwang's joint objective.

        The single-frame detection logits ``(z, y, x)`` are unsqueezed to a batch of one ``(1, z, y, x)`` and
        their centres wrapped in a one-element list because the detection loss is batched; the centres are
        already downsampled voxel indices, so the loss just indexes into the map with no coordinate work.
        """
        detection = TUNetDetectorTrainer._detection_loss(  # noqa: SLF001
            detection_t.unsqueeze(0), [centres_t], neg_weight
        ) + TUNetDetectorTrainer._detection_loss(  # noqa: SLF001
            detection_t1.unsqueeze(0), [centres_t1], neg_weight
        )
        edge = EdgeHardNegativeFinetuner._frontier_loss(edge_logits, edge_matrix)  # noqa: SLF001
        return edge + det_weight * detection
