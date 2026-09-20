"""The center-offset regression term: every cell voxel points at the centre it belongs to.

Detection alone cannot separate two cells whose responses have merged into one blob — the peak readout sees
one maximum and emits one node, which is the detection-merge half of the dense-regime gap. An offset field
carries the missing information: a voxel's vector to ITS OWN centre differs between two touching cells even
where their cellness does not, so a readout that clusters by predicted centre can split a blob the peak
readout cannot. This is the EmbedSeg construction, minus its instance masks — we have annotated centroids
and nothing else, so the target is the VORONOI assignment of each voxel to its nearest annotated centre.

Supervision is bounded to a physical radius around each centre. There are no masks, so a voxel far from
every annotation has no defensible owner: the nearest centroid may be a cell it is not part of, and the
annotation is a sparse subsample, so the true owner may not be annotated at all. Outside the radius the
term is silent rather than wrong.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.losses.offset_target import OffsetTarget


class CenterOffsetLoss:
    """Masked L1 between the predicted offset field and the Voronoi target — microns, supervised voxels only."""

    @staticmethod
    def of(
        offsets: Float[Tensor, "b three z y x"],
        centres: list[Int[Tensor, "n 3"]],
        radius: tuple[int, int, int],
        scale: tuple[float, float, float],
    ) -> Float[Tensor, ""]:
        """The batch's mean per-voxel offset error, over voxels within `radius` of an annotated centre.

        L1 rather than L2 because the target is piecewise-constant across a Voronoi boundary: a voxel on the
        seam between two cells flips its whole target vector, and a squared penalty would let those seam
        voxels dominate the gradient of the interiors the readout actually clusters on.

        A frame with no annotated centre contributes no loss rather than a zero, so an empty crop cannot
        dilute the batch mean towards a term that was never supervised there.
        """
        shape = (offsets.shape[2], offsets.shape[3], offsets.shape[4])
        errors = []
        for item, frame_centres in enumerate(centres):
            if not len(frame_centres):
                continue
            target = OffsetTarget.of(frame_centres.to(offsets.device), shape, radius, scale)
            supervised = target.supervised.sum()
            if not float(supervised):
                continue
            error = (offsets[item].float() - target.offsets).abs().sum(dim=0) * target.supervised
            errors.append(error.sum() / supervised)
        if not errors:
            return offsets.sum() * 0.0  # keeps the graph connected on an unsupervised batch
        return torch.stack(errors).mean()
