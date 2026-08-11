"""Four-view flip TTA on the EDGE head, fused by a Jensen-Shannon-reliability logarithmic opinion pool.

We already flip-TTA the detector and do nothing of the kind on the association head, so every candidate
matrix rests on a single spatial presentation of the gap. This runs the head over the detector's own four
views and fuses them with `JensenShannonLogPool` — a soft AND, so a pair survives only if the views agree.

**This is not the refuted 8-fold detector TTA (`fcd`), and must not be retired by association with it.**
That experiment ensembled EIGHT dihedral views — flips AND rotations — of the DETECTOR. It failed for a
stated mechanism: the backbone is trained with in-plane flips only, so a rotated view presents in-plane
features it never learned, its response misaligns, and the arithmetic average smears the very logit peaks the
equality-NMS read-out depends on. Three things differ here, each of them the reason the refutation does not
carry: the views are the FOUR flips the network was actually trained under, with no rotation; the stage is the
association head, whose output is a per-target distribution over candidate sources rather than a voxel peak
map that must stay sharp; and the fusion is a reliability-weighted logarithmic pool, which pulls a dissenting
view DOWN, where the refuted arithmetic mean averaged it in. The frontier's own note on this fusion is that it
"avoids the graph inflation caused by a single optimistic spatial view" — the failure an arithmetic mean has
and this does not.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from celltrack.edges.js_log_pool import JensenShannonLogPool
from celltrack.models.edge_transformer import EdgeGap, EdgeTransformerScorer
from celltrack.models.prior_velocity import GapHistory
from celltrack.models.temporal_unet_detector import FlipView


class FlipViewEdgeScorer:
    """One seed's edge head scored over the detector's four flip views and pooled into a single logit matrix.

    The views come from `FlipView.tta_ensemble` — the same definition the detector's own TTA iterates, so the
    two stages cannot drift apart into two flip sets. Each view mirrors the frames the backbone reads AND the
    node positions read out of it, so every view returns its logits in the caller's node order and the pool is
    over matrices that already correspond entry for entry.
    """

    def __init__(self, scorer: EdgeTransformerScorer) -> None:
        self.scorer = scorer

    def gap_logits(self, gap: EdgeGap, history: GapHistory) -> tuple[Float[Tensor, "s t"], GapHistory]:
        """The gap's pooled forward logits, and the history the next gap reads — taken from the IDENTITY view.

        The carried history is a velocity the head will consume as an input next gap, not a score to ensemble:
        it must be the one this seed's weights produced on the presentation the trainer fed them. Pooling it
        across views would hand the next gap a displacement no view actually predicted.
        """
        views: list[Float[Tensor, "s t"]] = []
        carried = GapHistory()
        for view in FlipView.tta_ensemble():
            window = self.scorer._gap_window(gap, reverse=False, view=view)  # noqa: SLF001
            views.append(self.scorer._head_logits(window, history))  # noqa: SLF001
            if not view.dims:
                carried = self.scorer._history_after(window)  # noqa: SLF001
        return JensenShannonLogPool.pool(torch.stack(views)), carried

    def reverse_gap_logits(self, gap: EdgeGap) -> Float[Tensor, "t s"]:
        """The gap's pooled logits with the temporal pair SWAPPED — the stateless direction, so no history."""
        views = [
            self.scorer._head_logits(self.scorer._gap_window(gap, reverse=True, view=view), GapHistory())  # noqa: SLF001
            for view in FlipView.tta_ensemble()
        ]
        return JensenShannonLogPool.pool(torch.stack(views))
