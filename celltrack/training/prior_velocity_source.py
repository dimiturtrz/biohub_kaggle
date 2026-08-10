"""Where a training step's prior velocity comes from — the model's own affinity on the previous gap.

`PriorVelocity` defines what the feature IS (the expected incoming displacement, and the zero-init widening
that keeps a warm start exact). This is the training-time half: the history has to be produced the same way
the deployed scorer would produce it, which means it is a MODEL OUTPUT, not a column of the dataset. Ground
truth edges would be cheaper and wrong — inference has none, and this pipeline has already paid once for a
train/inference asymmetry.

COST: the velocity is one extra backbone forward per step (plus the extra frame decode the dataset does to
feed it), so a run with it on is roughly 50% slower per step. That is the price of the experiment.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import cast

import torch
from jaxtyping import Float
from torch import Tensor, nn

from celltrack.data.joint_dataset import PairSample
from celltrack.models.joint_model import JointModel
from celltrack.models.prior_velocity import PriorVelocity


@dataclass(frozen=True)
class PriorVelocitySource:
    """The sources' prior displacement for one training pair, and the head widening that lets it be read."""

    enabled: bool

    def widen(self, model: JointModel) -> None:
        """Append the velocity's columns to the head's input projection — a no-op while the feature is off.

        The new columns are exactly zero, so the widened layer's output is bitwise identical to the pretrained
        one at step 0: a warm start survives intact and the velocity is learned from zero gain upward rather
        than through a random projection into a converged head.

        ORDER MATTERS: this REPLACES the projection's parameter object, so it must happen after the pretrained
        weights are loaded and before the optimiser is constructed. An optimiser built first would hold the
        discarded projection and train nothing through it, in silence.
        """
        if not self.enabled:
            return
        projection = cast(nn.Linear, model.transformer.proj)
        model.transformer.proj = PriorVelocity.widen_linear_inputs(projection, PriorVelocity.DIM)

    def of(
        self, model: JointModel, sample: PairSample, precision: AbstractContextManager[object]
    ) -> Float[Tensor, "s 3"] | None:
        """Each source's expected incoming displacement over the `t-1 -> t` gap; `None` while the feature is off.

        The previous gap is scored by the model itself and soft-maxed over its sources — the scorer's own
        normalisation — so the history moves as the head trains and matches what inference could compute.
        It runs under `torch.no_grad`: a step optimises the pair being scored, never a two-gap chain. That
        forward passes ZERO velocity for its own sources, which both stops the recursion one step back and
        keeps the head's input width the one the widened projection expects.

        A pair with no predecessor — a video's first annotated frame — takes the exact zeros
        `expected_incoming` returns for an absent history, the same value an unmatched node degrades to, so
        "no history" and "no usable history" reach the head as one case rather than two.
        """
        if not self.enabled:
            return None
        sources = sample.source_centres.to(torch.float32)
        if sample.previous_frame is None or sample.previous_centres is None:
            return PriorVelocity.expected_incoming(sources)
        previous = sample.previous_centres.to(torch.float32)
        with torch.no_grad(), precision:
            logits = model.forward(
                sample.previous_frame,
                sample.frame_t,
                sample.previous_centres,
                sample.source_centres,
                torch.zeros_like(previous),
            ).edge_logits
        return PriorVelocity.expected_incoming(sources, previous, torch.softmax(logits.float(), dim=0))
