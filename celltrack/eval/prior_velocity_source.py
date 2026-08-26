"""Where a training step's prior velocity comes from — the model's own affinity on the previous gap.

`PriorVelocity` defines what the feature IS (the expected incoming displacement, and the zero-init widening
that keeps a warm start exact). This is the training-time half: the history has to be produced the same way
the deployed scorer would produce it, which means it is a MODEL OUTPUT, not a column of the dataset. Ground
truth edges would be cheaper and wrong — inference has none, and this pipeline has already paid once for a
train/inference asymmetry.

COST: the velocity is one extra backbone forward per step (plus the extra frame decode the dataset does to
feed it). `of_batch` forwards the whole step's previous gaps in ONE batched backbone call, so the extra
forward scales with the step, not the batch — a per-sample loop launched B tiny forwards and left the GPU
idle between them.
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
from celltrack.models.prior_velocity import GapHistory, PriorVelocity


@dataclass(frozen=True)
class PriorVelocitySource:
    """The sources' prior displacement for one training pair, and the head widening that lets it be read."""

    enabled: bool
    # Steps to bootstrap the velocity off GROUND-TRUTH `t-1 -> t` links before handing back to the model's own
    # affinity. Zero = off (the model-derived velocity is used unchanged from step 0). The from-scratch model's
    # prev-gap affinity is confident-WRONG early, so the velocity it derives points at the wrong cell and cannot
    # break the confusor it exists to fix — a chicken-egg. GT links carry the real motion while the head learns
    # to READ velocity; a linear anneal α = 1 − step/warmup hands the input distribution back to the
    # self-derived one it must match at inference, so no train/inference asymmetry survives past the warmup.
    gt_warmup_steps: int = 0

    def _gt_alpha(self, step: int | None) -> float:
        """The GT velocity's blend weight at `step`: 1 at step 0, linearly to 0 at `gt_warmup_steps`, then off."""
        if step is None or self.gt_warmup_steps <= 0 or step >= self.gt_warmup_steps:
            return 0.0
        return 1.0 - step / self.gt_warmup_steps

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
        """One pair's velocity — the `B = 1` case of `of_batch`, kept for the single-pair eval/diagnostic paths."""
        return self.of_batch(model, [sample], precision)[0]

    def of_batch(
        self,
        model: JointModel,
        samples: list[PairSample],
        precision: AbstractContextManager[object],
        step: int | None = None,
    ) -> list[Float[Tensor, "s 3"] | None]:
        """Each sample's sources' expected incoming displacement over `t-1 -> t`; `None` per sample when off.

        The previous gaps are scored by the model itself in ONE batched backbone forward — the expensive part
        the per-sample loop launched serially, leaving the GPU idle between tiny pairs — and the raw logits go
        into a `GapHistory`, the ONE definition of how a scored gap becomes a velocity, shared with the
        inference scorer, so a trained head is read the way it was trained rather than through a second copy of
        the same three lines that can drift. It runs under `torch.no_grad`: a step optimises the pair being
        scored, never a two-gap chain. Each previous-gap forward passes ZERO velocity for its own sources,
        which both stops the recursion one step back and keeps the head's input width the one the widened
        projection expects (so it stays on `forward_batch`'s per-pair-head branch, not the velocity-blind
        fully-padded path) — and it is what `EdgeTransformerScorer._history_after` reproduces at inference.

        A pair with no predecessor — a video's first annotated frame — takes the exact zeros an empty
        `GapHistory` yields, the same value an unmatched node degrades to, so "no history" and "no usable
        history" reach the head as one case rather than two, and never enters the batched forward.

        `forward_batch` over `B = 1` is `model.forward`, so a batched velocity equals the per-pair one it
        replaces — the equality `test_of_batch` pins.
        """
        results: list[Float[Tensor, "s 3"] | None] = [None] * len(samples)
        if not self.enabled:
            return results
        scored: list[int] = []
        windows: list[Tensor] = []
        source_positions: list[Tensor] = []
        target_positions: list[Tensor] = []
        velocities: list[Tensor | None] = []
        for index, sample in enumerate(samples):
            sources = sample.source_centres.to(torch.float32)
            if sample.previous_frame is None or sample.previous_centres is None:
                results[index] = GapHistory().velocity(sources)
                continue
            scored.append(index)
            windows.append(torch.stack([sample.previous_frame, sample.frame_t], dim=0))
            source_positions.append(sample.previous_centres)
            target_positions.append(sample.source_centres)
            velocities.append(torch.zeros_like(sample.previous_centres.to(torch.float32)))
        if not scored:
            return results
        with torch.no_grad(), precision:
            forwards = model.forward_batch(torch.stack(windows, dim=0), source_positions, target_positions, velocities)
        alpha = self._gt_alpha(step)
        for forward, index in zip(forwards, scored, strict=True):
            sample = samples[index]
            previous = sample.previous_centres.to(torch.float32)  # type: ignore[union-attr]
            sources = sample.source_centres.to(torch.float32)
            model_velocity = GapHistory(previous, forward.edge_logits).velocity(sources)
            results[index] = self._blend(model_velocity, sample, previous, sources, alpha)
        return results

    @staticmethod
    def _blend(
        model_velocity: Float[Tensor, "s 3"],
        sample: PairSample,
        previous: Float[Tensor, "p 3"],
        sources: Float[Tensor, "s 3"],
        alpha: float,
    ) -> Float[Tensor, "s 3"]:
        """Mix the GT-link velocity into the model's by `alpha` — the model's own value untouched when `alpha == 0`.

        The GT term goes straight through `expected_incoming` WITHOUT a softmax: the link matrix is already a
        valid incoming-mass column (one parent → mass 1, no parent → mass 0), so an unparented source degrades
        to the exact zero velocity it should, where a softmax would spread full mass over every candidate and
        manufacture a confident wrong direction — the very failure the warmup exists to avoid.
        """
        if alpha <= 0.0 or sample.previous_edge_matrix is None:
            return model_velocity
        links = sample.previous_edge_matrix.to(sources.dtype)
        gt_velocity = PriorVelocity.expected_incoming(sources, previous, links)
        return alpha * gt_velocity + (1.0 - alpha) * model_velocity
