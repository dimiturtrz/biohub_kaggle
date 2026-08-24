"""SIGReg — keep the embeddings spread as an isotropic Gaussian WITHOUT negatives, so cells push apart for free.

THE COLLAPSE PROBLEM a contrastive objective solves with negatives. An invariance term alone (two views of a
cell should embed alike) is minimised by mapping every cell to the same point — perfect invariance, zero
identity. The usual fix is explicit negatives: pull other cells away. But the confusor is a HARD-negative problem
(the rival is the nearest cell, by construction), so sampling negatives well is exactly the part that is hard,
and getting it wrong is what the existing InfoNCE stack traded detection away on.

SIGReg (the anti-collapse half of LeJEPA) removes the negatives entirely. Instead of pushing named pairs apart it
constrains the whole batch's embedding distribution to be an isotropic Gaussian — the unique maximum-entropy
distribution at fixed scale, so matching it spreads the embeddings as far as they can go in every direction at
once. Different cells then separate as a CONSEQUENCE of the distribution being full-rank, never by naming a rival.
One term, one weight; no negative sampling, no temperature, no teacher.

HOW the match is measured — Cramér-Wold. A distribution in R^d is the isotropic Gaussian iff EVERY one-dimensional
projection of it is a standard 1D Gaussian. So SIGReg draws random unit directions, projects the batch onto each,
and penalises each projection's departure from `N(0, 1)`. The 1D test is a characteristic-function distance (the
Epps-Pulley statistic): the empirical CF `mean(exp(i t s))` against the Gaussian's `exp(-t^2 / 2)` over a grid of
frequencies. It is smooth, differentiable, and reads mean, variance and every higher moment at once — a shifted
projection shows up as a phase (imaginary part), a mis-scaled one as the wrong decay — so no separate moment
terms are needed. Redrawing the directions every step is the Monte-Carlo estimate of the full Cramér-Wold
integral; a fixed set would let the encoder be Gaussian along those axes and collapse between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, override

import torch
from jaxtyping import Float
from torch import Tensor, nn


@dataclass(frozen=True)
class SigRegConfig:
    """SIGReg's knobs. `weight` is the ONE hyperparameter that matters — the balance against invariance.

    `directions` is the Monte-Carlo sample count for the Cramér-Wold integral and `frequencies` the CF grid; both
    are estimator resolution, traded against cost, not tuning surface — more of either lowers variance and changes
    no optimum. The default frequency band spans the range where a unit Gaussian's CF carries its mass
    (`exp(-t^2/2)` is ~1 at 0 and ~0.0001 by t=4), so a projection's low and high moments both leave a signature.
    """

    weight: float = 1.0
    directions: int = 64
    frequencies: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


@dataclass(frozen=True)
class SigRegResult:
    """The total and its two readable halves — the trainer logs which one is moving.

    Invariance falling while the Gaussianity term holds is the objective working; invariance falling while
    Gaussianity climbs is the collapse SIGReg is there to forbid, caught in the log before it reaches the gate.
    """

    total: Float[Tensor, ""]
    invariance: Float[Tensor, ""]
    gaussianity: Float[Tensor, ""]


class SigReg(nn.Module):
    """Invariance (two views embed alike) plus the negative-free Gaussianity spread — a module for its device grid.

    A module rather than a function only to own the frequency grid as a buffer, so it rides `.to(device)` with the
    model and the CF distance never silently computes on the wrong device. It holds no trained parameters.
    """

    def __init__(self, config: SigRegConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("frequencies", torch.tensor(config.frequencies, dtype=torch.float32))

    @override
    def forward(self, view_one: Float[Tensor, "b d"], view_two: Float[Tensor, "b d"]) -> SigRegResult:
        """The SIGReg objective for two encoded views of the same batch of cells."""
        invariance = (view_one - view_two).pow(2).sum(dim=-1).mean()
        gaussianity = self._gaussianity(torch.cat([view_one, view_two], dim=0))
        total = invariance + self.config.weight * gaussianity
        return SigRegResult(total=total, invariance=invariance, gaussianity=gaussianity)

    def _gaussianity(self, embeddings: Float[Tensor, "n d"]) -> Float[Tensor, ""]:
        """Cramér-Wold departure from isotropic Gaussian: mean CF distance over random 1D projections."""
        directions = self._directions(embeddings.shape[-1], embeddings.device, embeddings.dtype)
        projections = embeddings @ directions.T  # (n, k)
        return self._cf_distance(projections).mean()

    def _directions(self, dim: int, device: torch.device, dtype: torch.dtype) -> Float[Tensor, "k d"]:
        """Fresh unit directions each call — the Monte-Carlo draw that makes the estimate cover all of R^d."""
        raw = torch.randn(self.config.directions, dim, device=device, dtype=dtype)
        return raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _cf_distance(self, projections: Float[Tensor, "n k"]) -> Float[Tensor, "k"]:
        """Per-projection Epps-Pulley distance of the sample to `N(0, 1)`, over the frequency grid.

        No standardisation of the projections first: leaving the raw mean and variance in is what lets the term
        pull them to 0 and 1 — a centred, rescaled sample would hide exactly the collapse (zero variance) the
        objective must see.
        """
        frequencies = cast(Tensor, self.frequencies).to(projections.dtype)  # a registered buffer is Tensor | Module
        phase = projections.unsqueeze(-1) * frequencies  # (n, k, f)
        empirical_real = phase.cos().mean(dim=0)  # (k, f)
        empirical_imag = phase.sin().mean(dim=0)
        target_real = torch.exp(-0.5 * frequencies.pow(2))  # (f,)
        return ((empirical_real - target_real).pow(2) + empirical_imag.pow(2)).mean(dim=-1)
