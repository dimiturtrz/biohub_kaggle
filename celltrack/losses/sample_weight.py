"""Per-sample reliability weight — trust clean examples, let the model stay uncertain on hard ones.

A sparse-annotation detection loss and a crowded-neighbourhood association loss both punish the model hardest
exactly where the data is least decidable: a low-contrast cell the annotator may have missed, a source with
many near-equal candidates where the true successor is genuinely ambiguous at this resolution. Forcing
confidence there fits noise. This weights each sample's loss by how RELIABLE its supervision is — high for a
bright, isolated cell, low for a dim, crowded one — so the gradient trusts the clean cases and eases off the
ambiguous ones. It is a soft LOSS (it scales the gradient contribution), not a soft LOGIT (it never touches
what the head outputs); the two are independent and this is only the former.

The weight is `SNR / (1 + crowd)`: local contrast in the numerator (bright, well-separated peaks are
trustworthy), the in-gate candidate count in the denominator (a source contested by many rivals carries less
certain supervision). One formula for every sample — synthetic and real alike, no special-casing, because a
branch there is more logic for a worse expected result. The raw per-pair scalar is normalised to batch-mean 1
by the caller, so the weighting redistributes emphasis WITHOUT rescaling the overall loss (learning-rate
invariant).
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor

_EPS = 1e-6
_CONTRAST_RADIUS = 1  # a ±1-voxel window: the peak against its immediate neighbourhood, the detector's own scale


class SampleWeight:
    """The reliability weight of one pair — a namespace over the SNR and crowd terms, no instance state."""

    @staticmethod
    def of(
        frame: Float[Tensor, "z y x"],
        source_grid: Int[Tensor, "s 3"],
        target_um: Float[Tensor, "u 3"],
        source_um: Float[Tensor, "s 3"],
        gate_um: float,
    ) -> Float[Tensor, ""]:
        """One pair's reliability scalar — the mean over its sources of `SNR / (1 + crowd)`.

        `source_grid` indexes `frame` (the heads' downsampled grid); `source_um`/`target_um` are micrometre
        positions for the gate. A pair with no sources is maximally uninformative and weighs zero.
        """
        if source_grid.shape[0] == 0:
            return frame.new_zeros(())
        snr = SampleWeight._snr(frame, source_grid)
        crowd = SampleWeight._crowd(source_um, target_um, gate_um)
        return (snr / (1.0 + crowd)).mean()

    @staticmethod
    def _snr(frame: Float[Tensor, "z y x"], source_grid: Int[Tensor, "s 3"]) -> Float[Tensor, "s"]:
        """Local contrast at each source: centre intensity over the mean of its ±radius window, clamped ≥ 0."""
        shape = torch.tensor(frame.shape, device=frame.device)
        clamped = source_grid.clamp(min=torch.zeros(3, dtype=torch.long, device=frame.device), max=shape - 1)
        centre = frame[clamped[:, 0], clamped[:, 1], clamped[:, 2]]
        background = SampleWeight._window_mean(frame, clamped)
        return (centre / (background + _EPS)).clamp(min=0.0)

    @staticmethod
    def _window_mean(frame: Float[Tensor, "z y x"], grid: Int[Tensor, "s 3"]) -> Float[Tensor, "s"]:
        """Mean intensity of the ±`_CONTRAST_RADIUS` cube around each grid point, edges clamped into the volume."""
        shape = torch.tensor(frame.shape, device=frame.device)
        offsets = torch.arange(-_CONTRAST_RADIUS, _CONTRAST_RADIUS + 1, device=frame.device)
        grid_z, grid_y, grid_x = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
        window = torch.stack([grid_z.reshape(-1), grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)  # (27, 3)
        neighbours = (grid[:, None, :] + window[None, :, :]).clamp(
            min=torch.zeros(3, dtype=torch.long, device=frame.device), max=shape - 1
        )  # (s, 27, 3)
        values = frame[neighbours[..., 0], neighbours[..., 1], neighbours[..., 2]]  # (s, 27)
        return values.mean(dim=1)

    @staticmethod
    def _crowd(source_um: Float[Tensor, "s 3"], target_um: Float[Tensor, "u 3"], gate_um: float) -> Float[Tensor, "s"]:
        """Candidate density per source: how many targets sit within the linker gate — the rivals it competes with."""
        if target_um.shape[0] == 0:
            return source_um.new_zeros(source_um.shape[0])
        distances = torch.cdist(source_um, target_um)  # (s, u) micrometres
        return (distances <= gate_um).sum(dim=1).to(source_um.dtype)
