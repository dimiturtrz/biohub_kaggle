"""Running error sums per stratum — so a whole run of frames needs no per-frame storage."""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# The separation the head has to resolve to be worth a readout: one (1,4,4) voxel in y/x, which is the
# distance a confusor pair sits apart (see the confusor interpretation — pairwise-unresolvable at 1.6um).
_CONFUSOR_SEPARATION_UM = 1.625


@dataclass(frozen=True)
class PrecisionRow:
    """One stratum's offset error — what the head achieves on that slice of voxels, and how many there are."""

    label: str
    error_um: float
    voxels: int

    def resolves_confusor(self) -> bool:
        """Whether this stratum's error is under the separation a readout must resolve to split a merged blob."""
        return self.error_um < _CONFUSOR_SEPARATION_UM


class Stratification:
    """Running error sums and counts for one stratification, so a whole run needs no per-frame storage."""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.sums = torch.zeros(len(labels), dtype=torch.float64)
        self.counts = torch.zeros(len(labels), dtype=torch.float64)

    def add(
        self,
        error: Float[Tensor, "*shape"],
        index: Int[Tensor, "*shape"],
        supervised: Bool[Tensor, "*shape"],
    ) -> None:
        flat_index = index[supervised].flatten().cpu()
        flat_error = error[supervised].flatten().double().cpu()
        self.sums += torch.zeros_like(self.sums).scatter_add_(0, flat_index, flat_error)
        self.counts += torch.zeros_like(self.counts).scatter_add_(0, flat_index, torch.ones_like(flat_error))

    def rows(self) -> list[PrecisionRow]:
        return [
            PrecisionRow(label, float(total / count) if count else 0.0, int(count))
            for label, total, count in zip(self.labels, self.sums, self.counts, strict=True)
        ]
