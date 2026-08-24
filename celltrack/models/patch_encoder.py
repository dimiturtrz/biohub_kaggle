"""A per-cell appearance encoder trained on RAW voxels — the discriminator the detection trunk never learned.

WHY A SEPARATE ENCODER, not the shared node features. The measured association failure is a discriminator gap:
on the dense movie the head prefers a wrong near neighbour to the true far successor, and every learned-feature
probe reads at chance (`appearance_identity`). Those features come from the detection trunk, which is trained to
answer "is this a cell" — an INVARIANCE, every cell should look alike — so identity is exactly the axis it is
optimised to discard. Contrasting the trunk features directly buys discrimination out of that invariance and
regresses detection (measured proxy 0.8414 -> 0.7292). The way out is a SECOND representation with no detection
job: an encoder that reads the raw box around a cell and is free to be maximally discriminative because nothing
asks it to be invariant.

WHY RAW BOXES, not the pooled map. The pooled representation is `(1, 4, 4)`-downsampled, where a nucleus spans
two or three voxels and the confusor cells sit ~1.6 um apart — one voxel. The full-resolution image is 16x finer
in plane, so whatever separates these cells lives there or nowhere; `appearance_identity` asks exactly that of
raw correlation and finds it too weak, which is the case for a LEARNED read of the same voxels rather than a
fixed one.

The encoder is small and anisotropic-aware: plane is isotropic and fine, z is 4x coarser, so the plane is
downsampled harder than z on the way to a single pooled vector. It carries no classifier and no reconstruction
head — its only output is the embedding a contrastive/`SIGReg` objective shapes and the edge head then reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from jaxtyping import Float
from torch import Tensor, nn


@dataclass(frozen=True)
class PatchEncoderConfig:
    """The encoder's shape — its channel schedule and the width of the embedding it emits.

    `embedding_dim` is the one number the objective and the edge head both agree on. It is bounded below by the
    Johnson-Lindenstrauss angle-preservation bound for the ~1e3 candidates contrasted per frame (~220 dims at
    distortion 0.5) and above only by cost; 64 sits an order under the bound yet an order over the 32-channel
    trunk block it replaces, so it can hold identity the trunk cannot without being a rank-deficient lift.
    """

    embedding_dim: int = 64
    channels: tuple[int, ...] = (16, 32, 64)
    in_channels: int = 1

    def __post_init__(self) -> None:
        if not self.channels:
            msg = "patch encoder needs at least one convolution stage"
            raise ValueError(msg)


class PatchEncoder(nn.Module):
    """Raw box `(b, 1, z, y, x)` -> embedding `(b, d)` in R^d; identity, not detection.

    Each stage is a padded 3x3x3 convolution, group norm and a GELU, followed by an anisotropic pool that halves
    the plane every stage and z only on the later stages — the plane is fine and isotropic so it can afford the
    reduction, while z is coarse and a few planes cannot survive being halved from the start. A global average
    over whatever spatial extent survives makes the encoder accept the ragged full-resolution box sizes that
    `RawBoxes` hands it (odd, per-axis, spacing-derived) without a fixed input shape, and a final linear projects
    the pooled channels to the embedding width.

    The output is NOT L2-normalised. `SIGReg` shapes the batch towards an isotropic Gaussian in R^d, which a
    fixed-norm sphere cannot be, so the embedding is left in the plane where that objective is defined; cosine —
    for the confusor gate and for the edge head — normalises at read time, where a direction is what is wanted.
    """

    def __init__(self, config: PatchEncoderConfig) -> None:
        super().__init__()
        self.config = config
        stages: list[nn.Module] = []
        channels_in = config.in_channels
        for stage, channels_out in enumerate(config.channels):
            stages.append(self._stage(channels_in, channels_out, pool_z=stage >= 1))
            channels_in = channels_out
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.project = nn.Linear(channels_in, config.embedding_dim)

    @staticmethod
    def _stage(channels_in: int, channels_out: int, *, pool_z: bool) -> nn.Module:
        """One conv-norm-act block and its anisotropic pool — the plane always halves, z only when `pool_z`."""
        groups = min(8, channels_out)
        return nn.Sequential(
            nn.Conv3d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels_out),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2 if pool_z else 1, 2, 2)),
        )

    @property
    def embedding_dim(self) -> int:
        """The width of the embedding `forward` returns — what the objective and the edge head must expect."""
        return self.config.embedding_dim

    @override
    def forward(self, boxes: Float[Tensor, "b c z y x"]) -> Float[Tensor, "b d"]:
        """Encode a batch of raw boxes to identity embeddings in R^d (un-normalised; see class)."""
        pooled = self.pool(self.features(boxes)).flatten(1)
        return self.project(pooled)
