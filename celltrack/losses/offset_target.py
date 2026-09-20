"""The Voronoi offset target: each voxel's micron vector to the annotated centre that owns it."""

import torch
from jaxtyping import Float, Int
from torch import Tensor

_AXES = 3


class OffsetTarget:
    """The Voronoi offset field for one frame, plus the mask of voxels close enough to a centre to supervise."""

    def __init__(self, offsets: Float[Tensor, "three z y x"], supervised: Float[Tensor, "z y x"]) -> None:
        self.offsets = offsets
        self.supervised = supervised

    def supervised_fraction(self) -> float:
        """What share of the volume carries the offset loss — the rest has no defensible owner."""
        return float(self.supervised.float().mean())

    @classmethod
    def of(
        cls,
        centres: Int[Tensor, "n 3"],
        shape: tuple[int, int, int],
        radius: tuple[int, int, int],
        scale: tuple[float, float, float],
    ) -> "OffsetTarget":
        """Each voxel's micron vector to the nearest annotated centre, within `radius` voxels of one.

        `radius` is the per-axis half-window in VOXELS (anisotropic: the z axis is coarser than y/x), and
        `scale` the microns per voxel on each axis, so the regressed quantity is physical and the same
        network output means the same displacement on either axis.

        Vectorised over centres rather than looped. Inside a centre's window the target offset is exactly the
        window's own grid offset, identical for every centre, so the only per-centre work is deciding which
        centre owns a voxel two windows both cover. That is a scatter-amin over the flattened volume on the
        micron distance, after which a voxel keeps the offset of whichever centre reached it at that distance.
        """
        device = centres.device
        grid = cls._window_offsets(radius, device)  # (w, 3) voxel offsets spanning one centre's box
        microns = grid * torch.tensor(scale, device=device, dtype=torch.float32)
        distances = microns.norm(dim=1)  # (w,) — the same distances in every centre's window
        voxels = centres[:, None, :] + grid[None].to(centres.dtype)  # (n, w, 3)
        extent = torch.tensor(shape, device=device)
        inside = ((voxels >= 0) & (voxels < extent)).all(dim=-1)  # (n, w) — windows clip at the volume's edge
        flat = cls._ravel(voxels.clamp(min=torch.zeros_like(extent), max=extent - 1), shape)[inside]  # (m,)
        reach = distances[None].expand(len(centres), -1)[inside]  # (m,) each candidate's distance to its centre
        nearest = torch.full((int(extent.prod()),), torch.inf, device=device).scatter_reduce(
            0, flat, reach, reduce="amin", include_self=True
        )
        # A voxel's owner is the centre that reached it at the winning distance. Exact-tie voxels (equidistant
        # between two centres) take whichever write lands last, which is the Voronoi boundary and arbitrary there.
        wins = reach == nearest[flat]
        offsets = torch.zeros((int(extent.prod()), _AXES), device=device)
        offsets[flat[wins]] = -microns[None].expand(len(centres), -1, -1)[inside][wins]
        supervised = nearest.isfinite()
        return cls(
            offsets.T.reshape(_AXES, *shape),
            supervised.reshape(shape).float(),
        )

    @staticmethod
    def _window_offsets(radius: tuple[int, int, int], device: torch.device) -> Float[Tensor, "w 3"]:
        """Every voxel offset in the anisotropic box `[-r, +r]` on each axis, as a flat list of (dz, dy, dx)."""
        axes = [torch.arange(-r, r + 1, device=device, dtype=torch.float32) for r in radius]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, _AXES)

    @staticmethod
    def _ravel(voxels: Int[Tensor, "n w 3"], shape: tuple[int, int, int]) -> Int[Tensor, "n w"]:
        """Flat indices into a volume of `shape` for (z, y, x) triples — the C-order ravel, on device."""
        z, y, x = shape
        return (voxels[..., 0] * y + voxels[..., 1]) * x + voxels[..., 2]
