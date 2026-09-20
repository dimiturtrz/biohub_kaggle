import pytest
import torch

from celltrack.losses.offset_target import OffsetTarget

_SHAPE = (6, 12, 12)
_ISOTROPIC = (1.0, 1.0, 1.0)
_RADIUS = (1, 3, 3)


def test_offset_target_of():
    centres = torch.tensor([[3, 6, 6]])
    target = OffsetTarget.of(centres, _SHAPE, _RADIUS, _ISOTROPIC)

    assert target.offsets[:, 3, 6, 6].abs().sum() == 0.0
    assert torch.equal(target.offsets[:, 3, 6, 9], torch.tensor([0.0, 0.0, -3.0]))
    assert torch.equal(target.offsets[:, 2, 4, 6], torch.tensor([1.0, 2.0, 0.0]))


def test_supervision_is_the_anisotropic_box_and_nothing_else():
    target = OffsetTarget.of(torch.tensor([[3, 6, 6]]), _SHAPE, _RADIUS, _ISOTROPIC)

    box = (2 * 1 + 1) * (2 * 3 + 1) * (2 * 3 + 1)
    assert target.supervised.sum() == box
    assert target.supervised[3, 6, 6] == 1.0
    assert target.supervised[0, 0, 0] == 0.0


def test_scale_makes_the_regressed_quantity_microns_not_voxels():
    anisotropic = OffsetTarget.of(torch.tensor([[3, 6, 6]]), _SHAPE, _RADIUS, (4.0, 1.0, 1.0))

    assert torch.equal(anisotropic.offsets[:, 2, 6, 6], torch.tensor([4.0, 0.0, 0.0]))


def test_overlapping_windows_split_at_the_voronoi_boundary():
    """A voxel covered by two centres' windows takes the nearer centre, not the later-written one."""
    target = OffsetTarget.of(torch.tensor([[3, 6, 4], [3, 6, 8]]), _SHAPE, _RADIUS, _ISOTROPIC)

    assert torch.equal(target.offsets[:, 3, 6, 5], torch.tensor([0.0, 0.0, -1.0]))
    assert torch.equal(target.offsets[:, 3, 6, 7], torch.tensor([0.0, 0.0, 1.0]))


def test_windows_clip_at_the_volume_edge_without_wrapping():
    target = OffsetTarget.of(torch.tensor([[0, 0, 0]]), _SHAPE, _RADIUS, _ISOTROPIC)

    assert target.supervised.sum() == 2 * 4 * 4
    assert target.supervised[-1, -1, -1] == 0.0


def test_supervised_fraction():
    """The share of the volume carrying the term — the box around one centre, not the whole frame."""
    target = OffsetTarget.of(torch.tensor([[3, 6, 6]]), _SHAPE, _RADIUS, _ISOTROPIC)

    box = (2 * 1 + 1) * (2 * 3 + 1) * (2 * 3 + 1)
    assert target.supervised_fraction() == pytest.approx(box / (6 * 12 * 12))
