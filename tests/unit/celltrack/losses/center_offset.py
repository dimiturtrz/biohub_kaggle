import torch

from celltrack.losses.center_offset import CenterOffsetLoss
from celltrack.losses.offset_target import OffsetTarget

_SHAPE = (6, 12, 12)
_ISOTROPIC = (1.0, 1.0, 1.0)
_RADIUS = (1, 3, 3)


def test_center_offset_loss_of():
    centres = [torch.tensor([[3, 6, 6]])]
    target = OffsetTarget.of(centres[0], _SHAPE, _RADIUS, _ISOTROPIC)
    perfect = target.offsets[None]

    assert CenterOffsetLoss.of(perfect, centres, _RADIUS, _ISOTROPIC) == 0.0
    assert CenterOffsetLoss.of(torch.zeros_like(perfect), centres, _RADIUS, _ISOTROPIC) > 0.0


def test_loss_ignores_error_outside_the_supervised_box():
    centres = [torch.tensor([[3, 6, 6]])]
    target = OffsetTarget.of(centres[0], _SHAPE, _RADIUS, _ISOTROPIC)
    wrong_far_away = target.offsets[None].clone()
    wrong_far_away[0, :, 0, 0, 0] = 99.0

    assert CenterOffsetLoss.of(wrong_far_away, centres, _RADIUS, _ISOTROPIC) == 0.0


def test_an_unannotated_frame_contributes_no_loss_but_keeps_the_graph():
    empty = [torch.zeros((0, 3), dtype=torch.long)]
    predicted = torch.zeros((1, 3, *_SHAPE), requires_grad=True)

    loss = CenterOffsetLoss.of(predicted, empty, _RADIUS, _ISOTROPIC)

    assert loss == 0.0
    assert loss.requires_grad
