import torch

from celltrack.training.classification_loss import MaskedBCEClassificationLoss

LOSS = MaskedBCEClassificationLoss()


def _target() -> torch.Tensor:
    target = torch.zeros(1, 2, 4, 4)
    target[0, 1, 2, 2] = 1.0
    return target


def test_forward():
    """A confident, correct classification scores near zero; inverting the logits scores worse."""
    target = _target()
    mask = torch.ones(1, 2, 4, 4, dtype=torch.bool)
    confident = torch.full((1, 2, 4, 4), -20.0)
    confident[0, 1, 2, 2] = 20.0
    assert LOSS.forward(confident, target, mask).item() < 1e-3
    assert LOSS.forward(-confident, target, mask).item() > LOSS.forward(confident, target, mask).item()


def test_forward_ignores_unmasked_error():
    """A large error outside the mask does not change the loss."""
    target = _target()
    mask = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
    mask[0, 1, 2, 2], mask[0, 0, 0, 0] = True, True
    prediction = torch.zeros(1, 2, 4, 4)
    prediction_far = prediction.clone()
    prediction_far[0, 0, 3, 3] = 999.0
    assert LOSS.forward(prediction, target, mask).item() == LOSS.forward(prediction_far, target, mask).item()


def test_forward_zeroes_gradient_outside_the_mask():
    """Masked-out voxels carry no gradient — an un-annotated cell is never taught to be background."""
    target = torch.zeros(1, 3, 3, 3)
    target[0, 1, 1, 1] = 1.0
    mask = torch.zeros(1, 3, 3, 3, dtype=torch.bool)
    mask[0, 1, 1, 1] = True
    prediction = torch.zeros(1, 3, 3, 3, requires_grad=True)
    LOSS.forward(prediction, target, mask).backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 1, 1, 1] != 0.0
    assert torch.count_nonzero(prediction.grad[~mask]) == 0
