import torch

from celltrack.training.loss import MaskedDetectionLoss

LOSS = MaskedDetectionLoss()


def test_forward():
    """The loss is the mean squared error over the supervised voxels only."""
    prediction = torch.zeros(1, 2, 2, 2)
    target = torch.ones(1, 2, 2, 2)
    mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    assert LOSS.forward(prediction, target, mask).item() == 1.0


def test_forward_ignores_unmasked_error():
    """A large error outside the mask does not change the loss."""
    prediction = torch.zeros(1, 2, 2, 2)
    target = torch.ones(1, 2, 2, 2)
    mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    prediction_far = prediction.clone()
    prediction_far[0, 1, 1, 1] = 999.0
    assert LOSS.forward(prediction, target, mask).item() == LOSS.forward(prediction_far, target, mask).item()


def test_forward_zeroes_gradient_outside_the_mask():
    """The load-bearing property: unlabelled (masked-out) voxels carry no gradient, so they never train.

    A masked-out bright blob is a real but un-annotated cell; if it received gradient it would be pushed
    toward background, teaching the model to suppress genuine cells.
    """
    prediction = torch.zeros(1, 3, 3, 3, requires_grad=True)
    target = torch.ones(1, 3, 3, 3)
    mask = torch.zeros(1, 3, 3, 3, dtype=torch.bool)
    mask[0, 1, 1, 1] = True
    LOSS.forward(prediction, target, mask).backward()
    gradient = prediction.grad
    assert gradient[0, 1, 1, 1] != 0.0
    assert torch.count_nonzero(gradient[~mask]) == 0
