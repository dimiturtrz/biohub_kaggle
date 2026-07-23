import torch

from celltrack.training.model import ANISOTROPIC_STRIDES, Checkpoint, DetectionUNet


def test_from_checkpoint():
    """A checkpoint rebuilds the same architecture and restores its weights."""
    original = DetectionUNet(window=1, width=4, strides=((2, 2, 2),))
    checkpoint = Checkpoint(state_dict=original.state_dict(), strides=[[2, 2, 2]], window=1, width=4)
    rebuilt = DetectionUNet.from_checkpoint(checkpoint)
    assert torch.equal(rebuilt.head.weight, original.head.weight)


def test_forward():
    """The shallow isotropic U-Net maps a temporal window to one heatmap at the input resolution."""
    net = DetectionUNet(window=3, width=8)
    out = net.forward(torch.zeros(2, 3, 8, 16, 16))
    assert out.shape == (2, 8, 16, 16)


def test_forward_with_anisotropic_pooling():
    """The deep anisotropy-aware detector pools y/x more than z, still returning the input resolution."""
    net = DetectionUNet(window=3, width=4, strides=ANISOTROPIC_STRIDES)
    out = net.forward(torch.zeros(1, 3, 8, 32, 32))
    assert out.shape == (1, 8, 32, 32)
