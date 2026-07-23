import torch

from celltrack.training.model import DetectionUNet


def test_forward():
    """The U-Net maps a temporal window to one heatmap at the input resolution."""
    net = DetectionUNet(window=3, width=8)
    out = net.forward(torch.zeros(2, 3, 8, 16, 16))
    assert out.shape == (2, 8, 16, 16)
