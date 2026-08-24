"""Unit tests for SIGReg — the negative-free anti-collapse objective.

The load-bearing claim is that the Gaussianity term reads collapse: a real isotropic-Gaussian batch scores LOW,
a collapsed (constant) batch scores HIGH, and two identical views score zero invariance. The fixtures are drawn
tensors with a known distribution, because that is exactly what the Cramér-Wold statistic measures. A seed is
fixed so the Monte-Carlo direction draw does not make the test flaky.
"""

import pytest
import torch

from celltrack.losses.sigreg import SigReg, SigRegConfig


def test_forward():
    """The total is invariance plus weighted Gaussianity, and identical views contribute zero invariance."""
    torch.manual_seed(0)
    sigreg = SigReg(SigRegConfig(weight=1.0, directions=64))
    views = torch.randn(128, 32)

    result = sigreg.forward(views, views.clone())

    assert result.invariance == pytest.approx(0.0, abs=1e-6)  # two identical views embed alike
    assert result.total == pytest.approx(float(result.gaussianity), abs=1e-6)  # invariance zero → total is the spread


def test_gaussianity():
    """A real isotropic-Gaussian batch scores far lower than a collapsed constant batch — collapse is caught."""
    torch.manual_seed(0)
    sigreg = SigReg(SigRegConfig(directions=64))

    gaussian = float(sigreg._gaussianity(torch.randn(512, 32)))
    collapsed = float(sigreg._gaussianity(torch.zeros(512, 32)))

    assert collapsed > gaussian  # the whole point: a degenerate distribution is penalised more than a real Gaussian
    assert gaussian < 0.05  # a genuine N(0, I) sits near the target


def test_cf_distance():
    """A standard-normal projection reads near zero; a shifted one reads positive through the imaginary part."""
    torch.manual_seed(0)
    sigreg = SigReg(SigRegConfig())
    centred = torch.randn(4096, 1)

    near = float(sigreg._cf_distance(centred)[0])
    shifted = float(sigreg._cf_distance(centred + 3.0)[0])

    assert near < shifted  # a mean shift departs from N(0, 1) and the CF distance sees it
