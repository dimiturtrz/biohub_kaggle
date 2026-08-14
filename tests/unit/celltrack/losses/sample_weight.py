import torch

from celltrack.losses.sample_weight import SampleWeight

torch.manual_seed(0)  # deterministic tensors so a failure reproduces


def test_of():
    """A bright, isolated source outweighs a dim, crowded one — reliability rises with SNR, falls with crowd."""
    frame = torch.zeros(8, 8, 8)
    frame[4, 4, 4] = 10.0  # a bright isolated peak
    frame[4, 4, 6] = 1.0  # a dim one

    bright_isolated = SampleWeight.of(
        frame,
        source_grid=torch.tensor([[4, 4, 4]]),
        target_um=torch.tensor([[100.0, 100.0, 100.0]]),  # its only candidate is far — uncrowded
        source_um=torch.tensor([[0.0, 0.0, 0.0]]),
        gate_um=10.0,
    )
    dim_crowded = SampleWeight.of(
        frame,
        source_grid=torch.tensor([[4, 4, 6]]),
        target_um=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),  # three rivals in gate
        source_um=torch.tensor([[0.0, 0.0, 0.0]]),
        gate_um=10.0,
    )
    assert bright_isolated > dim_crowded


def test_of_empty_sources():
    """A pair with no annotated sources carries no supervision, so it weighs exactly zero."""
    frame = torch.ones(4, 4, 4)
    weight = SampleWeight.of(
        frame,
        source_grid=torch.zeros(0, 3, dtype=torch.long),
        target_um=torch.zeros(0, 3),
        source_um=torch.zeros(0, 3),
        gate_um=10.0,
    )
    assert weight.item() == 0.0


def test_of_crowd_lowers_weight():
    """Adding in-gate rivals to an otherwise identical source strictly lowers its weight (the 1/(1+crowd) term)."""
    frame = torch.zeros(8, 8, 8)
    frame[4, 4, 4] = 5.0
    args = {"frame": frame, "source_grid": torch.tensor([[4, 4, 4]]), "source_um": torch.tensor([[0.0, 0.0, 0.0]])}
    alone = SampleWeight.of(**args, target_um=torch.tensor([[50.0, 0.0, 0.0]]), gate_um=10.0)
    crowded = SampleWeight.of(**args, target_um=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), gate_um=10.0)
    assert crowded < alone
