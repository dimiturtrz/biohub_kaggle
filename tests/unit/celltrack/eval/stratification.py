import torch

from celltrack.eval.stratification import PrecisionRow, Stratification


def test_resolves_confusor():
    assert PrecisionRow("near", 1.0, 10).resolves_confusor()
    assert not PrecisionRow("far", 2.9, 10).resolves_confusor()


def test_add():
    accumulator = Stratification(["a", "b"])
    error = torch.tensor([1.0, 3.0, 99.0])
    index = torch.tensor([0, 1, 1])
    supervised = torch.tensor([True, True, False])

    accumulator.add(error, index, supervised)
    rows = accumulator.rows()

    assert [row.error_um for row in rows] == [1.0, 3.0]
    assert [row.voxels for row in rows] == [1, 1]


def test_rows():
    assert Stratification(["a"]).rows()[0].voxels == 0
