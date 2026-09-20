from pathlib import Path
from typing import cast

import pytest
import torch

from celltrack.data.tunet_dataset import FrameTarget
from celltrack.detectors.detection import CELL_SCALE_UM
from celltrack.detectors.tunet import TemporalUNetDetector
from celltrack.eval.kernel_reference import VOXEL_SCALE_UM
from celltrack.eval.offset_precision import FramePrediction, OffsetPrecision
from celltrack.eval.stratification import PrecisionRow
from celltrack.losses.offset_target import OffsetTarget
from core.paths import DataRoot

_SHAPE = (6, 12, 12)
_ISOTROPIC = (1.0, 1.0, 1.0)
_RADIUS = (2, 2, 2)
_UNIT_DOWNSAMPLE = (1, 1, 1)


def test_shape():
    prediction = FramePrediction(torch.zeros(_SHAPE), torch.zeros((3, *_SHAPE)), torch.tensor([[3, 6, 6]]))

    assert prediction.shape() == _SHAPE


def test_of():
    """The target is rebuilt inside `of` from the downsample, so the fixture must use the same voxel scale."""
    scale = cast(tuple[float, float, float], VOXEL_SCALE_UM)
    radius = cast(tuple[int, int, int], tuple(max(1, round(CELL_SCALE_UM / size)) for size in scale))
    centres = torch.tensor([[3, 6, 6]])
    target = OffsetTarget.of(centres, _SHAPE, radius, scale)
    prediction = FramePrediction(torch.zeros(_SHAPE), target.offsets, centres)

    precision = OffsetPrecision.of([prediction], _UNIT_DOWNSAMPLE)

    assert precision.overall.error_um == 0.0
    assert precision.overall.voxels == int(target.supervised.sum())
    assert sum(row.voxels for row in precision.confidence) == precision.overall.voxels


def test_of_skips_a_frame_with_no_annotated_centres():
    empty = FramePrediction(torch.zeros(_SHAPE), torch.zeros((3, *_SHAPE)), torch.zeros((0, 3), dtype=torch.long))

    assert OffsetPrecision.of([empty], _UNIT_DOWNSAMPLE).overall.voxels == 0


def test_report(caplog: pytest.LogCaptureFixture):
    row = PrecisionRow("all", 1.0, 5)

    with caplog.at_level("INFO"):
        OffsetPrecision([row], [row], row).report()

    assert len(caplog.records) == 3


def test_ring_count_reaches_the_box_corner_rather_than_ending_at_the_axis():
    """An axis-length ring count would dump the whole diagonal into the outermost bucket."""
    assert OffsetPrecision._ring_count(_RADIUS, _ISOTROPIC) > max(_RADIUS)


def test_ring_index_puts_the_centre_voxel_in_ring_zero_and_the_corner_in_the_last():
    target = OffsetTarget.of(torch.tensor([[3, 6, 6]]), _SHAPE, _RADIUS, _ISOTROPIC)
    rings = OffsetPrecision._ring_count(_RADIUS, _ISOTROPIC)

    index = OffsetPrecision._ring_index(target, rings, 1.0)

    assert index[3, 6, 6] == 0
    assert index[1, 4, 4] == rings - 1


def test_confidence_index_splits_at_the_readout_thresholds():
    logits = torch.tensor([-10.0, 0.0, 10.0])

    assert torch.equal(OffsetPrecision._confidence_index(logits), torch.tensor([0, 1, 2]))


def test_stream(video_store: Path, geff_store: Path):
    """Forwarding the indexed frames yields one prediction per frame, with the offsets head installed."""
    detector = TemporalUNetDetector(8, (8, 16), offset_head=True)
    targets = FrameTarget.index(DataRoot(geff_store.parent), [video_store], downsample=(1, 1, 1))

    predictions = list(FramePrediction.stream(detector, targets, (1, 1, 1), "cpu", frames=2))

    assert len(predictions) == 2
    assert predictions[0].shape() == (2, 4, 4)
    assert predictions[0].offsets.shape == (3, 2, 4, 4)
