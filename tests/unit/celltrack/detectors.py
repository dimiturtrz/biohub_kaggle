from pathlib import Path

import pytest
from pydantic import ValidationError

from celltrack import detectors as detectors_module
from celltrack.detectors import (
    DETECTOR_NAMES,
    DetectorConfig,
    DetectorContext,
    _DogDetector,
    _LearnedDetector,
    _ScorerDetector,
)
from celltrack.tunet import DetectorRecipe
from core.geometry import Spacing


class _FakeUNet:
    """A stand-in TemporalUNetDetector whose device moves are no-ops, so a build mounts without weights."""

    def to(self, device: str) -> "_FakeUNet":
        return self

    def eval(self) -> "_FakeUNet":
        return self


def test_build(monkeypatch: pytest.MonkeyPatch):
    """`build` dispatches the name to its concrete detector, carrying the readout the config chose."""
    monkeypatch.setattr(
        detectors_module.TemporalUNetDetector,
        "from_pack",
        classmethod(lambda cls, weight, map_location: (_FakeUNet(), DetectorRecipe())),
    )
    context = DetectorContext(device="cpu", weights=(Path("pack_a"), Path("pack_b")), responses=Path("cache"))

    pilktunet = DetectorConfig(name="pilktunet", threshold=0.97).build(context)
    dog = DetectorConfig(name="dog", keep_per_video=5000).build(DetectorContext(device="cpu"))

    assert isinstance(pilktunet, _ScorerDetector)
    assert pilktunet.threshold == 0.97
    assert len(pilktunet.scorer.detectors) == 2  # both seeds mounted for the blend
    assert isinstance(dog, _DogDetector)
    assert dog.keep_per_video == 5000


def test_scorer_detector_nodes():
    """`_ScorerDetector.nodes` asks its scorer for centres at the fixed threshold."""

    class _Scorer:
        def nodes(self, video_key: str, path: Path, threshold: float) -> str:
            return f"{video_key}@{threshold}"

    detector = _ScorerDetector(scorer=_Scorer(), threshold=0.97)  # type: ignore[arg-type]
    assert detector.nodes("m", Path("m.zarr")) == "m@0.97"


def test_dog_detector_nodes(monkeypatch: pytest.MonkeyPatch):
    """`_DogDetector.nodes` builds the DoG detector at the video's spacing and detects its node budget."""

    class _Video:
        spacing = Spacing(z=1.0, y=1.0, x=1.0)
        timepoint_count = 3

        def normalised_frames(self, low: float, high: float) -> list[str]:
            return ["f"]

    captured: list[int] = []

    class _Dog:
        def __init__(self, spacing: Spacing, device: str) -> None:
            self.spacing = spacing

        def detect(self, frames: list[str], count: int, keep: int) -> str:
            captured.append(keep)
            return "graph"

    monkeypatch.setattr(detectors_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video()))
    monkeypatch.setattr(detectors_module, "DoGDetector", _Dog)

    assert _DogDetector(scales_um=(), keep_per_video=900, device="cpu").nodes("m", Path("m.zarr")) == "graph"
    assert captured == [900]


def test_learned_detector_nodes(monkeypatch: pytest.MonkeyPatch):
    """`_LearnedDetector.nodes` rebuilds the peak-picker at the video's spacing and detects above the threshold."""

    class _Video:
        spacing = Spacing(z=1.0, y=1.0, x=1.0)

    captured: list[float] = []

    class _Learned:
        def __init__(self, model: object, peaks: object, window: int, device: str) -> None:
            self.window = window

        def detect_above(self, video: object, threshold: float) -> str:
            captured.append(threshold)
            return "graph"

    monkeypatch.setattr(detectors_module.CellVideo, "from_ome_zarr", staticmethod(lambda path: _Video()))
    monkeypatch.setattr(detectors_module, "PeakExtractor", lambda spacing, scale, device: "peaks")
    monkeypatch.setattr(detectors_module, "LearnedDetector", _Learned)

    detector = _LearnedDetector(model=object(), threshold=0.95, scale_um=4.0, window=2, device="cpu")  # type: ignore[arg-type]
    assert detector.nodes("m", Path("m.zarr")) == "graph"
    assert captured == [0.95]


def test_build_names_the_whole_family():
    """Every registered name is dispatchable — the registry and the validator agree on the family."""
    assert set(DETECTOR_NAMES) == set(detectors_module._BUILDERS)


def test_build_rejects_an_unknown_name():
    """An unregistered name is rejected at construction (validated boundary), not deep inside build."""
    with pytest.raises(ValidationError):
        DetectorConfig(name="cellpose")


def test_config_rejects_an_out_of_range_threshold():
    """The readout threshold is a probability — a value at or past the bounds is caught at construction."""
    with pytest.raises(ValidationError):
        DetectorConfig(threshold=1.0)
