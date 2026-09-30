"""Equivalence-class tests for the reference mounts' PATH resolution.

Mounting for real needs the packs and a checkpoint on disk, so what is testable offline is the part that was
duplicated and could drift: which paths each mount hands `CellTracker`. The tracker factories are stubbed, so
the classes are (arbitrary pack pair, the champion pilkwang pair, a joint checkpoint) and each asserts the
resolved arguments rather than a mounted object.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from celltrack import reference_mount
from celltrack.models.temporal_unet_detector import DetectorRecipe
from celltrack.operating_point import TrackerConfig
from celltrack.reference_mount import PILKWANG_PRIMARY, PILKWANG_SECONDARY, RESPONSE_CACHE, ReferenceMount

_PROC = Path("processed")


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Every argument the mounts pass to the tracker factories, with nothing loaded."""
    calls: dict[str, object] = {}

    def from_packs(pack1: Path, pack2: Path, responses: Path, device: str, config: TrackerConfig | None = None) -> str:
        calls.update(kind="packs", pack1=pack1, pack2=pack2, responses=responses, device=device, config=config)
        return "tracker"

    def from_joint(
        checkpoint: Path,
        recipe: DetectorRecipe,
        device: str,
        config: TrackerConfig | None = None,
        responses: Path | None = None,
    ) -> str:
        calls.update(kind="joint", checkpoint=checkpoint, recipe=recipe, device=device, responses=responses)
        return "tracker"

    monkeypatch.setattr(reference_mount.CellTracker, "from_packs", staticmethod(from_packs))
    monkeypatch.setattr(reference_mount.CellTracker, "from_joint", staticmethod(from_joint))
    return calls


def test_packs(recorded: dict[str, object]) -> None:
    ReferenceMount.packs(_PROC, Path("a"), Path("b"), "cpu")
    assert recorded["pack1"] == _PROC / "a"
    assert recorded["pack2"] == _PROC / "b"
    assert recorded["responses"] == _PROC / RESPONSE_CACHE


def test_pilkwang(recorded: dict[str, object]) -> None:
    ReferenceMount.pilkwang(_PROC, "cuda")
    assert (recorded["pack1"], recorded["pack2"]) == (_PROC / PILKWANG_PRIMARY, _PROC / PILKWANG_SECONDARY)
    assert recorded["device"] == "cuda"


def test_joint(recorded: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    # The one thing a call site could get wrong: a recipe at a grid the transformer never saw.
    class _Model:
        downsample = (1, 2, 2)

    monkeypatch.setattr(reference_mount.JointModel, "from_checkpoint", staticmethod(lambda *_: _Model()))
    ReferenceMount.joint(Path("ckpt.pt"), _PROC, "cpu")
    recipe = recorded["recipe"]
    assert isinstance(recipe, DetectorRecipe)
    assert recorded["kind"] == "joint"
    assert tuple(recipe.downsample) == (1, 2, 2)
    assert recorded["responses"] == _PROC / RESPONSE_CACHE
