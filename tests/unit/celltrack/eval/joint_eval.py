"""Unit test for the saved-joint-checkpoint scorer — the mount a held-out read of our own weights goes through."""

from pathlib import Path
from typing import cast

import pytest

from celltrack.eval.joint_eval import JointCheckpointEval
from celltrack.eval.model_evaluator import ModelEvaluator
from celltrack.eval.proxy import TestMovieProxy
from celltrack.operating_point import TrackerConfig
from core.paths import DataRoot


def test_mounted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The chosen movies and the CHECKPOINT'S downsample reach the mount, behind the published packs' paths."""
    captured: dict[str, object] = {}

    def _fake_load(root: DataRoot, stems: tuple[str, ...]) -> str:
        captured["stems"] = stems
        return "proxy"

    def _fake_mount(proxy: object, packs: tuple[Path, Path], recipe: object, device: str, config: TrackerConfig) -> str:
        captured["proxy"], captured["packs"], captured["recipe"] = proxy, packs, recipe
        return "evaluator"

    monkeypatch.setattr(TestMovieProxy, "load", staticmethod(_fake_load))
    monkeypatch.setattr(ModelEvaluator, "mount", staticmethod(_fake_mount))
    root = cast(DataRoot, type("_Root", (), {"processed": lambda self, dataset: tmp_path})())

    evaluator = JointCheckpointEval.mounted(root, ("6bba_05db0fb1",), (1, 4, 4), "cpu", TrackerConfig())

    assert evaluator == "evaluator"
    assert captured["stems"] == ("6bba_05db0fb1",)
    assert captured["proxy"] == "proxy"
    assert cast(tuple[Path, Path], captured["packs"])[0] == tmp_path / "reference/pilkwang/split_0"
    assert cast(object, captured["recipe"]).downsample == (1, 4, 4)  # type: ignore[missing-attribute]
