"""Unit tests for a joint run's two files — and for the difference that made three arms unmeasurable."""

from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from celltrack.models.joint_model import JointModel
from celltrack.models.temporal_unet_detector import TemporalUNetDetector
from celltrack.training.joint_checkpoint import JointCheckpoint, RunProgress


class _Optimization:
    """The smallest thing that round-trips like the trainer's optimisation bundle."""

    def __init__(self, value: float = 0.0) -> None:
        self.value: object = value

    def state_dict(self) -> dict[str, object]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.value = state["value"]


def _checkpoint() -> JointCheckpoint:
    return JointCheckpoint(out_channels=1, layers=(2, 4), downsample=(1, 4, 4), device="cpu")


def _model() -> JointModel:
    """A joint model whose heads are trivial — these tests are about persistence, not architecture.

    Both heads stand in as plain linear layers: persistence round-trips whatever `state_dict` gives it, and a
    real backbone would make the test about construction cost instead of about the files.
    """
    return JointModel(cast(TemporalUNetDetector, nn.Linear(2, 2)), nn.Linear(2, 2), (1, 4, 4))


def test_payload():
    """The saved model carries both heads AND the shape needed to rebuild them, under one set of names."""
    payload = _checkpoint().payload(_model())

    assert set(payload) == {"detector_state", "transformer_state", "config"}
    assert payload["config"] == {
        "out_channels": 1,
        "layers": [2, 4],
        "downsample": [1, 4, 4],
        "temporal_position": False,
        "relative_position": None,
        "norm": "batch",
        "head": "pack",
    }


def test_save_best(tmp_path: Path):
    """The best file is exactly the model payload — what `JointModel.from_checkpoint` expects."""
    path = tmp_path / "best.pt"
    _checkpoint().save_best(path, _model())

    assert set(torch.load(path, weights_only=False)) == {"detector_state", "transformer_state", "config"}


def test_save_snapshot(tmp_path: Path):
    """A snapshot is the SAME payload plus what continuing needs, so either file rebuilds a model.

    This is the point of the split: `save-best` keeps the initialisation when no window beats it, so an arm
    that failed to clear its own bar leaves a best-file that is the warm start. The trained weights exist only
    here, and a matched comparison across arms has to be able to read them.
    """
    path = tmp_path / "run.resume.pt"
    aux: dict[str, nn.Module] = {"contrastive": nn.Linear(2, 2), "objective": nn.Linear(2, 2)}
    _checkpoint().save_snapshot(path, _model(), aux, _Optimization(), RunProgress(done=250, best=0.8))

    state = torch.load(path, weights_only=False)
    assert {"detector_state", "transformer_state", "config"} <= set(state)
    assert (state["done"], state["best"]) == (250, 0.8)


def test_restore(tmp_path: Path):
    """A snapshot round-trips the heads, the optimiser and the progress it was written at."""
    path = tmp_path / "run.resume.pt"
    saved, optimization = _model(), _Optimization(value=7.0)
    aux: dict[str, nn.Module] = {"contrastive": nn.Linear(2, 2), "objective": nn.Linear(2, 2)}
    _checkpoint().save_snapshot(path, saved, aux, optimization, RunProgress(done=500, best=0.9))

    restored, into = _model(), _Optimization()
    into_aux: dict[str, nn.Module] = {"contrastive": nn.Linear(2, 2), "objective": nn.Linear(2, 2)}
    progress = _checkpoint().restore(path, restored, into_aux, into)

    assert progress == RunProgress(done=500, best=0.9)
    assert into.value == 7.0
    assert torch.equal(cast(Tensor, restored.detector.weight), cast(Tensor, saved.detector.weight))
