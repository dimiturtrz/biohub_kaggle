"""The durable, across-runs half of a training run's record — what a step-windowed loop tells mlflow.

`core.tracking.Tracker` owns the backend (and its `CELLTRACK_NO_MLFLOW` opt-out, which turns every call
below into a no-op); `core.obs.Obs` owns the live, human, this-run log. This owns the third thing: WHAT a
training run records so two runs are comparable months apart — the whole frozen config plus the split it
was fitted on as params, the checkpoint it produced as a tag, and one metric row per eval window keyed by
the optimiser step. Both trainers run the same loop shape, so they say it the same way.

Weights are never uploaded as artifacts (hundreds of MB per checkpoint); the run records their PATH.

The mlflow run id lives in a directory beside the checkpoint (`<weights>.mlflow/`), so `--resume` continues
the SAME mlflow run while two different weights files stay two different runs.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from core.tracking import Tracker

_EXPERIMENT = "celltrack-training"


@dataclass(frozen=True)
class TrainingSplit:
    """How many videos each side of the three-way split holds — provenance for what a run was fitted on."""

    train: int
    validation: int
    test: int


@dataclass(frozen=True)
class RunSetup:
    """One invocation of a trainer: where its checkpoint goes, how it starts, and what it is fitted on.

    The shared vocabulary both trainers' `train` speaks, and exactly the provenance a tracked run records —
    so an arm is one object at the call site instead of four trailing keyword arguments.
    """

    save_to: Path
    warm_start: bool = False
    resume: bool = False
    split: TrainingSplit | None = None


class TrainingRun:
    """One tracked training run: config + split as params, checkpoint provenance as tags, a row per window."""

    def __init__(self, handle: Tracker._Noop | Tracker._Live) -> None:
        self._handle = handle

    @classmethod
    def open(cls, config: Mapping[str, object], setup: RunSetup) -> "TrainingRun":
        """Open (or, under `setup.resume`, continue) the mlflow run that owns this invocation's checkpoint."""
        save_to, run_dir = setup.save_to, setup.save_to.with_suffix(".mlflow")
        if not setup.resume:
            shutil.rmtree(run_dir, ignore_errors=True)  # a fresh training run is a fresh mlflow run, not a resume
        start = "warm" if setup.warm_start else "scratch"
        params: dict[str, object] = {**config, "warm_start": setup.warm_start, "weights": save_to.name}
        if setup.split is not None:
            params["split"] = asdict(setup.split)
        tags: dict[str, object] = {"checkpoint": str(save_to), "start": start}
        return cls(Tracker(_EXPERIMENT, f"{save_to.stem}-{start}", params, tags).track_run(run_dir))

    def window(self, step: int, metrics: Mapping[str, float]) -> None:
        """Record one eval window's numbers at the optimiser step they were measured at."""
        for key, value in metrics.items():
            self._handle.metric(key, value, step=step)

    def close(self, best: float) -> None:
        """End the run, tagging the score of the checkpoint it selected."""
        self._handle.tag("best_score", f"{best:.4f}")
        self._handle.end()
