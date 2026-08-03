"""Optional MLflow experiment tracking — local file backend, `runs/` stays canonical.

mlflow is a required dep; the only opt-out is `CELLTRACK_NO_MLFLOW`, which makes every call a no-op so
training/scoring never depend on it. It is the cross-run comparison UI (`mlflow ui`), not the source of
truth — `runs/<name>/{config.json, train.log}` remain authoritative. Every live call is guarded:
tracking must never break a run.

    trk = Tracker("celltrack", run_name, params=cfg.model_dump()).track_run(run_dir)
    trk.metric("val_score", vs, step=epoch)
    trk.artifact("runs/x/config.json"); trk.end()
"""

from __future__ import annotations

import contextlib
import os
import types
from pathlib import Path

import mlflow

_ROOT = Path(__file__).resolve().parents[1]
_MLRUNS = _ROOT / "mlruns"  # artifact store (default root)
_DB_URI = f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}"  # metadata + registry (file store deprecated)
_DISABLE_ENV = "CELLTRACK_NO_MLFLOW"
_RUN_ID_FILE = ".mlflow_run_id"


class Tracker:
    """A tracking RUN is a session: experiment + run_name + params/tags logged once at open. Construct
    with that session config, then open the run — fresh via `start`, or resume-or-create tied to a
    `run_dir` via `track_run`. Both return a handle (`_Live`, or the `_Noop` used when tracking's off
    or setup failed — nested: the handles exist only to serve this subject)."""

    class _Noop:
        """Returned when tracking is off or setup failed — every method is a no-op."""

        def metric(self, key: str, value: int | float, step: int | None = None) -> None: ...
        def tag(self, key: str, value: str | int | float) -> None: ...
        def artifact(self, path: str | Path) -> None: ...
        def end(self) -> None: ...

    class _Live:
        """A live mlflow run handle. Each call is suppressed on failure — a tracking error never
        propagates into the training loop."""

        def __init__(self, backend: types.ModuleType) -> None:
            self._mlflow = backend

        def metric(self, key: str, value: int | float, step: int | None = None) -> None:
            with contextlib.suppress(Exception):
                self._mlflow.log_metric(key, float(value), step=step)

        def tag(self, key: str, value: str | int | float) -> None:
            with contextlib.suppress(Exception):
                self._mlflow.set_tag(key, str(value))

        def artifact(self, path: str | Path) -> None:
            with contextlib.suppress(Exception):
                if Path(path).exists():
                    self._mlflow.log_artifact(str(path))

        def end(self) -> None:
            with contextlib.suppress(Exception):
                self._mlflow.end_run()

    def __init__(
        self,
        experiment: str,
        run_name: str,
        params: dict[str, object] | None = None,
        tags: dict[str, object] | None = None,
    ) -> None:
        self.experiment, self.run_name = experiment, run_name
        self.params, self.tags = params, tags

    @staticmethod
    def _backend() -> types.ModuleType | None:
        """The mlflow module if tracking is enabled AND installed, else None (CELLTRACK_NO_MLFLOW opt-out
        or a venv without mlflow)."""
        if os.environ.get(_DISABLE_ENV):
            return None
        return mlflow

    @staticmethod
    def _flat(nested: dict[str, object], prefix: str = "") -> dict[str, object]:
        """Flatten a nested config dict to dotted keys (mlflow params are scalar)."""
        out: dict[str, object] = {}
        for key, value in nested.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict):
                out.update(Tracker._flat(value, dotted + "."))
            else:
                out[dotted] = value
        return out

    def _configure(self, backend: types.ModuleType) -> None:
        """Point mlflow at the local store + this session's experiment. A fatal setup error propagates
        to the caller's outer guard (-> no-op handle)."""
        _MLRUNS.mkdir(exist_ok=True)
        backend.set_tracking_uri(_DB_URI)
        backend.set_experiment(self.experiment)
        with contextlib.suppress(Exception):
            backend.enable_system_metrics_logging()  # GPU/CPU/mem if psutil+pynvml present

    def _log_open(self, backend: types.ModuleType) -> None:
        if self.params:
            backend.log_params(Tracker._flat(self.params))
        for key, value in (self.tags or {}).items():
            backend.set_tag(key, str(value))

    def start(self) -> Tracker._Noop | Tracker._Live:
        """Begin a fresh tracked run (local mlruns/). Returns a handle; no-op if tracking is off."""
        backend = Tracker._backend()
        if backend is None:
            return Tracker._Noop()
        try:
            self._configure(backend)
            backend.start_run(run_name=self.run_name)
            self._log_open(backend)
            return Tracker._Live(backend)
        except Exception:  # noqa: BLE001 — tracking must never break a run
            return Tracker._Noop()

    def track_run(self, run_dir: str | Path | None = None) -> Tracker._Noop | Tracker._Live:
        """Resume the run tied to `run_dir` (via `<run_dir>/.mlflow_run_id`) if it exists, else start a
        fresh one and persist its id there — so a resumed training run logs into the SAME mlflow run."""
        backend = Tracker._backend()
        if backend is None:
            return Tracker._Noop()
        try:
            self._configure(backend)
            id_file = Path(run_dir) / _RUN_ID_FILE if run_dir else None
            run_id = id_file.read_text(encoding="utf-8").strip() if id_file and id_file.exists() else None
            if run_id:
                backend.start_run(run_id=run_id)  # resume — don't re-log params
            else:
                backend.start_run(run_name=self.run_name)
                self._log_open(backend)
                active = backend.active_run()
                if id_file is not None and active is not None:
                    id_file.parent.mkdir(parents=True, exist_ok=True)
                    id_file.write_text(active.info.run_id, encoding="utf-8")
            return Tracker._Live(backend)
        except Exception:  # noqa: BLE001 — tracking must never break a run
            return Tracker._Noop()
