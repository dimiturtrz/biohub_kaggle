"""The tracked epoch loop — the reusable driver every training arm shares.

It owns the cross-cutting concerns (per-epoch MLflow metrics, early stopping, save-best, human log)
and stays agnostic to WHAT an epoch does: the caller injects three callables — `train` (one epoch of
updates), `evaluate` (a scored validation pass, dict must carry "score"), and `save` (persist the best
checkpoint). So the same loop drives the joint ourstunet recipe, an all-199 run, or a warm-start, each
a queryable mlflow run. The loop never touches the model, so it is unit-testable with scripted scores.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from celltrack.training.early_stop import EarlyStop

if TYPE_CHECKING:
    from core.hparams import RunConfig
    from core.tracking import Tracker

_SCORE_KEY = "score"


class TrackedRun:
    """Drive up to `config.optim.epochs` epochs, scoring every `config.eval.every` (and the last),
    logging each metric to the tracker at `step=epoch`, saving on every improvement, and stopping once
    early-stop patience is exhausted. `fit` returns the best score seen."""

    def __init__(self, config: RunConfig, tracker: Tracker._Noop | Tracker._Live, log: logging.Logger) -> None:
        self._config, self._tracker, self._log = config, tracker, log

    def fit(
        self,
        train: Callable[[], Mapping[str, float]],
        evaluate: Callable[[], Mapping[str, float]],
        save: Callable[[], None],
    ) -> float:
        optim, evaluation = self._config.optim, self._config.eval
        stop = EarlyStop(optim.patience, optim.es_min_delta)
        last = optim.epochs - 1
        for epoch in range(optim.epochs):
            train_metrics = train()
            if epoch % evaluation.every != 0 and epoch != last:
                self._record(epoch, train_metrics)
                continue
            eval_metrics = evaluate()
            improved = stop.update(eval_metrics[_SCORE_KEY])
            self._record(epoch, {**train_metrics, **eval_metrics, "best": stop.best})
            if improved:
                save()
            self._log_epoch(epoch, train_metrics, eval_metrics, stop.best, saved=improved)
            if stop.should_stop:
                self._log.info("early stop at epoch %d (patience %d, best=%.4f)", epoch, optim.patience, stop.best)
                break
        return stop.best

    def _record(self, epoch: int, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            self._tracker.metric(name, value, step=epoch)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        eval_metrics: Mapping[str, float],
        best: float,
        *,
        saved: bool,
    ) -> None:
        parts = " ".join(f"{k}={v:.4f}" for k, v in {**train_metrics, **eval_metrics}.items())
        marker = " *" if saved else ""
        epochs = self._config.optim.epochs
        self._log.info("epoch %3d/%d | %s | best=%.4f%s", epoch, epochs, parts, best, marker)
