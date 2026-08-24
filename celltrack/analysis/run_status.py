"""Classify a training run as CONVERGED / TRUNCATED / UNDERTRAINED from its driver log.

A number off a run nobody verified finished was never true (kneemri, 2026-08-24). Our recipes anneal a
cosine LR sized to the full epoch count: stop early and the anneal — where fine convergence happens —
never runs, so the partial proxy is a FLOOR of a *different* experiment, not "the result minus polish".
Two runs can share a best epoch and mean opposite things (converged vs floored); only whether the run
finished, plus the curve shape, tells them apart.

Three outcomes, read from the eval lines this project's joint trainer already logs
(`epoch E/EMAX | step S/SMAX | ... | proxy P (sel ..) | best B | node R ..`):

- TRUNCATED   — last epoch < epochs asked. On a cosine schedule the anneal never ran → the proxy is a
                FLOOR, not comparable to a finished run. But the floor's SIZE is set by DEPTH, not by the
                verdict (kneemri, 2026-08-24): a one-cycle schedule is cheap late, expensive early — a kill
                at 10/12 can owe nothing measurable while a kill at 7/12 costs several noise floors. So this
                outcome carries `last_epoch/epochs_asked`; price the floor against your own FINISHED curves
                at that depth (best-through-N vs best-overall) before calling a killed run's absence a result.
- UNDERTRAINED — ran to the end but the best proxy lands on (near) the last eval with proxy still rising:
                the schedule was too short / the rate wrong for this trunk. Curve says: refit and rerun.
- CONVERGED   — ran to the end and the best proxy is mid-run with a decline/plateau after: trust the peak,
                the rate was right. Do NOT refit on principle (kneemri's ConvNeXt correction).
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# `epoch 3.5/10 | step 63084/180240 | ... | proxy 0.5223 (sel ..) | best 0.5413 | ...`
_EVAL = re.compile(
    r"epoch\s+(?P<epoch>[\d.]+)/(?P<emax>[\d.]+)\s*\|\s*step\s+(?P<step>\d+)/(?P<smax>\d+)"
    r".*?\bproxy\s+(?P<proxy>[\d.]+).*?\bbest\s+(?P<best>[\d.]+)"
)
_TAIL_WINDOW = 3  # evals over which "still rising" is judged
_MIN_RISE_EVALS = 2  # need at least two tail evals to measure a rise between them
_END_FRACTION = 0.90  # best must sit past this fraction of asked epochs to count as end-loaded
_RISE_EPS = 0.003  # proxy noise floor; a rise below this is not a rise (project score-noise rule)


class Outcome(str, Enum):
    TRUNCATED = "TRUNCATED"
    UNDERTRAINED = "UNDERTRAINED"
    CONVERGED = "CONVERGED"
    NO_EVALS = "NO_EVALS"


@dataclass(frozen=True)
class _Eval:
    epoch: float
    proxy: float
    best: float


@dataclass(frozen=True)
class RunStatus:
    """The verdict on a single training run, plus the numbers that justify it."""

    outcome: Outcome
    epochs_asked: float
    last_epoch: float
    best_proxy: float
    best_epoch: float
    last_proxy: float
    n_evals: int
    reason: str

    @property
    def trustworthy(self) -> bool:
        """A finished, converged run is the only one whose proxy may be compared to another run's."""
        return self.outcome is Outcome.CONVERGED

    @staticmethod
    def _parse(log_text: str) -> tuple[list[_Eval], float]:
        evals: list[_Eval] = []
        emax = 0.0
        for m in _EVAL.finditer(log_text):
            emax = float(m["emax"])
            evals.append(_Eval(epoch=float(m["epoch"]), proxy=float(m["proxy"]), best=float(m["best"])))
        return evals, emax

    @classmethod
    def classify(cls, log_text: str) -> RunStatus:
        evals, epochs_asked = cls._parse(log_text)
        if not evals:
            return cls(
                Outcome.NO_EVALS, epochs_asked, 0.0, float("nan"), float("nan"), float("nan"), 0, "no eval lines in log"
            )

        last = evals[-1]
        best_eval = max(evals, key=lambda e: e.proxy)
        finished = last.epoch >= _END_FRACTION * epochs_asked

        if not finished:
            return cls(
                Outcome.TRUNCATED,
                epochs_asked,
                last.epoch,
                best_eval.proxy,
                best_eval.epoch,
                last.proxy,
                len(evals),
                f"stopped at epoch {last.epoch:g}/{epochs_asked:g} (<{_END_FRACTION:.0%}); cosine anneal "
                f"never ran -> proxy {best_eval.proxy:.4f} is a FLOOR, not comparable to a finished run",
            )

        # ran to the end: is the peak end-loaded AND still climbing over the last window? -> undertrained.
        best_is_end_loaded = best_eval.epoch >= _END_FRACTION * epochs_asked
        tail = evals[-min(_TAIL_WINDOW, len(evals)) :]
        still_rising = len(tail) >= _MIN_RISE_EVALS and (tail[-1].proxy - tail[0].proxy) > _RISE_EPS
        if best_is_end_loaded and still_rising:
            return cls(
                Outcome.UNDERTRAINED,
                epochs_asked,
                last.epoch,
                best_eval.proxy,
                best_eval.epoch,
                last.proxy,
                len(evals),
                f"best proxy {best_eval.proxy:.4f} at epoch {best_eval.epoch:g} (end-loaded) with proxy "
                f"still rising +{tail[-1].proxy - tail[0].proxy:.4f} over last {len(tail)} evals -> "
                f"schedule too short / rate wrong for this trunk; refit and rerun",
            )

        return cls(
            Outcome.CONVERGED,
            epochs_asked,
            last.epoch,
            best_eval.proxy,
            best_eval.epoch,
            last.proxy,
            len(evals),
            f"ran to epoch {last.epoch:g}/{epochs_asked:g}; best proxy {best_eval.proxy:.4f} at epoch "
            f"{best_eval.epoch:g} with decline/plateau after -> trust the peak, rate was right",
        )


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="driver log with `epoch E/EMAX | ... | proxy P | best B` evals")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    status = RunStatus.classify(args.log.read_text(encoding="utf-8", errors="replace"))
    logger.info(
        "%s  (%d evals, epoch %g/%g)", status.outcome.value, status.n_evals, status.last_epoch, status.epochs_asked
    )
    logger.info("  best proxy %.4f @ epoch %g | last %.4f", status.best_proxy, status.best_epoch, status.last_proxy)
    logger.info("  %s", status.reason)
    logger.info("  trustworthy (comparable to another run): %s", status.trustworthy)


if __name__ == "__main__":
    main()
