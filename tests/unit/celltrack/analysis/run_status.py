"""Equivalence-class tests for the run-status classifier.

Classes partition the (finished?, peak-location, still-rising?) space: no evals, truncated (didn't finish),
undertrained (finished + peak end-loaded + rising), converged (finished + peak mid + declined), plus the
boundary where a finished run's peak is end-loaded but flat (converged, not undertrained).
"""

from __future__ import annotations

from celltrack.analysis.run_status import Outcome, RunStatus


def _eval(epoch: float, emax: float, proxy: float, best: float) -> str:
    return (
        f"10:00:00 INFO __main__ | epoch {epoch:g}/{emax:g} | step 100/1000 | "
        f"train loss 0.05 (edge 0 det 0 nce 0 hn 0) | proxy {proxy:.4f} (sel {proxy:.4f}) | "
        f"best {best:.4f} | node R 0.9 ratio +1 | mislinks 0 inv 0.9 P true 0.1 vs chosen 0.3 | "
        f"sanity AUC 1.0 | 10 it/s | elapsed 1h ETA 1h"
    )


def test_classify_no_evals() -> None:
    text = "00:27:44 INFO __main__ | init proxy 0.0000 (sel 0.0000) | node R 0.000 ratio -1.00"
    assert RunStatus.classify(text).outcome is Outcome.NO_EVALS


def test_classify_truncated_is_a_floor() -> None:
    # cosine sized to 8 epochs, killed at 5 -> anneal never ran
    text = "\n".join(_eval(e, 8, 0.40 + 0.02 * e, 0.40 + 0.02 * e) for e in (1, 2, 3, 4, 5))
    status = RunStatus.classify(text)
    assert status.outcome is Outcome.TRUNCATED
    assert not status.trustworthy
    assert status.last_epoch == 5  # depth carried on the verdict for floor-pricing


def test_classify_undertrained_peak_at_end_still_rising() -> None:
    # ran the full 8 epochs but proxy monotone up through the last eval -> schedule too short
    text = "\n".join(_eval(e, 8, 0.40 + 0.03 * e, 0.40 + 0.03 * e) for e in (2, 4, 6, 7, 8))
    status = RunStatus.classify(text)
    assert status.outcome is Outcome.UNDERTRAINED
    assert not status.trustworthy


def test_classify() -> None:
    # peak at epoch 4, declines after -> converged; trust the peak, do not refit
    proxies = {2: 0.50, 4: 0.62, 6: 0.60, 7: 0.59, 8: 0.58}
    text = "\n".join(_eval(e, 8, p, max(proxies[k] for k in proxies if k <= e)) for e, p in proxies.items())
    status = RunStatus.classify(text)
    assert status.outcome is Outcome.CONVERGED
    assert status.trustworthy
    assert status.best_epoch == 4


def test_classify_finished_but_flat_tail_is_converged_not_undertrained() -> None:
    # boundary: peak end-loaded yet the tail is flat (rise below the noise eps) -> converged
    text = "\n".join(_eval(e, 8, p, p) for e, p in [(2, 0.58), (4, 0.60), (6, 0.601), (8, 0.6015)])
    assert RunStatus.classify(text).outcome is Outcome.CONVERGED


def test_trustworthy() -> None:  # only a finished, converged run is comparable to another run

    converged = "\n".join(_eval(e, 8, p, max(0.50, p)) for e, p in [(2, 0.50), (4, 0.62), (6, 0.60), (8, 0.58)])
    truncated = "\n".join(_eval(e, 8, 0.40 + 0.02 * e, 0.40 + 0.02 * e) for e in (1, 2, 3))
    assert RunStatus.classify(converged).trustworthy
    assert not RunStatus.classify(truncated).trustworthy
