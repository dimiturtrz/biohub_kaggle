"""Equivalence-class tests for the confusor referee's pure parts.

Scoring a row for real needs the card, the logit cache and the pilkwang packs, so the affinity source and the
mislink extraction are stubbed and what is asserted is the logic the referee itself owns: the RANK BINNING of a
pooled signal (`score`), the head-gain gate (`verdict`), the print row (`log`) and the source dispatch.
Gate classes: all three conditions met, and one failing condition per condition (top1, inv, r1).
Rank classes: r0, r1, r2, r>=3 and the UNSCORED -1 that folds into the r3+ bin.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from celltrack.analysis import confusor_referee
from celltrack.analysis.confusor_referee import AffinitySource, JointAffinity, PackAffinity, RefereeRow
from celltrack.eval.dense_diagnosis import MislinkSignal
from celltrack.eval.proxy import TestMovieProxy
from core.metrics.matching import DistanceMatcher

# A row clearing every gate arm: top1 > 0.45, inv < 0.45, r1 < 30.
_PASSING = RefereeRow(
    linker="ilp", n=60, inv=0.30, p_true=0.60, p_chosen=0.55, top1=0.70, r0=3, r1=20, r2=10, r3plus_uns=8
)


def test_score(monkeypatch: pytest.MonkeyPatch) -> None:
    # One movie, one hand-built pooled signal: the referee's own job is binning its ranks, so the affinity and
    # the mislink extraction are stubbed and the bins are what the assertion reads.
    ranks = np.array([0, 1, 1, 2, 3, 5, -1, -1], dtype=np.int64)
    signal = MislinkSignal(p_true=np.full(len(ranks), 0.4), p_chosen=np.full(len(ranks), 0.6), ranks=ranks)
    tracked = SimpleNamespace(graph=object(), detections=object(), affinity=object())
    source = SimpleNamespace(tracker_for=lambda proc, cfg: SimpleNamespace(run_scored=lambda key, path: tracked))
    proxy = SimpleNamespace(paths=(Path("movie_a.zarr"),), truths=(SimpleNamespace(graph=object()),))
    matcher = SimpleNamespace(match=lambda prediction, truth: object())
    monkeypatch.setattr(confusor_referee.AffinityIndex, "of", staticmethod(lambda scored, affinity: object()))
    monkeypatch.setattr(confusor_referee.MislinkSignal, "of", staticmethod(lambda *_: signal))

    row = RefereeRow.score(
        cast(AffinitySource, source),
        Path("processed"),
        cast(TestMovieProxy, proxy),
        cast(DistanceMatcher, matcher),
        "ilp",
    )

    assert (row.linker, row.n) == ("ilp", len(ranks))
    assert (row.r0, row.r1, row.r2) == (1, 2, 1)
    assert row.r3plus_uns == 4  # ranks 3 and 5, plus the two UNSCORED -1 endpoints
    assert row.inv == 1.0  # every hand-built pair has p_chosen > p_true


def test_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=confusor_referee.__name__):
        _PASSING.log()
    assert "ilp" in caplog.text


def test_verdict() -> None:
    assert "SUBMIT-CANDIDATE" in _PASSING.verdict()


def test_verdict_is_head_bound_when_any_single_arm_fails() -> None:
    # The gate is a conjunction, so each arm alone must be able to fail it.
    for failing in (replace(_PASSING, top1=0.40), replace(_PASSING, inv=0.50), replace(_PASSING, r1=37)):
        assert "HEAD-BOUND-CONFIRMED" in failing.verdict()


def test_build() -> None:
    # The dispatch is a conjunction of flag and path resolution, so both source classes are asserted.
    proc = Path("processed")
    assert isinstance(AffinitySource.build(argparse.Namespace(source="pack", checkpoint="x.pt"), proc), PackAffinity)
    joint = AffinitySource.build(argparse.Namespace(source="joint", checkpoint="armB_warmdet.pt"), proc)
    assert isinstance(joint, JointAffinity)
    assert joint.checkpoint == proc / "armB_warmdet.pt"  # the stem resolves under processed/
