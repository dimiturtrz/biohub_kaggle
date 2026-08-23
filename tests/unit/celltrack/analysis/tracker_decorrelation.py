"""The decorrelation config set must expand each shipped knob into a distinct 0.900-tying operating point."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from celltrack.analysis.tracker_decorrelation import DecorrelationRun
from celltrack.operating_point import TrackerConfig
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.paths import DataRoot

# Two cells (x=0, x=10) across two timepoints; columns are t, z, y, x.
_COORDS = np.array([[0, 0, 0, 0], [0, 0, 0, 10], [1, 0, 0, 0], [1, 0, 0, 10]])
_IDS = np.array([10, 11, 20, 21])
_BOTH = np.array([[10, 20], [11, 21]])  # both cells linked across the frames
_ONE = np.array([[10, 20]])  # only cell 0 linked


def _graph(edges: np.ndarray) -> TrackGraph:
    return TrackGraph(node_ids=_IDS, coordinates=_COORDS, edges=edges)


def test_configs() -> None:
    base = TrackerConfig.shipped()
    configs = DecorrelationRun.configs(base)

    assert configs["shipped_thr097"] is base  # the champion is the untouched base itself
    assert configs["flow_thr080"].threshold == 0.80  # each threshold variant moves only the operating point
    assert configs["flow_thr085"].threshold == 0.85
    assert configs["view_tta_thr090"].edge_options.view_tta is True  # the view-TTA knob is the only edge change
    assert configs["align_seed_thr090"].edge_options.align_seed_moments is True
    assert configs["recall_thr085"].keep_boundary_tracks is True  # the recall variant opens boundary tracks + reuse


def test_geometric_div_config_only_when_the_base_has_a_division_stage() -> None:
    base = TrackerConfig.shipped()
    if base.division is not None:
        division = DecorrelationRun.configs(base)["geometric_div"].division
        assert division is not None
        assert division.candidacy == "geometric"
    else:
        # no division stage -> no geometric-div config to add
        assert "geometric_div" not in DecorrelationRun.configs(base)


def test_agreement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`agreement` reduces every config pair to a pooled, direction-symmetrised edge Jaccard in [0, 1].

    The tracker run is stubbed to two configs over one movie: an identical pair agrees perfectly (1.0), a pair
    differing by one of two links agrees partially — the decorrelation an ensemble ADD criterion reads.
    """
    proxy = SimpleNamespace(spacing=Spacing(1, 1, 1), paths=[tmp_path / "m.zarr"])
    graphs = {
        "shipped_thr097": {"m": _graph(_BOTH)},
        "twin": {"m": _graph(_BOTH)},
        "flow_thr080": {"m": _graph(_ONE)},
    }
    monkeypatch.setattr(DecorrelationRun, "_graphs", lambda self, root: (proxy, graphs))

    pairwise = DecorrelationRun("cpu").agreement(cast(DataRoot, tmp_path / "paths.yaml"))

    assert set(pairwise) == {("shipped_thr097", "twin"), ("shipped_thr097", "flow_thr080"), ("twin", "flow_thr080")}
    assert pairwise[("shipped_thr097", "twin")] == 1.0  # identical link sets agree perfectly
    assert 0.0 <= pairwise[("shipped_thr097", "flow_thr080")] < 1.0  # a dropped link decorrelates the pair
