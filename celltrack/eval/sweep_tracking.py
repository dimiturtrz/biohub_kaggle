"""Recording one mlflow run per swept grid cell — so a sweep is comparable across CLI invocations.

Split out of `proxy_eval` because a file gets one subject: that module's subject is the sweep itself, and
this is how the sweep is remembered. Kept beside it because it exists only to serve that CLI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

from pydantic import BaseModel

from celltrack.operating_point import TrackerConfig
from core.metrics.score import VideoMetrics
from core.tracking import Tracker


class SweepTracking:
    """One mlflow run per swept grid cell, so a sweep is comparable across CLI invocations, not just within one.

    The whole RESOLVED `TrackerConfig` goes in as params (every knob, `--set` overrides folded in), which is
    what makes "what did smooth_strength=0.3 score" answerable months later; the split score and, when asked,
    each movie's raw Jaccard and node ratio go in as metrics. `CELLTRACK_NO_MLFLOW` makes all of it a no-op.
    """

    _EXPERIMENT = "celltrack-proxy"

    def __init__(self, proxy: str) -> None:
        self._proxy = proxy

    def cell(self, config: TrackerConfig, score: float, breakdown: Mapping[str, VideoMetrics]) -> None:
        """Record one grid cell: its config as params, its score (and any per-movie metrics) as metrics."""
        name = f"{self._proxy}-thr{config.threshold:g}-dis{config.linker.disappearance_cost:g}"
        handle = Tracker(self._EXPERIMENT, name, SweepTracking._params(config), {"proxy": self._proxy}).start()
        handle.metric("proxy_score", score)
        for stem, metric in breakdown.items():
            handle.metric(f"{stem}.raw_jaccard", metric.edges.jaccard())
            handle.metric(f"{stem}.node_ratio", metric.total_node_ratio())
        handle.end()

    @staticmethod
    def _params(config: TrackerConfig) -> dict[str, object]:
        """The config as a nested plain dict — pydantic sub-configs dumped so the tracker can flatten them."""
        return {
            key: value.model_dump() if isinstance(value, BaseModel) else value for key, value in asdict(config).items()
        }
