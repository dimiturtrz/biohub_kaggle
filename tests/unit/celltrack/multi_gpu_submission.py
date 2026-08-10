from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from celltrack import multi_gpu_submission as dispatch
from celltrack.multi_gpu_submission import MultiGpuSubmission
from celltrack.tracker import TrackerConfig
from core.data.submission import Submission
from core.data.tracks import TrackGraph

_GRAPH = TrackGraph(
    node_ids=np.array([0, 1], dtype=np.int64),
    coordinates=np.array([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.int64),
    edges=np.array([[0, 1]], dtype=np.int64),
)


class _StubTracker:
    """A mounted tracker whose `run` is a fixed graph, so the dispatcher runs without weights or a GPU."""

    def __init__(self, device: str) -> None:
        self.device = device

    def run(self, video_key: str, path: Path) -> TrackGraph:
        return _GRAPH


def _mount(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Swap the ephemeral mount for a stub, recording the device each shard mounted on."""
    devices: list[str] = []

    def ephemeral(pack1: Path, pack2: Path, device: str, config: TrackerConfig | None = None) -> _StubTracker:
        devices.append(device)
        return _StubTracker(device)

    monkeypatch.setattr(dispatch.CellTracker, "ephemeral", staticmethod(ephemeral))
    return devices


def _submission(devices: tuple[str, ...]) -> MultiGpuSubmission:
    return MultiGpuSubmission(packs=(Path("p1"), Path("p2")), config=TrackerConfig(), devices=devices)


def test_available_devices(monkeypatch: pytest.MonkeyPatch):
    """Several visible GPUs become explicit ordinals; one is the unindexed device, none is CPU."""
    for count, expected in ((2, ("cuda:0", "cuda:1")), (1, ("cuda",)), (0, ("cpu",))):
        monkeypatch.setattr(dispatch.torch.cuda, "device_count", lambda count=count: count)
        assert MultiGpuSubmission.available_devices() == expected


def test_shards():
    """Videos are dealt round-robin over the devices, keyed by name, with empty shards dropped."""
    videos = [Path(f"/m/{name}.zarr") for name in ("a", "b", "c")]

    shards = _submission(("cuda:0", "cuda:1", "cuda:2", "cuda:3")).shards(videos)

    assert [shard.device for shard in shards] == ["cuda:0", "cuda:1", "cuda:2"]  # the fourth got nothing
    assert [[key for key, _ in shard.videos] for shard in shards] == [["a"], ["b"], ["c"]]


def test_shard_graphs(monkeypatch: pytest.MonkeyPatch):
    """A shard mounts the tracker on its own device and returns one graph per video it owns."""
    mounted = _mount(monkeypatch)
    submission = _submission(("cpu",))
    shard = submission.shards([Path("/m/a.zarr"), Path("/m/b.zarr")])[0]

    graphs = submission.shard_graphs(shard)

    assert mounted == ["cpu"]
    assert sorted(graphs) == ["a", "b"]


class _FailedProcess:
    """A worker that dies without delivering its shard — the crash the parent must turn into a loud raise."""

    def __init__(self, target: object, args: tuple[object, ...]) -> None:
        self.name = "failed-worker"
        self.exitcode: int | None = None

    def start(self) -> None:
        self.exitcode = 1

    def join(self) -> None:
        return None


def test_run_raises_on_worker_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A dead worker fails the submission instead of writing one that silently omits its shard's videos."""
    monkeypatch.setattr(dispatch, "_POLL_SECONDS", 0.01)
    context = SimpleNamespace(Queue=Queue, Process=_FailedProcess)
    monkeypatch.setattr(dispatch.torch_mp, "get_context", lambda start_method: context)
    out = tmp_path / "submission.csv"

    with pytest.raises(RuntimeError, match="worker failed"):
        _submission(("cuda:0", "cuda:1")).run([Path("/m/a.zarr"), Path("/m/b.zarr")], out)

    assert not out.exists()


def test_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A single device runs in-process (no spawn) and writes every video into one merged submission CSV."""
    _mount(monkeypatch)
    videos = [Path("/m/b.zarr"), Path("/m/a.zarr")]
    out = tmp_path / "submission.csv"

    _submission(("cpu",)).run(videos, out)

    assert sorted(Submission.read_csv(out).graphs) == ["a", "b"]
