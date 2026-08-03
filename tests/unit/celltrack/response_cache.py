from pathlib import Path

import numpy as np

from celltrack.response_cache import ResponseCache


def test_responses(tmp_path: Path):
    """A first request forwards and persists; a second replays the stored volumes *exactly*, without forwarding.

    Exactly, not approximately: `0.99998` is not representable in float16, so a half-precision store would
    round it and the equality-based peak NMS would read a different response on replay than on the live run.
    """
    volumes = [np.full((2, 3, 3), 0.99998, dtype=np.float32), np.full((2, 3, 3), 0.25, dtype=np.float32)]
    calls = 0

    def forward():
        nonlocal calls
        calls += 1
        return volumes

    cache = ResponseCache(tmp_path, "weights_a")
    cache.responses("video_x", forward)
    replayed = cache.responses("video_x", forward)

    assert calls == 1
    assert np.array_equal(np.stack(volumes), np.stack(replayed))


def test_a_different_checkpoint_gets_its_own_cache(tmp_path: Path):
    """Keying by checkpoint stem keeps one detector's volumes from masquerading as another's after a retrain."""
    marker = [np.ones((1, 2, 2), dtype=np.float32)]
    ResponseCache(tmp_path, "weights_a").responses("video_x", lambda: marker)

    forwarded = False

    def forward():
        nonlocal forwarded
        forwarded = True
        return [np.zeros((1, 2, 2), dtype=np.float32)]

    ResponseCache(tmp_path, "weights_b").responses("video_x", forward)
    assert forwarded
