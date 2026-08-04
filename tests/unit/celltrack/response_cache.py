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
    cache.responses("video_x", forward, kind="logit", fingerprint="f")
    replayed = cache.responses("video_x", forward, kind="logit", fingerprint="f")

    assert calls == 1
    assert np.array_equal(np.stack(volumes), np.stack(replayed))


def test_a_different_checkpoint_gets_its_own_cache(tmp_path: Path):
    """Keying by checkpoint stem keeps one detector's volumes from masquerading as another's after a retrain."""
    marker = [np.ones((1, 2, 2), dtype=np.float32)]
    ResponseCache(tmp_path, "weights_a").responses("video_x", lambda: marker, kind="logit", fingerprint="f")

    forwarded = False

    def forward():
        nonlocal forwarded
        forwarded = True
        return [np.zeros((1, 2, 2), dtype=np.float32)]

    ResponseCache(tmp_path, "weights_b").responses("video_x", forward, kind="logit", fingerprint="f")
    assert forwarded


def test_a_different_kind_does_not_replay_the_wrong_response(tmp_path: Path):
    """Logits and their sigmoid share a weights dir but differ in kind — the second must re-forward, not replay."""
    cache = ResponseCache(tmp_path, "weights_a")
    cache.responses("video_x", lambda: [np.ones((1, 2, 2), dtype=np.float32)], kind="logit", fingerprint="f")

    forwarded = False

    def prob_forward():
        nonlocal forwarded
        forwarded = True
        return [np.full((1, 2, 2), 0.5, dtype=np.float32)]

    replayed = cache.responses("video_x", prob_forward, kind="prob", fingerprint="f")
    assert forwarded  # the prob request did not read the logit store
    assert replayed[0][0, 0, 0] == 0.5


def test_a_different_recipe_fingerprint_re_forwards(tmp_path: Path):
    """A recipe change that alters the forward (a different downsample/TTA) gets a fresh cache, not a stale one."""
    cache = ResponseCache(tmp_path, "weights_a")
    cache.responses("video_x", lambda: [np.ones((1, 2, 2), dtype=np.float32)], kind="logit", fingerprint="ds1x4x4_tta1")

    forwarded = False

    def forward():
        nonlocal forwarded
        forwarded = True
        return [np.zeros((1, 2, 2), dtype=np.float32)]

    cache.responses("video_x", forward, kind="logit", fingerprint="ds1x2x2_tta0")
    assert forwarded
