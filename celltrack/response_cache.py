"""Persisting a detector's per-frame probability volumes so read-out experiments skip the forward pass.

Forwarding a trained detector over a whole video is the expensive part of an evaluation; thresholding,
keep-N, linking and post-processing that follow are cheap. But every one of those experiments re-ran the
forward, so a detector's fold-0 score cost a full video sweep each time. The response is a pure function of
(weights, video), though — so it is forwarded once and written to disk, and every later read-out replays it
instantly. A retrain invalidates only its own weights' cache, keyed by the checkpoint's stem.

Volumes are stored half-precision: the response is a probability the peak-picker reads through a
float32 cast anyway, and fp16 halves the disk footprint of a mostly-dark volume that compresses well.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from jaxtyping import Float


class ResponseCache:
    """A directory of per-video probability-volume stacks for one detector checkpoint."""

    def __init__(self, root: Path, weights_key: str) -> None:
        self._directory = root / weights_key
        self._directory.mkdir(parents=True, exist_ok=True)

    def responses(
        self, video_key: str, forward: Callable[[], list[Float[np.ndarray, "z y x"]]]
    ) -> list[Float[np.ndarray, "z y x"]]:
        """The cached per-frame volumes for a video, forwarding and persisting them on a first miss."""
        store = self._directory / f"{video_key}.npy"
        if store.exists():
            return list(np.load(store).astype(np.float32))
        volumes = forward()
        np.save(store, np.stack(volumes).astype(np.float16))
        return volumes
