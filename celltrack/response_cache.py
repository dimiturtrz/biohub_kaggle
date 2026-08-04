"""Persisting a detector's per-frame response volumes so read-out experiments skip the forward pass.

Forwarding a trained detector over a whole video is the expensive part of an evaluation; thresholding,
keep-N, linking and post-processing that follow are cheap. But every one of those experiments re-ran the
forward, so a detector's fold-0 score cost a full video sweep each time. The response is a pure function of
(weights, video, response kind, recipe), though — so it is forwarded once and written to disk, and every
later read-out replays it instantly. A retrain invalidates only its own weights' cache, keyed by stem.

The stored filename carries the response `kind` and a recipe `fingerprint` alongside the video key, so a
cache never masquerades as one it is not: logit volumes and their sigmoid probabilities (or a centre-prior's
heatmaps) share a weights directory but differ in kind, and a recipe change (a different downsample or TTA)
that changes the forward differs in fingerprint. A mismatch is a miss that re-forwards, not a stale replay
of the wrong array — the bug a single-key cache invited when the read-out switched from probabilities to
logits under the same stem.

Volumes are stored at full float32 precision, not half. The peak read-out is an *equality* NMS
(`volume == max_pool(volume)`), and fp16 rounding fabricates ties between distinct neighbouring peaks —
false plateaus the component-collapse then merges into one centre. A cached replay must reproduce the live
score exactly (that faithfulness is the whole point of the cache), so the storage cannot round the response
the NMS reads through. Disk is out-of-repo and cheap; a wrong number replayed fast is not.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from jaxtyping import Float

# A video's per-frame response volumes — the one type the store contract, its cache, and its callers all speak.
_Volumes = list[Float[np.ndarray, "z y x"]]


class ResponseStore(Protocol):
    """A detector's response-volume store: a first forward computes the volumes, later calls decide whether to
    replay or recompute. The read-out (blend, threshold, link) speaks this contract, not a concrete cache."""

    def responses(self, video_key: str, forward: Callable[[], _Volumes], *, kind: str, fingerprint: str) -> _Volumes:
        """The response volumes for a video, computing them via `forward` when this store has nothing to replay."""
        ...


class EphemeralResponseStore:
    """Forwards on every call and persists nothing — the store a single-pass submission mounts.

    A submission forwards each video once and reads it out once; there is no later threshold or linker sweep to
    amortise, so persisting the full-precision volumes to disk buys nothing and only fills the kernel's bounded
    working disk (the frontier's dense movie is gigabytes of fp32 logits per seed). This satisfies the same
    `ResponseStore` contract as `ResponseCache`, but hands the freshly forwarded volumes straight to the read-out
    and drops them — the equality-NMS still reads them at full precision, in memory, never through disk.
    """

    def responses(self, video_key: str, forward: Callable[[], _Volumes], *, kind: str, fingerprint: str) -> _Volumes:
        """Forward the volumes once and return them unpersisted — no replay to amortise in a single pass."""
        return forward()


class ResponseCache:
    """A directory of per-video response-volume stacks for one detector checkpoint, keyed by kind and recipe."""

    def __init__(self, root: Path, weights_key: str) -> None:
        self._directory = root / weights_key
        self._directory.mkdir(parents=True, exist_ok=True)

    def responses(self, video_key: str, forward: Callable[[], _Volumes], *, kind: str, fingerprint: str) -> _Volumes:
        """The cached per-frame volumes for a video, forwarding and persisting them on a first miss.

        `kind` (logit / prob / heatmap) and the recipe `fingerprint` are part of the store name, so a request
        for a different response than the one on disk misses and re-forwards rather than replaying the wrong
        array. Both are required — a caller cannot silently reuse a cache for a response it did not produce.
        """
        store = self._directory / f"{video_key}.{kind}.{fingerprint}.npy"
        if store.exists():
            return list(np.load(store).astype(np.float32))
        volumes = forward()
        np.save(store, np.stack(volumes).astype(np.float32))
        return volumes
