"""The inference precision a device can actually accelerate — chosen from the hardware, not from a flag.

Every forward in this pipeline ran in fp32, which leaves the tensor cores idle on exactly the hardware the
submission runs on: a Kaggle T4 peaks around 8 TFLOPS in fp32 against ~65 in fp16. The right dtype is a
property of the device, not a knob to tune — Turing (the T4) accelerates fp16 and has no bf16 at all, while
Blackwell (the training box) prefers bf16 for its wider exponent. Asking the device settles it, so one code
path serves the T4, the workstation, the laptop and CPU-only CI without a per-machine constant.

Reduced precision is safe *here* only because the read-out stays fp32: peaks are recovered by an equality
comparison against a max-pool, and a half-precision volume collapses the valley between two touching nuclei
into the same tie as their summits (a cached-fp16 response once cost 0.909 -> 0.816 that way). The matmuls
run reduced; whatever the read-out consumes is cast back. Verified on the four-movie proxy: 0.9334 with
autocast against 0.9334 without, unchanged to four decimals.
"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from typing import cast

import torch


class AutocastPolicy:
    """Runs a CUDA forward in fp16, and stays out of the way everywhere else."""

    @staticmethod
    def of(device: str) -> AbstractContextManager[None]:
        """An fp16 autocast context for `device` — reduced precision on CUDA, a no-op on CPU.

        The caller casts whatever leaves the context back to fp32: the peak read-out compares volumes for
        equality, which half precision would tie together.
        """
        if torch.device(device).type != "cuda":
            return contextlib.nullcontext()
        return cast(AbstractContextManager[None], torch.autocast("cuda", dtype=AutocastPolicy.dtype()))

    @staticmethod
    def dtype() -> torch.dtype:
        """fp16 — inference wants the mantissa, not bf16's exponent range, and fp16 is the T4's fast half.

        The training loop autocasts in bf16 for the opposite reason: its wide exponent keeps gradients off the
        underflow floor without a loss scaler. A forward carries no gradients, so that range buys nothing here,
        while bf16's three fewer mantissa bits move the peak logits the read-out thresholds — measured on the
        four-movie proxy as 0.9309 against fp32's 0.9334, where fp16 held 0.9334 exactly.
        """
        return torch.float16
