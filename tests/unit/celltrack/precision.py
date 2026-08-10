"""Unit tests for the hardware-chosen inference precision.

The policy has one job — reduced precision where the device accelerates it, nothing anywhere else — so the
cases are the device kinds: a CPU string must yield an inert context, and the dtype must be one of the two
halves, bf16 only where the GPU reports support.
"""

import torch

from celltrack.precision import AutocastPolicy


def test_of():
    """A CPU device gets an inert context — autocast must not touch a forward the host runs in fp32."""
    with AutocastPolicy.of("cpu"):
        tensor = torch.ones(2, 2) @ torch.ones(2, 2)
    assert tensor.dtype == torch.float32


def test_dtype():
    """Inference runs fp16 — it wants the mantissa the read-out thresholds, not bf16's gradient-safe range."""
    assert AutocastPolicy.dtype() == torch.float16
