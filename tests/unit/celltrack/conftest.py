"""Stand-ins for the third-party model classes, so the celltrack suite runs without the `external/` checkout.

The `TemporalUNet3D` backbone and the `SimpleNodeTransformer` edge head live in the pinned `external/`
checkout (their architecture, our weights) and are absent in CI. Every test here exercises the code this
repo owns — the inference recipe, peak read-out, feature indexing, pack (de)serialisation — none of which
depends on the real network internals, only on their input/output contract. An autouse fixture swaps both
classes for tiny torch modules honouring that contract, so construction and forward run anywhere.
"""

from __future__ import annotations

from typing import override

import pytest
from torch import Tensor, nn

from celltrack.models.edge_transformer import EdgeTransformerScorer
from celltrack.models.temporal_unet_detector import TemporalUNetDetector


class _StubBackbone(nn.Module):
    """A 1x1x1-conv stand-in for the third-party `TemporalUNet3D`, so construction/forward run without it.

    It honours the same window contract — `(B, T, C_in, Z, Y, X)` in, `(B, T, C_out, Z, Y, X)` out — which
    is all the detector's forward/inference path depends on; the real backbone is exercised in the field, not
    in CI (it lives in the pinned `external/` checkout, absent here).
    """

    def __init__(self, in_channels: int, out_channels: int, layers: list[int]) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    @override
    def forward(self, window: Tensor) -> Tensor:
        batch, frames = window.shape[:2]
        merged = window.reshape(batch * frames, *window.shape[2:])
        out = self.conv(merged)
        return out.reshape(batch, frames, *out.shape[1:])


class _StubTransformer(nn.Module):
    """A dot-product stand-in for the third-party `SimpleNodeTransformer`, honouring its scoring contract.

    Same constructor signature (`feat_dim, hidden_dim, n_heads, n_blocks`) and same forward contract as the
    real head — source features `(s, feat_dim)` and target features `(t, feat_dim)` (plus their voxel
    coordinates) map to a `(s, t)` logit matrix. A single learnable `proj` linear is enough: it carries a
    `proj.weight` so the pack loader's `transformer.*` state round-trips, and its logits soft-max to a valid
    per-source probability column, which is all the mount's `_score_gap`/`affinities` read.
    """

    def __init__(self, feat_dim: int, hidden_dim: int, n_heads: int, n_blocks: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden_dim)

    @override
    def forward(self, feat_src: Tensor, feat_tgt: Tensor, src_voxel: Tensor, tgt_voxel: Tensor) -> Tensor:
        return self.proj(feat_src) @ self.proj(feat_tgt).T  # (s, t) logits


@pytest.fixture(autouse=True)
def _stub_external_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap both third-party model classes for local stubs so no test reaches the absent `external/` dep."""
    monkeypatch.setattr(TemporalUNetDetector, "_backbone_cls", staticmethod(lambda: _StubBackbone))
    monkeypatch.setattr(EdgeTransformerScorer, "_transformer_cls", staticmethod(lambda: _StubTransformer))
