"""Low-rank adapters that reshape a FROZEN pretrained backbone — the freeze/unfreeze resolution.

Fully unfreezing pilkwang drifts detection (a warm-start detection loss supervises ~98% of real cells as
background, so the shared features slide off detection as they move for association). Fully freezing kills the
association mechanism (the crowded true-vs-decoy signal lives in the FEATURES, and a frozen backbone cannot
reshape them). LoRA takes the middle: the base weights stay frozen, and a small rank-`r` delta `B @ A` is added
in parallel. Detection is preserved by construction — the base forward is untouched and `B` initialises to
zero, so a freshly-adapted model is BYTE-IDENTICAL to its parent — while association gets exactly `r` degrees of
freedom per layer to bend the features it needs. Whatever detection cost the adaptation does incur is bounded by
the rank, not left to a full-parameter gradient.

Both layer kinds the backbone is built from are covered: `LoRA.Linear` (the edge transformer) and `LoRA.Conv3d`
(the U-Net), nested so the whole subsystem is one subject. `LoRA.inject` walks a module, swaps every matched
leaf for its adapted twin in place, and freezes everything else — a caller adapts "the decoder + the
transformer" by predicate rather than by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import cast, override

import torch
from jaxtyping import Float
from torch import Tensor, nn


@dataclass(frozen=True)
class LoraConfig:
    """The adapter's two numbers and where it lands — a config object rather than a long inject() arg list.

    `rank` is the delta's capacity (columns of `A`); `alpha` scales the delta by `alpha / rank` so the effective
    strength is decoupled from the rank (the standard LoRA parameterisation — raising rank adds capacity without
    also amplifying the update). `targets` names the leaf modules to adapt by a substring of their dotted path,
    so `("transformer", "decoder_blocks.3")` adapts the whole transformer and one late decoder stage.
    """

    rank: int = 16
    alpha: float = 16.0
    targets: tuple[str, ...] = ()

    def scaling(self) -> float:
        """The delta's multiplier — `alpha / rank`, so strength holds as rank (capacity) is swept."""
        return self.alpha / self.rank


class LoRA:
    """The adapter subsystem — its two layer kinds and the injection over a module, one subject, no instance state."""

    class Linear(nn.Module):
        """A frozen `nn.Linear` plus a rank-`r` parallel delta — `base(x) + scaling·(x A^T)B^T`, `B` zero at init."""

        def __init__(self, base: nn.Linear, config: LoraConfig) -> None:
            super().__init__()
            self.base = base
            self.base.requires_grad_(requires_grad=False)
            self.scaling = config.scaling()
            self.in_features = base.in_features  # a drop-in Linear reports its own dims (callers introspect proj)
            self.out_features = base.out_features
            self.down = nn.Parameter(torch.empty(config.rank, base.in_features))
            self.up = nn.Parameter(torch.zeros(base.out_features, config.rank))
            nn.init.kaiming_uniform_(self.down, a=5**0.5)  # delta is `up @ down`; `up`=0 makes it zero at init

        @override
        def forward(self, inputs: Float[Tensor, "... in"]) -> Float[Tensor, "... out"]:
            """The base projection plus the low-rank delta — identical to the base while `up` is still zero."""
            delta = torch.nn.functional.linear(torch.nn.functional.linear(inputs, self.down), self.up)
            return self.base(inputs) + self.scaling * delta

    class Conv3d(nn.Module):
        """A frozen `nn.Conv3d` plus a rank-`r` delta — a 1³ down-projection then a base-shaped conv, zero at init."""

        def __init__(self, base: nn.Conv3d, config: LoraConfig) -> None:
            super().__init__()
            self.base = base
            self.base.requires_grad_(requires_grad=False)
            self.scaling = config.scaling()
            self.in_channels = base.in_channels  # a drop-in Conv3d reports its own dims for any introspection
            self.out_channels = base.out_channels
            kernel = cast(tuple[int, int, int], base.kernel_size)
            padding = cast(tuple[int, int, int], base.padding)
            self.down = nn.Conv3d(base.in_channels, config.rank, kernel_size=1, bias=False)
            self.up = nn.Conv3d(config.rank, base.out_channels, kernel_size=kernel, padding=padding, bias=False)
            nn.init.zeros_(self.up.weight)  # zero up-projection => the delta is zero until training moves it

        @override
        def forward(self, inputs: Float[Tensor, "b c z y x"]) -> Float[Tensor, "b d z y x"]:
            """The base convolution plus the low-rank delta — identical to the base while `up` is still zero."""
            return self.base(inputs) + self.scaling * self.up(self.down(inputs))

    @staticmethod
    def inject(module: nn.Module, config: LoraConfig, keep: Iterable[nn.Parameter] = ()) -> int:
        """Freeze every parameter, then swap each targeted `Linear`/`Conv3d` leaf for its adapted twin in place.

        Returns the number of layers adapted, so a caller can assert the predicate actually matched something —
        a `targets` typo that matches nothing would otherwise train zero adapters and read as a silent null.

        `keep` re-enables grad on params installed BEFORE inject (a temporal table, a distance bias, widened
        velocity columns): the blanket freeze here would otherwise catch them and `adapter_parameters` exclude
        them, leaving each pinned at its zero init — ON in the config yet a silent no-op.
        """
        module.requires_grad_(requires_grad=False)
        skip = LoRA._attention_internals(module)
        adapted = 0
        for name, child in list(module.named_modules()):
            if id(child) in skip or not any(target in name for target in config.targets):
                continue
            replacement = LoRA._adapt(child, config)
            if replacement is not None:
                LoRA._set_submodule(module, name, replacement)
                adapted += 1
        for parameter in keep:
            parameter.requires_grad = True
        return adapted

    @staticmethod
    def _attention_internals(module: nn.Module) -> set[int]:
        """The ids of every module living inside an `nn.MultiheadAttention` — off-limits to adaptation.

        MHA reads its projections' `.weight`/`.bias` FUNCTIONALLY (`F.multi_head_attention_forward`), never
        through their `forward`, so a wrapped projection would both crash on the missing `.weight` AND leave its
        delta unapplied. The attention QKV is simply not adaptable by module-swap; the MLPs, `proj` and the
        feature convs — whose `forward` we do invoke — carry the reshape instead.
        """
        inside: set[int] = set()
        for child in module.modules():
            if isinstance(child, nn.MultiheadAttention):
                inside.update(id(sub) for sub in child.modules())
        return inside

    @staticmethod
    def _adapt(child: nn.Module, config: LoraConfig) -> nn.Module | None:
        """The adapted twin of a leaf, or None when the leaf is neither a `Linear` nor a `Conv3d`."""
        if isinstance(child, nn.Linear):
            return LoRA.Linear(child, config)
        if isinstance(child, nn.Conv3d):
            return LoRA.Conv3d(child, config)
        return None

    @staticmethod
    def _set_submodule(root: nn.Module, dotted: str, replacement: nn.Module) -> None:
        """Rebind `root.<dotted>` to `replacement`, walking the parent chain the dotted path names."""
        parent = root
        *ancestors, leaf = dotted.split(".")
        for step in ancestors:
            parent = parent.get_submodule(step)
        parent.add_module(leaf, replacement)

    @staticmethod
    def adapter_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
        """The trainable adapter parameters alone — what an optimiser is handed so the frozen base is untouched."""
        return (parameter for parameter in module.parameters() if parameter.requires_grad)

    @staticmethod
    def adapter_state(module: nn.Module) -> dict[str, Tensor]:
        """Just the adapter tensors — the checkpoint a LoRA run saves, tiny beside the frozen base it rides on."""
        trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
        return {name: tensor for name, tensor in module.state_dict().items() if name in trainable}

    @staticmethod
    def matched(module: nn.Module, targets: tuple[str, ...]) -> Callable[[str], bool]:
        """A predicate over module names for the given targets — exposed so a caller can preview what would match."""
        names = {name for name, _ in module.named_modules()}
        return lambda name: name in names and any(target in name for target in targets)
