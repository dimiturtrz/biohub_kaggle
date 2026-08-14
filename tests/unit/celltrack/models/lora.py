import torch
from torch import nn

from celltrack.models.lora import LoRA, LoraConfig

torch.manual_seed(0)  # deterministic randn so a failure reproduces


def _backbone() -> nn.Module:
    """A tiny stand-in with the two adapted leaf kinds under named submodules, for injection tests."""
    module = nn.Module()
    module.add_module("transformer", nn.Linear(8, 8))
    module.add_module("conv", nn.Conv3d(2, 3, kernel_size=3, padding=1))
    module.add_module("keep", nn.Linear(8, 8))  # not in targets — must stay frozen and unadapted
    return module


def test_scaling():
    """alpha/rank decouples strength from capacity, so a rank sweep does not also rescale the delta."""
    assert LoraConfig(rank=8, alpha=16.0).scaling() == 2.0


def test_linear_forward():
    """A freshly-adapted Linear is byte-identical to its base — the zero up-projection makes the delta vanish."""
    base = nn.Linear(8, 8)
    inputs = torch.randn(4, 8)
    adapted = LoRA.Linear(base, LoraConfig(rank=4))
    assert torch.equal(adapted.forward(inputs), base(inputs))
    with torch.no_grad():
        adapted.up.add_(1.0)  # once the up-projection moves, the delta is real
    assert not torch.equal(adapted.forward(inputs), base(inputs))


def test_conv3d_forward():
    """A freshly-adapted Conv3d is byte-identical to its base for the same zero-delta reason."""
    base = nn.Conv3d(2, 3, kernel_size=3, padding=1)
    inputs = torch.randn(1, 2, 4, 4, 4)
    adapted = LoRA.Conv3d(base, LoraConfig(rank=2))
    assert torch.equal(adapted.forward(inputs), base(inputs))


def test_inject():
    """inject adapts only targeted leaves, returns the count, and the whole module is identical at init."""
    module = _backbone()
    inputs = torch.randn(2, 8)
    before = module.get_submodule("transformer")(inputs)
    adapted = LoRA.inject(module, LoraConfig(rank=4, targets=("transformer", "conv")))
    assert adapted == 2  # transformer + conv, not keep
    assert isinstance(module.get_submodule("transformer"), LoRA.Linear)
    assert isinstance(module.get_submodule("keep"), nn.Linear)  # untargeted leaf untouched
    assert torch.equal(module.get_submodule("transformer")(inputs), before)  # byte-identical at init


def test_adapter_parameters():
    """Only the adapter tensors are trainable — the frozen base is kept out of the optimiser's reach."""
    module = _backbone()
    LoRA.inject(module, LoraConfig(rank=4, targets=("transformer",)))
    trainable = list(LoRA.adapter_parameters(module))
    assert trainable  # non-empty
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for parameter in module.get_submodule("keep").parameters())


def test_adapter_state():
    """adapter_state carries only the trainable tensors — the tiny checkpoint a LoRA run saves."""
    module = _backbone()
    LoRA.inject(module, LoraConfig(rank=4, targets=("transformer",)))
    state = LoRA.adapter_state(module)
    assert state  # non-empty
    assert all("transformer" in name for name in state)
    assert not any("keep" in name for name in state)


def test_matched():
    """matched previews which module names a target set would adapt, without mutating anything."""
    module = _backbone()
    predicate = LoRA.matched(module, ("transformer",))
    assert predicate("transformer")
    assert not predicate("keep")
