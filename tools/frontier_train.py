"""Run the frontier ``train_unet_transformer.py`` with a strict FULL-model warm start.

Its own ``--unet-weights`` loads non-strictly into the bare UNet, so a full checkpoint (``unet.``-prefixed keys
plus detect head and transformer) silently loads nothing. ``--init-full <ckpt>`` loads every key strictly.

    python tools/frontier_train.py --frontier-repo <repo> --init-full <ckpt> <trainer args>

Its detection→GT matching walks detections in a Python loop with one GPU sync each (~8k syncs per step on dense
frames, GPU ~5% busy); the loop is replaced by :func:`greedy_unique_match`, the same greedy assignment as a
scatter-min.

It also runs in untuned FP32 with an unguarded optimizer step, so one non-finite batch poisons the weights and
every later epoch reads ``det=nan`` with an exactly-zero edge loss (the empty-mask guard in its ``compute_loss``).
:func:`patch_step_guard` skips such a step and names what went non-finite first; ``--bf16`` autocasts the UNet
encode and ``--tf32`` (on by default) enables the tensor-core matmul paths.

Every patch is applied to the imported module in memory: the checkout under ``external/`` stays pristine, and the
patch travels with this file rather than with a disk edit that only one copy of the pack carries.
"""

import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType

import torch

log = logging.getLogger(__name__)

GRAD_CLIP_NORM = 1.0
NONFINITE_REPORT_LIMIT = 5

_SLOW_MATCH = """            order = min_d.argsort()
            gt_taken = torch.zeros(n_gt, dtype=torch.bool, device=device)
            for idx in order:
                if min_d[idx] > max_match_distance:
                    break
                gi = min_i[idx]
                if not gt_taken[gi]:
                    matched[idx] = gi
                    gt_taken[gi] = True
"""
_FAST_MATCH = "            matched = greedy_unique_match(min_d, min_i, n_gt, max_match_distance)\n"

_ENCODE = "        unet_out, det_logits = model.encode(imgs)\n"
_ENCODE_BF16 = """        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=imgs.is_cuda):
            unet_out, det_logits = model.encode(imgs)
        unet_out = unet_out.float()
        det_logits = [logit.float() for logit in det_logits]
"""

_UNGUARDED_STEP = """        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
"""
_GUARDED_STEP = """        if not torch.isfinite(loss):
            report_nonfinite(
                batch, imgs=imgs, unet_out=unet_out, det_logits=det_logits,
                det_losses=det_losses, edge_loss=edge_loss,
            )
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        if torch.isfinite(grad_norm):
            optimizer.step()
        else:
            NONFINITE["grad"] += 1
"""

_TIMING = '            f"total: {t_total:.1f}s"\n'
_TIMING_SKIPPED = (
    '            f"total: {t_total:.1f}s | "\n'
    "            f\"skipped: loss={NONFINITE['loss']} grad={NONFINITE['grad']}\"\n"
)


def greedy_unique_match(min_d: torch.Tensor, min_i: torch.Tensor, n_gt: int, max_distance: float) -> torch.Tensor:
    """Nearest-first greedy: each GT goes to the closest in-range detection whose nearest GT it is, else -1."""
    index = torch.arange(min_d.shape[0], device=min_d.device)
    within = min_d <= max_distance
    closest = torch.full((n_gt,), torch.inf, device=min_d.device, dtype=min_d.dtype).scatter_reduce(
        0, min_i, torch.where(within, min_d, torch.inf), "amin"
    )
    candidate = within & (min_d == closest[min_i])
    first = torch.full((n_gt,), min_d.shape[0], device=min_d.device, dtype=torch.long).scatter_reduce(
        0, min_i, torch.where(candidate, index, min_d.shape[0]), "amin"
    )
    return torch.where(candidate & (first[min_i] == index), min_i, -1)


def patch_fast_match(trainer: ModuleType) -> None:
    source = inspect.getsource(trainer.detect_and_match)
    if _SLOW_MATCH not in source:
        raise RuntimeError("frontier detect_and_match changed; greedy match patch no longer applies")
    trainer.greedy_unique_match = greedy_unique_match
    exec(compile(source.replace(_SLOW_MATCH, _FAST_MATCH), trainer.__file__, "exec"), trainer.__dict__)


NONFINITE = {"loss": 0, "grad": 0, "reported": 0}


def _finite_note(value: torch.Tensor | list[torch.Tensor]) -> object:
    """Whether the stage is finite, and how large it got -- per element when the stage is a list of frames."""
    if isinstance(value, list):
        return [_finite_note(item) for item in value]
    return (bool(torch.isfinite(value).all()), float(value.detach().abs().max()))


def report_nonfinite(batch: dict[str, torch.Tensor], **stages: torch.Tensor | list[torch.Tensor]) -> None:
    """Name which stage was already non-finite when the loss was: the inputs, the features, or a head."""
    NONFINITE["loss"] += 1
    NONFINITE["reported"] += 1
    if NONFINITE["reported"] > NONFINITE_REPORT_LIMIT:
        return
    log.warning(
        "non-finite loss on image_shape=%s n_gt_per_frame=%s; (finite, absmax) per stage: %s",
        tuple(batch["image_shape"][0].tolist()),
        batch["masks"].sum(dim=2).tolist(),
        {name: _finite_note(value) for name, value in stages.items()},
    )


def patch_step_guard(trainer: ModuleType, *, bf16: bool) -> None:
    """Skip a non-finite step instead of poisoning the weights, and optionally autocast the encode to bf16."""
    source = inspect.getsource(trainer.train_epoch)
    replacements = [(_UNGUARDED_STEP, _GUARDED_STEP), (_TIMING, _TIMING_SKIPPED)]
    if bf16:
        replacements.append((_ENCODE, _ENCODE_BF16))
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"frontier train_epoch changed; this patch no longer applies:\n{old}")
        source = source.replace(old, new)
    trainer.NONFINITE = NONFINITE
    trainer.GRAD_CLIP_NORM = GRAD_CLIP_NORM
    trainer.report_nonfinite = report_nonfinite
    exec(compile(source, trainer.__file__, "exec"), trainer.__dict__)


def enable_tf32() -> None:
    """Tensor-core matmul plus autotuned convolutions; the encode is ~73% of a step and all of it is conv/matmul."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _strict_init_class(base: type, checkpoint: Path) -> type:
    class StrictInit(base):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            log.warning("full-model init loaded strictly from %s", checkpoint)

    return StrictInit


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-repo", type=Path, required=True)
    parser.add_argument("--init-full", type=Path, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    args, trainer_args = parser.parse_known_args()

    if args.tf32 and torch.cuda.is_available():
        enable_tf32()
    sys.path[:0] = [str(args.frontier_repo / "src"), str(args.frontier_repo / "scripts")]
    trainer = importlib.import_module("train_unet_transformer")
    patch_fast_match(trainer)
    patch_step_guard(trainer, bf16=args.bf16 and torch.cuda.is_available())
    if args.init_full is not None:
        trainer.UNetNodeTransformer = _strict_init_class(trainer.UNetNodeTransformer, args.init_full)
    sys.argv = ["train_unet_transformer.py", *trainer_args]
    trainer.main()


if __name__ == "__main__":
    main()
