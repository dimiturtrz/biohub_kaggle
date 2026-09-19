"""Run a Kaggle kernel notebook locally on this machine's GPU — its own held-out validator + PP sweep included.

`/kaggle/...` paths are remapped onto the local checkouts (`external/frontier_ds/<slug>` for pilkwang datasets,
the `paths.yaml` data root for the competition); an optional probe spec swaps the PP candidates (see
donor_probe.py). The kernel's resume states live in the work dir, so a re-run with a new spec reuses the cached
test + validator GPU predictions and only re-runs the CPU post-processing sweep:

    cd kaggle; ../.venv/Scripts/python local_kernel.py --kernel kernels/celltrack-public-0947 \
        --spec probes/reid3_steal.json --work ../runs/local_kernel/0947
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from donor_probe import rewrite

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
FRONTIER_DATASETS = REPO / "external" / "frontier_ds"
PREDICTION_CACHE = REPO / "runs" / "local_kernel" / "_prediction_cache"
COMPETITION = "biohub-cell-tracking-during-development"
FUTURE_IMPORT = "from __future__ import annotations\n"


def competition_dir() -> Path:
    config = yaml.safe_load((REPO / "paths.yaml").read_text())
    return Path(config["data"]) / "raw" / "biohub_cell_tracking"


def remap(source: str, work: Path) -> str:
    mapping = (
        [(f"/kaggle/input/competitions/{name}", competition_dir()) for name in (COMPETITION, "{COMPETITION}")]
        + [(f"/kaggle/input/{name}", competition_dir()) for name in (COMPETITION, "{COMPETITION}")]
        + [
            ("/kaggle/input/datasets/pilkwang", FRONTIER_DATASETS),
            ("/kaggle/input", FRONTIER_DATASETS),
            ("/kaggle/working", work),
        ]
    )
    for kaggle_path, local_path in mapping:
        source = source.replace(kaggle_path, local_path.as_posix())
    assert "/kaggle/" not in source, "unmapped /kaggle/ path left in kernel source"
    return source


def kernel_source(kernel: Path) -> str:
    meta = json.loads((kernel / "kernel-metadata.json").read_text())
    notebook = json.loads((kernel / meta["code_file"]).read_text(encoding="utf-8"))
    source = "\n\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    return FUTURE_IMPORT + source.replace(FUTURE_IMPORT, "")


def guard_literal(value: str) -> str:
    try:
        return repr(float(value))
    except ValueError:
        return json.dumps(value)


def pin_env(source: str, overrides: dict[str, str]) -> str:
    """Pin kernel env vars, and the kernel's config-drift guard entry for each, so the guard expects the pin."""
    for key, value in overrides.items():
        assignment = re.compile(rf"os\.environ\[(['\"]){key}\1\] = .*")
        source = assignment.sub(lambda _, k=key, v=value: f"os.environ[{k!r}] = {v!r}", source)
        guard_entry = re.compile(rf"^(    \"{key}\": )(?:[-0-9.eE]+|\"[^\"]*\"),$", re.MULTILINE)
        source = guard_entry.sub(lambda m, v=value: f"{m.group(1)}{guard_literal(v)},", source)
    return source


def share_gpu(source: str, workers: int) -> str:
    """Point the kernel's multi-GPU shard path at N workers on this machine's single GPU."""
    two_gpu_cap = re.compile(r"min\(2, (?:_torch\.cuda\.device_count\(\)|available_gpu_count), (len\(\w+\))\)")
    assert len(two_gpu_cap.findall(source)) == 2, "expected the test + validator worker-count lines"  # noqa: PLR2004
    source = two_gpu_cap.sub(rf"min({workers}, \1)", source)
    one_token_per_gpu = "return [str(index) for index in range(count)]"
    assert source.count(one_token_per_gpu) == 1, "expected one CUDA token helper"
    return source.replace(one_token_per_gpu, 'return ["0"] * count')


def cache_predictions(source: str) -> str:
    """Patch the predict script once the kernel has checksummed and patched it itself, before it first runs."""
    first_predict = 'splits_path = REPO_DIR / "kaggle_test_splits_50ep.json"\n'
    assert source.count(first_predict) == 1, "expected one test-split line ahead of the first prediction"
    return source.replace(first_predict, '__import__("local_predict_patch").apply(REPO_DIR)\n' + first_predict)


def shard_spec(spec: dict, shard: str) -> dict:
    index, count = map(int, shard.split("/"))
    labels = list(spec["candidates"])[index::count]
    return {
        "candidates": {label: spec["candidates"][label] for label in labels},
        "diagnose": spec["diagnose"] if index == 0 else {},
    }


def stage(kernel: Path, work: Path, options: argparse.Namespace, overrides: dict[str, str]) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    source = cache_predictions(pin_env(kernel_source(kernel), overrides))
    if options.gpu_workers > 1:
        source = share_gpu(source, options.gpu_workers)
    if options.spec is not None:
        spec = json.loads(options.spec.read_text())
        if options.candidate_shard:
            spec = shard_spec(spec, options.candidate_shard)
        source = rewrite(source, spec)
    script = work / "kernel.py"
    script.write_text(remap(source, work), encoding="utf-8")
    return script


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="pin a kernel env var")
    parser.add_argument("--gpu-workers", type=int, default=1, help="prediction shards sharing the one GPU")
    parser.add_argument("--candidate-shard", metavar="I/K", help="run only every K-th spec candidate from I")
    parser.add_argument("--prediction-cache", type=Path, default=PREDICTION_CACHE)
    parser.add_argument("--batched-tta", action="store_true", help="one batched encode for the 7 TTA views")
    parser.add_argument("--amp", action="store_true", help="bf16 autocast around the detector encode")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    overrides = dict(pair.split("=", 1) for pair in args.env)
    script = stage(args.kernel, work, args, overrides)
    log.info("staged %s", script)
    if args.stage_only:
        return
    env = {
        **os.environ,
        **overrides,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]),
        "LOCAL_PREDICTION_CACHE": str(args.prediction_cache.resolve()),
        "LOCAL_WORK_DIR": str(work),
        "LOCAL_BATCHED_TTA": "1" if args.batched_tta else "0",
        "LOCAL_AMP": "1" if args.amp else "0",
    }
    subprocess.run([sys.executable, script.name], cwd=work, env=env, check=True)


if __name__ == "__main__":
    main()
