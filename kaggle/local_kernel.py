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
import subprocess
import sys
from pathlib import Path

import yaml
from donor_probe import rewrite

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
FRONTIER_DATASETS = REPO / "external" / "frontier_ds"
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


def stage(kernel: Path, spec: Path | None, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    source = kernel_source(kernel)
    if spec is not None:
        source = rewrite(source, json.loads(spec.read_text()))
    script = work / "kernel.py"
    script.write_text(remap(source, work), encoding="utf-8")
    return script


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    script = stage(args.kernel, args.spec, work)
    log.info("staged %s", script)
    if args.stage_only:
        return
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
    subprocess.run([sys.executable, script.name], cwd=work, env=env, check=True)


if __name__ == "__main__":
    main()
