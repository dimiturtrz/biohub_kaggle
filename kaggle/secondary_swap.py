"""Swap the 0947 notebook's secondary (dual-seed) model for our own frontier-trainer checkpoint.

Stages a private dataset laid out like pilkwang's secondary artifact (manifest + weights/unet_transformer/split_0/)
and a kernel variant whose pinned secondary SHA256 and manifest path point at it:

    python kaggle/secondary_swap.py --run-dir <repo>/weights/<method>/split_0 --slug <dataset-slug> \
        --base-kernel kaggle/kernels/celltrack-public-0947 --out-kernel kaggle/kernels/celltrack-public-0947-<tag>
"""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

OWNER = "dimiturnt"
BASE_SECONDARY_SHA = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
BASE_SECONDARY_DATASET = "pilkwang/biohub-temporal-unet3d-seed314159-v1"
WEIGHT_RELATIVE = "weights/unet_transformer/split_0/edge_predictor_best.pth"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_dataset(run_dir: Path, slug: str, staging: Path) -> str:
    target = staging / Path(WEIGHT_RELATIVE).parent
    target.mkdir(parents=True, exist_ok=True)
    for name in ("edge_predictor_best.pth", "config.json"):
        shutil.copy2(run_dir / name, target / name)
    sha = _sha256(staging / WEIGHT_RELATIVE)
    manifest = {"artifact_name": slug, "model": {"weight_path": WEIGHT_RELATIVE, "weight_sha256": sha}}
    (staging / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    metadata = {"title": slug, "id": f"{OWNER}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    return sha


def stage_kernel(base: Path, out: Path, slug: str, sha: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((base / "kernel-metadata.json").read_text())
    notebook = (base / meta["code_file"]).read_text(encoding="utf-8")
    assert BASE_SECONDARY_SHA in notebook, "base notebook no longer pins the expected secondary SHA"
    notebook = notebook.replace(BASE_SECONDARY_SHA, sha).replace(BASE_SECONDARY_DATASET, f"{OWNER}/{slug}")
    name = out.name
    meta.update(id=f"{OWNER}/{name}", title=name, code_file=f"{name}.ipynb")
    meta["dataset_sources"] = [f"{OWNER}/{slug}" if s == BASE_SECONDARY_DATASET else s for s in meta["dataset_sources"]]
    (out / meta["code_file"]).write_text(notebook, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--base-kernel", type=Path, required=True)
    parser.add_argument("--out-kernel", type=Path, required=True)
    parser.add_argument("--staging", type=Path, default=Path("kaggle/kit_staging"))
    args = parser.parse_args()
    sha = stage_dataset(args.run_dir, args.slug, args.staging / args.slug)
    stage_kernel(args.base_kernel, args.out_kernel, args.slug, sha)
    log.info("staged %s (sha %s) -> %s", args.slug, sha[:12], args.out_kernel)


if __name__ == "__main__":
    main()
