"""Assemble the celltrack-kit Kaggle dataset: our source tree + trained weights + offline dependency wheels.

The submission runs in a code competition with no internet, so everything the kernel imports has to be mounted.
This packs the current `celltrack`/`core` source (no vendoring — it is our own tree), the trained detector
weights, and the wheels for the few dependencies missing from Kaggle's base image (torch/numpy/scipy/polars are
already there) into a directory, writes the dataset metadata, and prints the push command.
"""

import argparse
import json
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DATASET_ID = "dimiturnt/celltrack-kit"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_REPO / "kaggle" / "kit")
    ap.add_argument("--weights", nargs="+", default=["detector_bce_aug.pt"], help="weight files under processed/")
    ap.add_argument("--wheels", type=Path, required=True, help="dir holding the offline dependency wheels")
    ap.add_argument("--processed", type=Path, required=True, help="processed data dir holding the weight files")
    arguments = ap.parse_args()

    out = arguments.out
    if out.exists():
        shutil.rmtree(out)
    (out / "celltrack_src").mkdir(parents=True)
    for package in ("celltrack", "core"):
        shutil.copytree(_REPO / package, out / "celltrack_src" / package, ignore=shutil.ignore_patterns("__pycache__"))
    for wheel in sorted(arguments.wheels.glob("*.whl")):
        shutil.copy2(wheel, out / wheel.name)
    for weight in arguments.weights:
        shutil.copy2(arguments.processed / weight, out / weight)
    (out / "dataset-metadata.json").write_text(
        json.dumps({"title": "celltrack-kit", "id": _DATASET_ID, "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )
    wheels = [w.name for w in out.glob("*.whl")]
    weights = [w for w in arguments.weights]
    print(f"kit at {out}: src=celltrack,core wheels={wheels} weights={weights}")
    print(f"push a new version:\n  kaggle datasets version -p {out} -m '<message>' --dir-mode zip")


if __name__ == "__main__":
    main()
