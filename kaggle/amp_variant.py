"""Stage a kernel variant whose detector encode runs under CUDA autocast (T4: fp16 tensor cores, no native bf16).

The patch lands on the predict script after the kernel has checksummed and patched it itself, right before the
first prediction, so every model the kernel runs (primary, secondary, validator) encodes in the given dtype:

    python kaggle/amp_variant.py --base-kernel kaggle/kernels/celltrack-public-0947 \
        --out-kernel kaggle/kernels/celltrack-public-0947-fp16 --dtype float16
"""

import argparse
import json
import logging
from pathlib import Path

from local_predict_patch import AMP_ENCODE, MODEL_READY

log = logging.getLogger(__name__)

OWNER = "dimiturnt"
FIRST_PREDICT = "splits_path = REPO_DIR / 'kaggle_test_splits_50ep.json'\n"


def amp_cell_patch(dtype: str) -> str:
    return (
        "_amp_ps = REPO_DIR / 'scripts' / 'predict_unet_transformer.py'\n"
        "_amp_s = _amp_ps.read_text()\n"
        f"assert _amp_s.count({MODEL_READY!r}) == 1, 'predict script model-ready anchor moved'\n"
        f"_amp_ps.write_text(_amp_s.replace({MODEL_READY!r}, {AMP_ENCODE.replace('{dtype}', dtype)!r}))\n"
        f"print('AMP encode patched: {dtype}')\n"
    )


def patch_notebook(notebook: dict, dtype: str) -> dict:
    cells = [c for c in notebook["cells"] if FIRST_PREDICT in "".join(c["source"])]
    assert len(cells) == 1, "expected one cell holding the first test prediction"
    source = "".join(cells[0]["source"])
    assert source.count(FIRST_PREDICT) == 1, "expected one test-split line ahead of the first prediction"
    cells[0]["source"] = source.replace(FIRST_PREDICT, amp_cell_patch(dtype) + FIRST_PREDICT).splitlines(keepends=True)
    return notebook


def stage_kernel(base: Path, out: Path, dtype: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((base / "kernel-metadata.json").read_text())
    notebook = json.loads((base / meta["code_file"]).read_text(encoding="utf-8"))
    name = out.name
    meta.update(id=f"{OWNER}/{name}", title=name, code_file=f"{name}.ipynb")
    (out / meta["code_file"]).write_text(json.dumps(patch_notebook(notebook, dtype), indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-kernel", type=Path, required=True)
    parser.add_argument("--out-kernel", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    stage_kernel(args.base_kernel, args.out_kernel, args.dtype)
    log.info("staged %s (%s encode)", args.out_kernel, args.dtype)


if __name__ == "__main__":
    main()
