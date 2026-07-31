# Kaggle submission — celltrack

`biohub-cell-tracking-during-development` is a **Research Code Competition**: the leaderboard **only accepts
submissions from Notebooks/Scripts**, never a file upload. A `kaggle competitions submit -f submission.csv`
returns `400 FAILED_PRECONDITION "only accepts Submissions from Notebooks"`. Every LB number goes through a
**script kernel** that runs against the hidden test set with no internet.

## The method (lib-as-dataset + light kernel)

1. **`celltrack-kit` dataset** (`dimiturnt/celltrack-kit`) — our own `celltrack`/`core` source tree, the
   trained detector weights, and the wheels for the few deps missing from Kaggle's base image
   (`torch`/`numpy`/`scipy`/`polars` are already there; we ship `zarr`, `numcodecs`, `jaxtyping`, `beartype`,
   `donfig`, `google_crc32c`). No code is duplicated into the kernel.
2. **A light script kernel** mounts the kit + the competition data, `pip install --no-index --no-deps *.whl`,
   adds the source root to `sys.path`, imports the pipeline, and writes `/kaggle/working/submission.csv`.

## Rebuild + submit

```bash
# 1. assemble the kit (source + weights + wheels)
uv run python kaggle/build_kit.py \
    --wheels <dir-of-wheels> \
    --processed <paths.yaml data root>/processed/biohub_cell_tracking \
    --weights detector_bce_aug.pt

# 2. push a new dataset version  (run from kaggle/ so the -p path has no slash — a Windows kaggle-CLI bug
#    corrupts the temp upload filename when -p contains "/")
cd kaggle && kaggle datasets version -p kit -m "<message>"

# 3. push + run the kernel  (metadata file must be named kernel-metadata.json)
cd kaggle && kaggle kernels push -p kernels

# 4. watch it, then submit its output to the competition
kaggle kernels status dimiturnt/celltrack-aug8k-nn
kaggle kernels output dimiturnt/celltrack-aug8k-nn -p /tmp/out   # sanity-check submission.csv
# submit the completed kernel version on the competition's "Submit" page (code-competition flow)
```

## Kernels

Named for the method, `celltrack-<detector>-<linker>` — a different detector or linker is a different
kernel, not a new version of an existing one. The version history of one slug is the same method retrained
or re-tuned; the slug itself says which method it is.

- `kernels/celltrack-aug8k-nn.py` — aug-recipe width-16 U-Net (`detector_bce_aug_8k.pt`) → NN linker →
  short-track → linefit.
- The classical baseline (`dimiturnt/celltrack-classical-submit`, LoG detector → NN linker, scored **0.427**)
  is the proven reference for the path.

## Gotchas that cost real time

- **T4, not the default P100** — set `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json`. Kaggle's
  default GPU is a P100 (`sm_60`, Pascal), which our torch build (2.10, compiled for `sm_70`+) refuses with
  *"no kernel image is available for execution on the device"*. The `--accelerator` CLI flag does **not**
  override it; only the metadata field does. Valid tokens: `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`.
- **fp16 inference, not bf16.** The forward autocasts in fp16 (CUDA autocast's default — `torch.autocast(
  device_type="cuda")` with no `dtype`). Do **not** switch it to bf16: the peak NMS collapses a plateau of
  *bit-identical* maxima to one centre, and bf16's 8-bit mantissa rounds distinct neighbouring peaks into
  false plateaus that then merge away, halving the score. bf16 is a *training*-only choice (gradient range).
- **keep-N vs threshold is now a free choice.** The kernel reads out with `peaks.centres(response, ~250)` — a
  bounded per-frame budget. It used to be *mandatory*: a saturated detector's flat plateaus made threshold-NMS
  emit ~1e5 "peaks" whose `N×N` linker `cdist` swapped the machine to death. `peaks.py` now collapses each
  plateau to one centre (connected components), so `above_threshold` is safe too; keep-N stays for count control.
- **`kaggle` CLI is `.venv/Scripts/kaggle.exe`, not `python -m kaggle`.** The repo's own `kaggle/` directory
  shadows the installed package, so `import kaggle` / `python -m kaggle` resolves to our source tree and fails.
  Call the venv entrypoint directly (or run from outside the repo root).
- **GPU on** (`enable_gpu: true`) — the full-frame 3-D forward is far too slow on CPU.
- **No internet** — every dependency must be a wheel in the kit.
- **`enable_gpu` kernels queue** — a run is not instant; poll `kaggle kernels status`.
