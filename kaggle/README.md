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

# 3. push + run a kernel  (one method per subdir; metadata file must be named kernel-metadata.json)
cd kaggle && kaggle kernels push -p kernels/celltrack-pilktunet-motion-stlf

# 4. watch it, then submit its output to the competition
kaggle kernels status dimiturnt/celltrack-pilktunet-motion-stlf
kaggle kernels output dimiturnt/celltrack-pilktunet-motion-stlf -p /tmp/out   # sanity-check submission.csv
# submit the completed kernel version on the competition's "Submit" page (code-competition flow)
```

## Datasets

Kaggle-dataset staging trees live under `datasets/<slug>/` (each holds its own `dataset-metadata.json` +
`README.md`; upload with `kaggle datasets version -p datasets/<slug>`).

- `datasets/biohub-association-ranker` — vendored CC0 local-association-ranker weights (third-party
  pilkwang asset, re-hosted as `dimiturnt/biohub-association-ranker`). See its README for provenance.

## Kernels

Named for the method, `celltrack-<detector>-<linker>` — a different detector or linker is a different
kernel, not a new version of an existing one. The version history of one slug is the same method retrained
or re-tuned; the slug itself says which method it is.

Kernels are named systematically `celltrack-<S1>-<S2>-<S3>` (detection-linking-postproc) from the closed
vocabulary in [`SUBMISSIONS.md`](SUBMISSIONS.md) — the name is the method tuple, and that file's coverage
grid is what is tried vs untried. Each kernel is a subdir `kernels/<slug>/` holding `<slug>.py` + its
`kernel-metadata.json`.

- `celltrack-pilktunet-motion-stlf` — pilkwang `TemporalUNet3D` (from the support pack, no wheels) → motion
  linker → short-track + linefit. Fold-0 local **0.877**.
- `celltrack-unetaug-nn-stlf` — our width-16 aug U-Net (`detector_bce_aug_8k.pt`) → NN → short-track +
  linefit. Fold-0 local 0.472.
- The classical baseline (`dog`-`nn`-`raw`, legacy slug `celltrack-classical-submit`) scored **0.427**.

## Gotchas that cost real time

- **T4, not the default P100** — set `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json`. Kaggle's
  default GPU is a P100 (`sm_60`, Pascal), which our torch build (2.10, compiled for `sm_70`+) refuses with
  *"no kernel image is available for execution on the device"*. The `--accelerator` CLI flag does **not**
  override it; only the metadata field does. Valid tokens: `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`.
  **Do not invent one** — an unrecognised token is not rejected, it silently degrades to the P100, and the
  only symptom is that `sm_60` error thrown hours later from the first forward (this cost a run: `NvidiaTeslaT4x2`).
- **`NvidiaTeslaT4` gives you TWO T4s.** Measured by `celltrack-gpu-probe`: `device_count() == 2`, each a
  `Tesla T4 sm_75` with 14.6 GB, torch arch list `sm_70…sm_120`. `MultiGpuSubmission` shards the videos over
  both, so there is no separate "x2" shape to ask for. Accelerator questions are answerable in a minute —
  kernel *pushes* are unlimited, only *submissions* are 5/day, so probe rather than learn it 6 hours in.
- **fp16 inference, not bf16.** `celltrack/precision.py` (`AutocastPolicy`) wraps both GPU stages — the
  detection forward and the edge affinity — and casts back to fp32 for the read-out. Do **not** switch it to
  bf16: the peak NMS collapses a plateau of *bit-identical* maxima to one centre, and bf16's 8-bit mantissa
  rounds distinct neighbouring peaks into false plateaus that then merge away. bf16 is a *training*-only
  choice (gradient range); a forward has no gradients, so it buys nothing and costs mantissa. Measured on the
  four-movie proxy: fp32 0.9334, fp16 0.9334 (identical), bf16 0.9309. The T4's `sm_75` has no bf16 anyway.
  *This paragraph once described the deleted `LearnedDetector` and stayed true-sounding for weeks after the
  pilkwang graft dropped the autocast — the forward really did run fp32 until it was restored. If you change
  the inference dtype, change this line in the same commit.*
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
