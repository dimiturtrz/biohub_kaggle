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
kaggle kernels status dimiturnt/celltrack-learned-submit
kaggle kernels output dimiturnt/celltrack-learned-submit -p /tmp/out   # sanity-check submission.csv
# submit the completed kernel version on the competition's "Submit" page (code-competition flow)
```

## Kernels

- `kernels/celltrack-learned-submit.py` — U-Net detector → NN linker → short-track → linefit.
- The classical baseline (`dimiturnt/celltrack-classical-submit`, LoG detector, scored **0.427**) is the
  proven reference for the path.

## Gotchas that cost real time

- **keep-N, never a probability threshold.** The detector saturates to flat plateaus at prob≈1; NMS on a
  plateau returns ~1e5 "peaks" per frame, and the linker's `N×N` cost matrix (`cdist`) then needs tens of GB
  → the machine swaps to death. `peaks.centres(response, keep)` caps the count (~250/frame) and the cost.
- **GPU on** (`enable_gpu: true`) — the full-frame 3-D forward is far too slow on CPU.
- **No internet** — every dependency must be a wheel in the kit.
- **`enable_gpu` kernels queue** — a run is not instant; poll `kaggle kernels status`.
