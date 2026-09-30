# Data manifest — what the data root held before the 2026-09-29 wipe

Recorded immediately before deleting 117.9 GB of the celltrack data root and repo build
artifacts, so a later re-download can be checked against what was actually here. Nothing in
this list is model weights: no checkpoint over 100 MB existed anywhere under the data root or
the repo at the time of writing.

`<data root>` is the `data:` entry of the gitignored `paths.yaml` (`core.paths.DataRoot`); on the
authoring machine that was a local drive, deliberately not spelled here.

## `<data root>/raw`

Total 81,70 GB across 24 915 files.

| child | GB | files |
|---|---:|---:|
| `biohub_cell_tracking` | 81,70 | 24 915 |

## `<data root>/valuable_other`

Total 14,69 GB across 1 840 files.

| child | GB | files |
|---|---:|---:|
| `biohub_cell_tracking` | 14,69 | 1 840 |

## `<data root>/synthetic`

Total 11,34 GB across 2 800 files.

| child | GB | files |
|---|---:|---:|
| `biohub_synthetic` | 11,34 | 2 800 |

## `<data root>/processed`

Total 3,16 GB across 25 124 files.

| child | GB | files |
|---|---:|---:|
| `biohub_cell_tracking` | 14,69 | 1 840 |
| `frontier_cache` | 0,41 | 7 720 |
| `synth_trainer_600` | 2,75 | 17 401 |
| *(loose files at root)* | 0,00 | 3 |

## Rebuild notes

- `raw` came from the Kaggle competition `biohub-cell-tracking-during-development`, which
  closed 2026-09-29. Re-fetch via the Kaggle CLI; if the data is gated after close, it is gone.
- `valuable_other/biohub_cell_tracking` is the public dense-GT set that the E-series refuted as
  not matching our embryos (step-length registration scored 0 on 7 needles x 4 volumes). Public,
  re-downloadable, and of no established value here.
- `synthetic` and `processed` are both ours and regenerate from `raw` plus the committed
  generators, so they die with `raw` and return with it.
- `.venv-repro` and `runs` were repo-local build and run artifacts, rebuilt by `uv sync` and
  by re-running the pipeline respectively.

