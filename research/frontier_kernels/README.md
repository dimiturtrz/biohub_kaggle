# Donor kernels — a manifest, not a copy

The notebooks studied during this campaign are **other competitors' work**, published on Kaggle. They were
pulled locally to read, and they are **not redistributed here**: `research/frontier_kernels/*/` is gitignored.
This file records *what* was read, *whose* it is, and *how to fetch it again*, which is the same rule this repo
applies to every third-party dependency (see `CLAUDE.md`: checked out, never vendored).

Nothing below is a licence grant. Each notebook's licence is whatever its author set on Kaggle; the
`kernel-metadata.json` that `kaggle kernels pull` produces does **not** carry a licence field, so none of these
can be assumed permissive. Read them on Kaggle, under their own terms. The repo's own `LICENSE` (MIT) covers
only code written here.

Commits before `04e7000` still contain the notebook files: they were untracked, not purged from history, because
every one of them is a *publicly published* Kaggle kernel and rewriting the branch would cost the whole build
log for ~15 MB. If an author would rather their kernel not be reachable here at all, say so on the repo's issues
and it will be removed from history (`git filter-repo --path research/frontier_kernels/ --invert-paths`).

## Fetching

```bash
uv run kaggle kernels pull <owner>/<slug> -p research/frontier_kernels/<slug> --metadata
```

Add `-w` to also pull the notebook's output. A kernel whose owner is listed as *unrecorded* was pulled before
this manifest existed and its metadata file was not kept — find it by searching the slug on Kaggle, or via the
competition's Code tab.

## What was read, and what came out of it

Our reading of these is in `research/solutions/` (per-solution write-ups),
`research/frontier_kernels/SYNTHESIS_2026-08-25_genuine_vs_hack.md` (which elements were genuine vs
metric-exploiting), and `interpretations/celltrack/converging/2026-08-27_conclusion_tree.md` (what we kept,
killed, and why). Those documents are ours and stay in git.

Most of this family descends from one public base — `pilkwang`'s DeepCenter/Temporal UNet3D weights plus the
`biohub-tracking-support-pack-50ep` dataset — which is why so many rows below declare the identical three
dataset sources. That shared ancestry is itself a finding: see
`celltrack-e60-frontier-kernels-are-our-champion-downgraded` in the campaign notes.

| Local dir | Kaggle `owner/slug` | Title |
|---|---|---|
| `0-947-lb-biohub-deepcenter-ilp-tracker` | `beraterolelk/0-947-lb-biohub-deepcenter-ilp-tracker` | 🔬 [0.947 LB] BioHub DeepCenter + ILP Tracker |
| `3d-kinematic-cell-tracking-mitotic-bifurcation` | `avikdas567/3d-kinematic-cell-tracking-mitotic-bifurcation` | 3D Kinematic Cell Tracking & Mitotic Bifurcation |
| `biohub-0947-short5-prepp-r1` | `howonkang/biohub-0947-short5-prepp-r1` | biohub-0947-short5-prepp-r1 |
| `biohub-095-owned-validation` | `codezzzsleep/biohub-095-owned-validation` | biohub-095-owned-validation |
| `biohub-cell-tracking-tabpfn-3-5-events` | `noisyislands/biohub-cell-tracking-tabpfn-3-5-events` | Biohub Cell Tracking TabPFN 3.5 Events |
| `biohub-ct-sp402` | `yongjilyu/biohub-ct-sp402` | biohub-ct-sp402 |
| `biohub-dae-alpha-0-17` | `ghazarosbarseghyan91/biohub-dae-alpha-0-17` | Biohub DAE Alpha 0 17 |
| `biohub-deepcenter-unet3d` | `gautiermarti/biohub-deepcenter-unet3d` | Biohub deepcenter unet3d |
| `biohub-dodecatiad` | `fabriciodasilva/biohub-dodecatiad-cell-tracking` | BioHub Dodecatiad Cell Tracking |
| `biohub-frontier947-divprec-v1`, `thtennant_divprec` | `thtennant/biohub-frontier947-divprec-v1` | Biohub frontier947 divprec v1 |
| `biohub-frontier947-fast-det096-tight60-v1` | `thtennant/biohub-frontier947-fast-det096-tight60-v1` | Biohub frontier947 fast det096 tight60 v1 |
| `biohub-frontier947-flow2-v1` | `thtennant/biohub-frontier947-flow2-v1` | Biohub frontier947 flow2 v1 |
| `biohub-frontier947-gapfill-det096-v1` | `thtennant/biohub-frontier947-gapfill-det096-v1` | Biohub frontier947 gapfill det096 v1 |
| `thtennant_gapfill` | `thtennant/biohub-frontier947-gapfill-v1` | Biohub frontier947 gapfill v1 |
| `thtennant_readmit` | `thtennant/biohub-frontier947-readmit-v1` | Biohub frontier947 readmit v1 |
| `biohub-lb-942` | `chukkkk/biohub-lb-942` | Biohub LB 942 |
| `biohub-learned-unet-transformer-ilp-gap-recovery` | `binasalama/biohub-learned-unet-transformer-ilp-gap-recovery` | Biohub Learned UNet Transformer ILP Gap Recovery |
| `biohub-linker-association-mlp` | `noisyislands/biohub-linker-association-mlp` | Biohub Linker Association MLP |
| `biohub-reid3` | `arnav170/biohub-reid3` | biohub-reid3 |
| `biohub-retro` | `arnav170/biohub-retro` | biohub-retro |
| `biohub-v1-grouped` | `newwang12/biohub-v1-grouped` | biohub-v1-grouped |
| `biohub-v1-infer` | `leonixis/biohub-v1-infer` | Biohub v1 infer |
| `biohub-xgboost-division-events` | `noisyislands/biohub-xgboost-division-events` | Biohub XGBoost Division Events |
| `biohub-zhincez947-fork-v1` | `pawanmali/biohub-zhincez947-fork-v1` | biohub-zhincez947-fork-v1 |
| `biohub-zhincez947-nodiv-ablation-v1` | `pawanmali/biohub-zhincez947-nodiv-ablation-v1` | biohub-zhincez947-nodiv-ablation-v1 |
| `biohub-zsns-harvester` | `giorgosi/zsns005` | ZSNS005 |
| `evg0942` | `evgendvorkin/biohub-0-942-lb-proxy-score-0-9417` | Biohub 0.942 LB PROXY_SCORE=0.9417 |
| `evgendvorkin_biohub-0-927-lb` | `evgendvorkin/biohub-0-927-lb` | Biohub 0.927 LB |
| `kirneo_metric-hack-last-call-update` | `kirneo/metric-hack-last-call-update` | metric-hack-last-call-update |
| `rockerritesh_0-926-biohub-divsub` | `rockerritesh/0-926-biohub-divsub` | 0.926 Biohub divsub |
| `testnote` | `nightpool33/testnote` | TestNote |

### Owner not recorded

Pulled before this manifest existed, metadata not retained. Slug is the local directory name; search it on
Kaggle to find the author.

| Local dir | Slug |
|---|---|
| `biohub-0-95` | `biohub-0-95` |
| `biohub-geometric-fusion` | `biohub-geometric-fusion` |
| `biohub-harmonic-fusion` | `biohub-harmonic-fusion` |
| `biohub-harmonic-fusion-v3` | `biohub-harmonic-fusion-v3` |
| `biohub-lf-dctta` | `biohub-lf-dctta` |
| `biohub-lf-hoctveto-div-b` | `biohub-lf-hoctveto-div-b` |
| `biohub-lineage-forge-precision-tracking` | `biohub-lineage-forge-precision-tracking` |
| `biohub-metric-hack-last-call` | `biohub-metric-hack-last-call` |

Also pulled but not from this competition's Code tab — the published write-ups of the final standings, whose
authors are credited by placement in `research/solutions/`.
