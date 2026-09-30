# celltrack

[![ci](https://github.com/dimiturtrz/biohub_kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/dimiturtrz/biohub_kaggle/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

3D cell tracking in developing embryos — detection, association and lineage reconstruction over
100-frame light-sheet volumes. Built for the Kaggle **Biohub Cell Tracking During Development**
competition (closed 2026-09-29).

## Result

**Public 0.953 · private 0.918 · rank 562 / ~4000.** Winner 0.977, bronze cut ~0.955.

The private board shook hard: the public top was one open notebook plus several hundred forks of it,
and on the private embryo the whole fork field collapsed into 0.917–0.923 — four hundred ranks inside
five thousandths. Our own seven submissions spanned 0.945–0.953 public and landed in **0.916–0.918**
private, a 0.002 spread, i.e. our own noise floor. The cause is that the two test embryos differ in
cell density by 2–3× at the median, the public one being the dense one and worth only 21% of the data.

But flatness was not the whole story, and the honest version is less flattering. **Two teams reached
private 0.942 and 0.946 (gold) starting from the same public notebooks we forked, with no new detector** —
one by adding ~9 MB of gradient-boosted tabular heads, the other by *tuning* the public models and adding
repair stages. Our own fork-and-tune line has a measured ceiling of **0.924 private / rank 161**, so
~0.006 was available inside it from three CPU-only switches and one 7.3k-parameter head. Our central
thesis — that the gap was a dense-regime detector we could not train — was an existence-proof away from
being wrong, and we never ran the cheap experiment that would have shown it.

These documents carry the substance, and they are the point of this repo for an outside reader:

- **[Post-mortem](interpretations/celltrack/converging/2026-09-30_private_lb_post_mortem.md)** — what we
  got right, what we got wrong, and why: a line-by-line grid of our kill list against four published
  solutions, the exact metric currency (one missed division costs as much as **74 edges**, and divisions
  were **51%** of the deficit we were not measuring), the division knob that turned out to have **no
  consumer** (the public ILP cannot form a fork at its shipped weights — found independently by three
  teams), and the denominator and self-selected-ceiling traps that produced our worst calls.
- **[3rd place — yu4u, private 0.967](research/solutions/2026-09-30_yu4u_3rd_place.md)** —
  cross-architecture detector ensemble with a DoG-based sparse-annotation loss mask, self-supervised dense
  flow, a learned matcher, three division CNNs, and a joint link/division LP whose objective is a
  linearised surrogate of the competition metric itself. Their published A0→A7 ablation is the cleanest
  external ruler we have for our own conclusions.
- **[12th place — Corwin, private 0.946, gold](research/solutions/2026-09-30_corwin_12th_place.md)** — no
  new detector: tuned public models plus consolidation, a mitosis specialist, a nine-rule repair engine and
  a learned recentring CNN. They restored the host scorer and replay it to 1e-12, which makes theirs the
  most forensically useful document in the field — every event priced exactly, plus five dataset artefacts
  (copied frames, duplicated crops, per-embryo z conventions, integer coordinates, whole-stack frame jumps)
  nobody else published.
- **[18th place — ymg_aq, private 0.942](research/solutions/2026-09-30_ymg_aq_18th_place.md)** — the
  cheapest path in the field: ~9 MB of LightGBM / CatBoost / TabPFN heads over a frozen public detector,
  external zebrafish data for the division model, and labels defined as *every edge the official metric can
  judge* rather than only the GT edges.
- **[5th place — a division head *inside* the linker](research/solutions/2026-09-30_5th_place_division_head_in_the_linker.md)**
  — the measured price of the one mechanism our own division campaign left standing and never ran:
  **+0.024 private**, the largest single stage in a gold solution, plus detection confidence as the ILP cell
  cost (+0.020 private) and a staged ILP whose candidate set differs between the solve and the output.
- **[89th place — label the candidates with the official metric](research/solutions/2026-09-30_katsumata_89th_metric_labelled_divisions.md)**
  — the one team that made the division term pay **on the same public chassis we forked**, with no new
  detector and no new linker: an 18-feature logistic regression on 157 candidates, labelled by *replaying the
  official scorer*, worth **+0.008 private**. Includes the break-even algebra (`p* = J/(1+J)`) and the
  cleanest published demonstration that a mechanism transfers while the knob fitted to the public board does
  not.
- **[hjyact — from scratch to private 0.939](research/solutions/2026-09-30_hjyact_from_scratch_private_0939.md)**
  — the most useful document for auditing our own conclusions: it independently reproduces **six** of our
  refutations from a different architecture and harness, contradicts exactly one (and the contradiction
  resolves), and names the cheap detector fix we never tried — a **local-contrast input channel**, which took
  missed cells' local maxima from 4.2% to 14.2%.
- **[14th place — vibes and edges](research/solutions/2026-09-30_vibes_and_edges_14th_place.md)** — a
  per-voxel flow field for segmentation (`FlowSeg`), correspondence *regression* as the division model
  (`divflow`), a GraphSAGE edge rescorer, and a per-video circuit breaker keyed on detector health. Also the
  arithmetic proving the public ILP cannot fork, independently of the three teams in the post-mortem.
- **[The coordinate-head axis](research/solutions/2026-09-30_the_coordinate_head_axis.md)** — five teams, one
  ≈**+0.007 private** lever, and three genuine contradictions between them resolved by mechanism: constant
  versus learned shifts, in-sample replay versus the board, and why averaging two heads under-corrects.
- **[External data — what each source was actually worth](research/solutions/2026-09-30_external_data_and_what_it_was_worth.md)**
  — it splits by *which component you pretrain* (linker +0.016 private, detector nil), the highest-yield
  transfer in the field was an **architecture** rather than a dataset, and a CC0 set with **165,267 labelled
  divisions** sat on this competition's own forum for two months.
- **[The fork-and-tune line](research/solutions/2026-09-30_the_fork_and_tune_line.md)** — the line we were
  actually on, measured by two other teams: its 0.924-private ceiling, the three switches worth +0.002
  private each, the reproduction of the 0.953 notebook's "private" coordinate head, and six silent harness
  bugs each worth more than any hyper-parameter.

Method development, refutations and dead ends live in `interpretations/celltrack/` (ours, with
[`converging/2026-08-27_conclusion_tree.md`](interpretations/celltrack/converging/2026-08-27_conclusion_tree.md)
as the single living synthesis), external/field synthesis in `research/`, and the study ramp in
`learning/`. Datasets, weights and run outputs are out of git by policy —
[`docs/data_manifest.md`](docs/data_manifest.md) records what the data root held and how to rebuild it.

## Layout

- `core/`, `celltrack/` — your packages. The guardrails target all of them; organise modules however the project needs.
- `tests/` — `unit/`, `integration/`, `e2e/`.
- `devtools/` — the guardrail tools (`graph.py` arch-fitness, plus the class-shape explorers).

The set of packages the gates lint/test/graph is the `packages` answer (comma-separated) — add more
without touching any gate file; every gate renders over the list. Directional layer contracts between
packages are **opt-in** (add import-linter yourself — see `devtools/README.md`), never imposed.

## Quickstart

```bash
git clone https://github.com/dimiturtrz/biohub_kaggle
uv sync --extra dev --extra devtools
nox -s lint test cov
```

## Gates

`nox -s lint test cov` (or `nox -s gates`) runs exactly what CI runs:

- **lint** — ruff check (pinned) + ruff format --check (advisory) + vulture + arch-fitness (`graph.py --assert`) + ast-grep + jscpd.
- **test** — pytest.
- **cov** — pytest --cov with a `--fail-under=90` floor.

## Architecture site

`nox -s archmap` writes `docs/architecture/graph.json` (committed, diffable — nodes + weighted import edges)
plus a self-contained interactive viewer (`index.html`, gitignored, rebuilt on demand). **Commit the
graph.json diff** — it's the architecture-erosion record.

▶ **Live viewer: [https://dimiturtrz.github.io/biohub_kaggle/architecture/](https://dimiturtrz.github.io/biohub_kaggle/architecture/)** — main; dev preview at [https://dimiturtrz.github.io/biohub_kaggle/architecture/preview/](https://dimiturtrz.github.io/biohub_kaggle/architecture/preview/)


Publishing to GitHub Pages — one repo = one Pages site, so pick ONE of:

- **Sole owner** — this repo has no other Pages deployer. Answer `archviz_pages` yes at scaffold time; the
  generated `pages.yml` owns a staged site (`/architecture/` = main, `/architecture/preview/` = dev, `/`
  redirects) and enables it. Nothing else to do.
- **Compose** — this repo ALREADY deploys Pages (a docs/demo site). Leave `archviz_pages` off and fold the
  archmap build into that existing workflow as an `/architecture/` subpath. Build it isolated (no project
  sync, no touch to the pinned devtools) — `write_viewer` only needs the vendored template + libs:

  ```yaml
  - uses: astral-sh/setup-uv@v5
  - name: Architecture view (/architecture/)
    run: |
      uv run --no-project --with "sdlc-devtools @ git+https://github.com/dimiturtrz/sdlc-scaffold.git@v1.26.0#subdirectory=sdlc-devtools" \
        python -c "from devtools.graph.archmap import Archmap; Archmap(['core', 'celltrack']).write_viewer(project='celltrack')"
      mkdir -p _site/architecture
      cp docs/architecture/graph.json docs/architecture/index.html _site/architecture/
  ```

  graph.json stays committed (the diff-truth); regen it locally with `nox -s archmap` when the import
  structure changes. (`../synthscape` is the reference — archmap folded into its `deploy-fit-explorer`
  workflow beside the fit-explorer views.)

## Development

Tooling + quality gates are provisioned by an in-house copier template (**sdlc-scaffold**). Refresh with
`uvx copier update`; `.copier-answers.yml` pins the template version.

### Local hooks

The same gates CI enforces are bound to git events via pre-commit. Install **both** stages:

```bash
pre-commit install                        # commit stage — fast static gates (ruff/vulture/arch-fitness/…)
pre-commit install --hook-type pre-push   # push stage — fast unit suite (tests/unit)
```

The pre-push hook runs `pytest tests/unit` so a change that lints clean but breaks a test **contract** (a
signature change a mirror test still calls the old way) is caught before the push, not after CI goes red. It
is deliberately push-only (not every commit) and unit-only (no integration/e2e); the coverage floor stays a
CI job. The default `pre-commit install` does **not** wire pre-push — run the second line once per clone.
