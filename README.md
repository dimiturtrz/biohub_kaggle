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

Two documents carry the substance, and they are the point of this repo for an outside reader:

- **[Post-mortem](interpretations/celltrack/converging/2026-09-30_private_lb_post_mortem.md)** — what we
  got right, what we got wrong, and why. Every division verdict we filed was true *for what it tested*
  (post-processing, constant sweeps, global ILP without a model) and the axis worth +0.054 of score was
  the one thing none of them tested. Includes a line-by-line grid of our kill list against the
  3rd-place ablation, plus the denominator and self-selected-ceiling traps that produced our two worst
  calls.
- **[3rd-place solution, written up faithfully](research/solutions/2026-09-30_yu4u_3rd_place.md)** —
  yu4u's private-0.967 pipeline in full: cross-architecture detector ensemble with a DoG-based
  sparse-annotation loss mask, self-supervised dense flow, a learned matcher, three division CNNs, and
  a joint link/division LP whose objective is a linearised surrogate of the competition metric itself.
  Their published A0→A7 ablation is the cleanest external ruler we have for our own conclusions.

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
