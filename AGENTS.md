# celltrack — agent instructions

Keep this file and `CLAUDE.md` IN SYNC (different harnesses read each; same content).
## Issue tracking — beads (bd)

This project uses **bd (beads)** for issue tracking. Run `bd init` once, then `bd prime` for the full
workflow. `.beads/issues.jsonl` is git-tracked; the Dolt engine is gitignored.

- `bd ready` — available work · `bd show <id>` — details · `bd update <id> --claim` · `bd close <id>`
- Use `bd` for ALL task tracking — not TodoWrite or markdown TODO lists.

## Build & Test

`uv` is the env + dependency manager; `nox` is the task runner (every session calls `uvx`/`uv run`
so local runs match CI exactly).

```bash
uv sync --extra dev --extra devtools   # create/refresh .venv
nox -s lint test cov                   # the merge gate, locally
nox -s gates                           # same, one entrypoint
```

## Package structure

Your code lives in your declared packages: `core/`, `celltrack/` — organise modules however the project needs.
The scaffold ships the **guardrails**, not a layering. `graph.py --assert` keeps each package's
internal import graph honest (no cycles, no god-modules, no oversized files, no unmirrored logic modules);
thresholds live in `[tool.structure]`.

Directional layer rules (the kernel imports none of the others; a viewer never imports a trainer) are
enforced by **import-linter** — a one-way forbidden import is no cycle, so it's the axis `graph.py` can't
see. Contracts live in `[tool.importlinter]` (a `# >>> LOCAL-SLOT: import-contracts` region you extend),
run via `lint-imports` in nox/CI/pre-commit.

## Static-analysis gates — graduated, never suppressed

Each gate is engine + params + invocation, run identically in `noxfile.py` and CI. Philosophy: a check
starts advisory, and once the code is clean it **graduates** to enforced; from then on any regression
fails. Fix the finding, don't suppress it; suppressions stay minimal and meaningful (a `# noqa: RULE` with
no prose, never a blanket ignore). Thresholds are legislated values (file_max, CC 10), never seeded from
the repo's current state. The linter is a bug-finder, not cosmetics.

- **ruff** (pinned `0.15.13`) — `check` on the full house select, enforced; `format --check` advisory.
  Encodes: logging not prints (T20), named constants not magic numbers (PLR2004), keyword-only bools
  (FBT), specific-exception-or-crash (BLE001/S), low complexity (C90/PLR), imports-at-top (PLC0415).
- **vulture** (pinned `2.16`) — dead-code at `min_confidence = 60`.
- **coverage** — pytest-cov with a `--fail-under=90` floor.
- **import-linter** — directional forbidden-import contracts (`[tool.importlinter]`); the layer axis graph.py can't express.
- **arch-fitness** — `python -m devtools.graph --assert`: god-module (fan-in/out), god-file, import-cycle,
  and test-mirror (a logic module needs its `tests/unit/<pkg>/…/test_<name>.py`); thresholds in
  `[tool.structure]`. Advisory: line-floor (off) + chokepoint.
- **archmap** (advisory / doc-gen) — `python -m devtools.graph.archmap <packages>` (or `nox -s archmap`) emits
  `docs/architecture/graph.json` (the committed, diffable architecture — nodes + weighted import edges) plus
  a self-contained interactive **cytoscape viewer** (`index.html`, regenerated + gitignored) that folds/expands
  packages to any depth and focuses a module's neighbourhood. Regenerate with `nox -s archmap` and **commit the
  graph.json diff** — it's the architecture-erosion record. The pre-commit hook regenerates on commit; CI runs
  `--check` (advisory) to flag a stale graph.json. **Publishing** (one repo = one Pages site): if the repo has
  no other Pages deployer, opt in with `archviz_pages` — it owns a staged site (`/architecture/` = main,
  `/architecture/preview/` = dev, `/` redirects). If the repo ALREADY deploys Pages, do NOT enable it; fold
  archmap into the existing workflow as an `/architecture/` subpath instead (see README: "Compose archmap into
  an existing Pages site"). Doc-gen, not enforcement — import-linter is the directional gate.
- **shape-contracts** (advisory) — `python -m devtools.shape_contracts`: a public array/tensor boundary must
  carry a **jaxtyping** shape (`Float[Tensor, "b c h w"]`), not a bare `np.ndarray`/`Tensor`. Aliases in
  `[tool.shape_contracts]`; make shapes live at runtime with a `@shapecheck` decorator (jaxtyping+beartype,
  see `devtools/README.md`). Migrate boundaries, then graduate to `--assert` to block.
## Scaffolding

Guardrails provisioned by **sdlc-scaffold** via copier — `.copier-answers.yml` pins the version. The gate
config, `devtools/`, and the nox/CI/pre-commit runners are **template-owned**: don't hand-edit them to pass
a gate — fix upstream in the scaffold and `uvx copier update`, or edit only within `# >>> LOCAL-SLOT`
regions. `copier update` pulls scaffold improvements as reviewable steps.

**Don't test template-owned code, either.** The `devtools/` detectors are unit-tested in the scaffold
itself; a `tests/unit/devtools/` here would re-litigate every template change locally (it duplicates
upstream coverage of byte-identical code, and breaks on the next `copier update`). Trust the upstream
tests — this repo's tests cover only ITS packages. Consumer-specific detector *config* (e.g. that your
`omit` list exempts your shells) is proven through the detector's public behavior, not a copied unit test.

## Conventions

- **Data lives out of the repo** (under a root pointed at by `paths.yaml`, gitignored). No datasets,
  weights, or run outputs in git.
- **No notebooks** (`.ipynb`) — a one-off run becomes a committed CLI / `python -m` entrypoint
  (reproducible + reviewable), never a REPL session.
- **Minimal comments** — prefer self-documenting names; a comment is a liability, not a default.
- **Config objects over long arg lists**; strategy pattern over `if`/type dispatch.
- **Cost-tiered experimenting** — iterate on the cached fold-0 eval (seconds–min); a ~1h train or the
  3-4h Kaggle hidden eval is an async confirm gated on that proxy. Kaggle is 5/day — batch candidates
  async, don't drip-and-wait.
- **Score noise floor** — a proxy or LB difference **below ~0.01–0.02 is noise, not signal**. Don't
  keep/kill a method, ship a default, or claim a win on a sub-0.01 delta; treat it as a tie and decide on
  mechanism (or a bigger, repeated gap). Only differences clearing the floor count as real movement.
- **Proxy inverts at the ceiling — gate the submission on MECHANISM, not a proxy number that beats 0.900.**
  The local proxy is FAITHFUL below ~0.89 (monotone, small negative offset) and BLINDS/INVERTS at the
  saturated top: dw0 scored +0.0267 held-out yet −0.028 on the LB (0.872 vs 0.900). So a held-out win
  above 0.90 is NOT evidence for the LB. Spend the one submission only on a named mechanism plus a recall
  axis the champion structurally lacks, or on decorrelation you can measure locally (edge-agreement between
  trackers) — never on a top-end proxy number alone.
- **Doc layers** — `learning/<date>_<topic>.md` = the study ramp / general understanding;
  `research/` = external / field synthesis (theirs); `interpretations/<task>/<date>_<topic>.md` =
  sense-making of *our own* results (per task, `converging/` for cross-task). The build log is git history;
  portfolio READMEs carry the result + link out.



<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:3216161c -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
