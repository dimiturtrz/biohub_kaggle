# Regenerating the metric fixtures

`tests/integration/oracle_agreement.py` checks our evaluator against the organizers' implementation on
frozen cases. The expectations in `tests/assets/metric_expected.json` come from **their** code, so the
check runs without their package installed — it is an oracle, not a dependency.

`generate_cases.py` writes the cases (voxel coordinates, anisotropic spacing, deliberately crowded frames
and dividing lineages) and needs nothing but numpy. `generate_expectations.py` scores them with
`tracking_cellmot` and needs the organizers' package.

```bash
git clone https://github.com/royerlab/kaggle-cell-tracking-competition.git \
  external/kaggle-cell-tracking-competition
git -C external/kaggle-cell-tracking-competition checkout 075fc5f5a52d11077f9dc2b074644618f26939e2

uv venv .oracle --python 3.12
uv pip install --python .oracle/Scripts/python.exe ./external/kaggle-cell-tracking-competition

uv run python tools/oracle/generate_cases.py
.oracle/Scripts/python.exe tools/oracle/generate_expectations.py
```

`external/` and `.oracle/` are gitignored; the checkout is reproducible from the pinned commit above.
Regenerate only when the organizers change the metric — a fixture diff is a change in what the
leaderboard rewards, and should be read as one.
