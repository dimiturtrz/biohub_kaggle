#!/usr/bin/env bash
# The data root for shell runners, resolved by the same rule as core/paths.py: `CELLTRACK_DATA` wins,
# else the `data:` entry of the gitignored paths.yaml. Source this from the repo root; it exports
# DATA_RAW and DATA_PROCESSED so no runner spells a machine-local path.
DATA_ROOT=${CELLTRACK_DATA:-$(sed -n 's/^data:[[:space:]]*//p' paths.yaml)}
[[ -n "$DATA_ROOT" ]] || { echo "no data root: set CELLTRACK_DATA or add 'data: <path>' to paths.yaml" >&2; exit 1; }
export DATA_ROOT DATA_RAW="$DATA_ROOT/raw" DATA_PROCESSED="$DATA_ROOT/processed"
