#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.lab/hf}" PYTHONUNBUFFERED=1
export PY="${PY:-$PWD/.venv/bin/python}" SEQS=16
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p outputs
exec 9>outputs/.evaluation.lock
flock -n 9 || { echo 'An evaluation is already running in this package.' >&2; exit 1; }
if [[ $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run_test.sh [seed=42] [unique_run_tag]' >&2
  exit 2
fi
SEED=${1:-42}
TAG=${2:-base_test_seed${SEED}}
MODEL=$("$PY" scripts/protocol.py base-path)
"$PY" scripts/protocol.py begin "$MODEL" "$TAG" --seed "$SEED"
# Empty model field intentionally preserves the original pinned baseline path.
bash scripts/run_eval_dpo2e.sh "$TAG:$SEED:32768::records/mmmu_test.json:1"
"$PY" scripts/protocol.py verify "$TAG"
