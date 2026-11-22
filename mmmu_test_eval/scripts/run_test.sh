#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run_test.sh /absolute/path/to/merged [seed=42] [unique_run_tag]' >&2
  exit 2
fi
export HF_HOME="${HF_HOME:-$PWD/.lab/hf}" PYTHONUNBUFFERED=1
export PY="${PY:-$PWD/.venv/bin/python}" SEQS=16
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
command -v flock >/dev/null 2>&1 || { echo "flock (util-linux) is required; use a Linux environment." >&2; exit 2; }
mkdir -p outputs
exec 9>outputs/.evaluation.lock
flock -n 9 || { echo 'An evaluation is already running in this package.' >&2; exit 1; }
MODEL=$(realpath -e -- "$1")
SEED=${2:-42}
TAG=${3:-dpo2e_repro_test_seed${SEED}}
"$PY" scripts/protocol.py begin "$MODEL" "$TAG" --seed "$SEED"
bash scripts/run_eval_dpo2e.sh "$TAG:$SEED:32768:$MODEL:records/mmmu_test.json:1"
"$PY" scripts/protocol.py verify "$TAG"
