#!/usr/bin/env bash
# MMMU Test 10500: baseline or merged model, 32768 output tokens, concurrency 16.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run_test.sh <baseline|/absolute/path/to/merged> unique_run_tag [generation_seed]' >&2
  exit 2
fi
TAG=$2
GENERATION_SEED=${3:-42}
export HF_HOME="${HF_HOME:-$PWD/.lab/hf}" PYTHONUNBUFFERED=1
export PY="${PY:-$PWD/.venv/bin/python}" SEQS=16
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
command -v flock >/dev/null 2>&1 || { echo "flock (util-linux) is required; use a Linux environment." >&2; exit 2; }
mkdir -p outputs
exec 9>outputs/.evaluation.lock
flock -n 9 || { echo 'An evaluation is already running in this package.' >&2; exit 1; }
profile_args=()
if [[ "$1" == baseline ]]; then
  MODEL=$("$PY" scripts/protocol.py base-path)
  GENERATION_MODEL=
  profile_args=(--baseline)
else
  MODEL=$(realpath -e -- "$1")
  GENERATION_MODEL=$MODEL
fi
"$PY" scripts/protocol.py begin "$MODEL" "$TAG" --generation-seed "$GENERATION_SEED" "${profile_args[@]}"
bash scripts/run_eval_dpo2e.sh "$TAG:$GENERATION_SEED:32768:$GENERATION_MODEL:records/mmmu_test.json:1"
"$PY" scripts/protocol.py verify "$TAG"
