#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.lab/hf}" PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
if [[ -z "${PY:-}" ]]; then
  if [[ ! -x .venv/bin/python ]]; then
    "${PYTHON_BIN:-python3.12}" -m venv .venv
  fi
  PY="$PWD/.venv/bin/python"
fi
export PY
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
"$PY" -m pip install -r env/requirements_eval.txt
"$PY" -m pip check
"$PY" scripts/protocol.py runtime-check
"$PY" scripts/download.py base judge mmmu
"$PY" scripts/cache_test.py
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p records outputs
"$PY" scripts/build_records.py mmmu-test --out records/mmmu_test.tmp.json
"$PY" scripts/protocol.py records-check records/mmmu_test.tmp.json
mv records/mmmu_test.tmp.json records/mmmu_test.json
echo 'Ready: frozen MMMU Test records verified. See README.md for this branch run_test.sh arguments.'
