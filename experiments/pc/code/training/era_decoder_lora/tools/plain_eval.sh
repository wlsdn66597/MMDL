#!/usr/bin/env bash
# Explicit commands only. Never starts training or changes power/GPU settings.
set -euo pipefail
cd /home/rlatjswo/era_decoder_lora_20261030
export PYTHONDONTWRITEBYTECODE=1
python=/home/rlatjswo/mmdl_test/.venv/bin/python
session=plain-mmmu900-resume
case "${1:-}" in
  check)
    exec "$python" -u tools/resume_plain_mmmu900.py --check
    ;;
  resume)
    extra=''
    if [[ $# -eq 2 && "${2:-}" == '--repair-tail' ]]; then extra=' --repair-tail';
    elif [[ $# -ne 1 ]]; then echo 'Unknown argument' >&2; exit 2; fi
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Session already exists: $session; no duplicate started." >&2; exit 1
    fi
    mkdir -p runs/plain_mmmu900_20261102/resume_launches
    log=$(mktemp "$PWD/runs/plain_mmmu900_20261102/resume_launches/launch.XXXXXXXX.log")
    tmux new-session -d -s "$session" "cd '$PWD' && PYTHONDONTWRITEBYTECODE=1 '$python' -u tools/resume_plain_mmmu900.py --resume$extra > '$log' 2>&1; code=\$?; printf '%s\\n' \"\$code\" > '$log.exit'; exit \"\$code\""
    echo "tmux: $session"
    echo "Launch log: $log (successful tmux creation is not proof that evaluation started)"
    ;;
  *) echo 'Usage: bash tools/plain_eval.sh check | resume [--repair-tail]' >&2; exit 2 ;;
esac
