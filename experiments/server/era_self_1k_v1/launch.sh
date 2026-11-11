#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec >>logs/pipeline.log 2>&1
trap 's=$?; printf "%s line %s exit %s\n" "$(date -Is)" "$LINENO" "$s" > failed; exit "$s"' ERR
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_BATCH_INVARIANT
exec 9>.pipeline.lock
flock -n 9
if test -f failed; then mv failed "logs/failure-$(date +%s).txt"; fi
echo "$(date -Is) tmux=$TMUX pid=$$ on $(hostname)"
EVAL="$HOME/mmdl_test/.venv/bin/python"
if ! test -f probe/000/done.json; then
    echo "$(date -Is) probe 000 starting"
    "$EVAL" -u infer.py --probe 0 --batch 6 >>logs/probe-000.log 2>&1
fi
echo "$(date -Is) probe 000 ready"
while ! test -f release.ready; do sleep 8; done
"$EVAL" -u pipeline.py
