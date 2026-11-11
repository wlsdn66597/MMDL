#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec >>logs/controller.log 2>&1
exec 9>.campaign.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_BATCH_INVARIANT
sha256sum --quiet -c release.sha256
echo "$(date -Is) tmux=$TMUX controller start"
exec "$HOME/mmdl_test/.venv/bin/python" -u campaign.py
