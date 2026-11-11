#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset VLLM_BATCH_INVARIANT
for arm in generic boundary; do
    mkdir -p "$arm"
    if ~/mmdl_test/.venv/bin/python -u run_arm.py "$arm" > "$arm/campaign.log" 2>&1; then
        touch "$arm/succeeded"
    else
        status=$?
        echo "$status" > "$arm/failed"
        exit "$status"
    fi
done
touch succeeded
