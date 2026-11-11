#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_BATCH_INVARIANT
trap 'echo "$? at line $LINENO" > campaign.failed' ERR
until [[ -f baseline.ok ]]; do
    [[ ! -f baseline.failed ]] || { echo "baseline failed" > campaign.failed; exit 1; }
    sleep 5
done
for arm in standard checked; do
    echo "training $arm"
    "$HOME/era_decoder_lora_20261030/.venv/bin/python" -u train_pair.py "$arm" > "$arm.train.log" 2>&1
    echo "merging $arm"
    "$HOME/era_decoder_lora_20261030/.venv/bin/python" -u train_pair.py "$arm" --merge > "$arm.merge.log" 2>&1
done
for arm in standard checked; do
    echo "evaluating $arm"
    "$HOME/mmdl_test/.venv/bin/python" -u eval.py "$arm" > "$arm.eval.log" 2>&1
done
touch complete
