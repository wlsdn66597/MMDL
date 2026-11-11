#!/usr/bin/env bash
set -uo pipefail
cd /home/rlatjswo/era_decoder_lora_20261030
mkdir -p runs/plain_mmmu900_20261102
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_BATCH_INVARIANT
/home/rlatjswo/mmdl_test/.venv/bin/python -u tools/plain_mmmu900_campaign_20261102.py --execute > runs/plain_mmmu900_20261102/campaign.log 2>&1
code=$?
printf '%s\n' "$code" > runs/plain_mmmu900_20261102/campaign.exit.tmp
mv runs/plain_mmmu900_20261102/campaign.exit.tmp runs/plain_mmmu900_20261102/campaign.exit
exit "$code"
