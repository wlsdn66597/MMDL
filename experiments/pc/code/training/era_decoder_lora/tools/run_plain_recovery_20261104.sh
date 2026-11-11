#!/usr/bin/env bash
set -uo pipefail
cd /home/rlatjswo/era_decoder_lora_20261030
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_BATCH_INVARIANT
/home/rlatjswo/mmdl_test/.venv/bin/python -u tools/recover_plain_mmmu900_20261104.py --execute > runs/plain_mmmu900_20261102/recovery_20261104/controller.log 2>&1
code=$?
printf '%s\n' "$code" > runs/plain_mmmu900_20261102/recovery.exit.tmp
mv runs/plain_mmmu900_20261102/recovery.exit.tmp runs/plain_mmmu900_20261102/recovery.exit
exit "$code"
