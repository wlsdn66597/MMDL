#!/bin/bash
cd /home/rlatjswo/era_decoder_lora_20261030 || exit 1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/rlatjswo/mmdl_test/.venv/bin/python -u tools/regrade_epoch_pending.py --execute > artifacts/era_reasoning_judge.log 2>&1
status=$?
printf '%s\n' "$status" > runs/era_4epoch_20261030/reasoning_judge.exit
exit "$status"
