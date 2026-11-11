#!/usr/bin/env bash
set -uo pipefail
cd /home/rlatjswo/era_decoder_lora_20261030
mkdir -p runs/mmmu900_epochs_20261031
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
/home/rlatjswo/mmdl_test/.venv/bin/python -u tools/mmmu900_epoch_campaign.py --execute > runs/mmmu900_epochs_20261031/campaign.log 2>&1
code=$?
printf '%s\n' "$code" > runs/mmmu900_epochs_20261031/campaign.exit.tmp
mv runs/mmmu900_epochs_20261031/campaign.exit.tmp runs/mmmu900_epochs_20261031/campaign.exit
exit "$code"
