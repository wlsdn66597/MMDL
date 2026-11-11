#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HOME=/home/rlatjswo/mmdl-cache/huggingface
export MMMU_MODEL_CACHE="$HF_HOME/hub"
export MMMU_DATASET_CACHE="$HF_HOME/datasets"
export CUDA_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset VLLM_BATCH_INVARIANT HF_HUB_OFFLINE HF_DATASETS_OFFLINE
root=/home/rlatjswo/mmmu_multiseed_20261011
exec /home/rlatjswo/mmdl_test/.venv/bin/python "$root/resume_mmmu_multiseed.py" --root "$root" >> "$root/driver.log" 2>&1
