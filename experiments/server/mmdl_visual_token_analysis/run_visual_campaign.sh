#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HOME=/home/rlatjswo/mmdl-cache/huggingface
export MMMU_MODEL_CACHE="$HF_HOME/hub"
export MMMU_DATASET_CACHE="$HF_HOME/datasets"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset VLLM_BATCH_INVARIANT HF_HUB_OFFLINE HF_DATASETS_OFFLINE
cd /home/rlatjswo/mmdl_visual_token_analysis
exec /home/rlatjswo/mmdl_test/.venv/bin/python src/run_visual_campaign.py \
  --campaign /home/rlatjswo/mmmu_visual_tokens_seed42_20261010 \
  --baseline-root /home/rlatjswo/mmmu_baseline_20261009 \
  >> /home/rlatjswo/mmmu_visual_tokens_seed42_20261010/driver.log 2>&1
