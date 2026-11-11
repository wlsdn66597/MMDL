#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
exec >>campaign.log 2>&1
trap 's=$?; echo "$s at line $LINENO on $(date -Is)" > failed; exit $s' ERR
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_BATCH_INVARIANT
TRAIN="$HOME/era_decoder_lora_20261030/.venv/bin/python"
EVAL="$HOME/mmdl_test/.venv/bin/python"
echo "$(date -Is) started on $(hostname) tmux=$TMUX"
for arm in control paired; do
    echo "$(date -Is) training $arm"
    "$TRAIN" -u train.py "$arm" > "$arm.train.log" 2>&1
    echo "$(date -Is) merging $arm"
    "$TRAIN" -u train.py "$arm" --merge > "$arm.merge.log" 2>&1
done
for arm in start control paired; do
    echo "$(date -Is) evaluating $arm"
    "$EVAL" -u eval.py "$arm" > "$arm.eval.log" 2>&1
done
"$EVAL" collect.py
echo "$(date -Is) completed"
touch complete
