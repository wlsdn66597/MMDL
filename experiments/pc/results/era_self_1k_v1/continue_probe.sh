#!/usr/bin/env bash
# Preserve useful GPU work during a recovery-test retry; never overlap the main runner.
set -Eeuo pipefail
cd "$(dirname "$0")"
exec >>logs/pipeline.log 2>&1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_BATCH_INVARIANT
exec 9>.pipeline.lock
flock -n 9
P="$HOME/mmdl_test/.venv/bin/python"
echo "$(date -Is) continuing actual rollouts while recovery is being corrected"
if ! test -e probe/001/receipt.json; then "$P" dataprep.py probe --shard 1 >logs/render-001.log; fi
if ! test -e probe/001/done.json; then "$P" -u infer.py --probe 1 >>logs/probe-001.log 2>&1; fi
"$P" dataprep.py inspect >logs/screen-001.json
echo "$(date -Is) second shard complete"
flock -u 9
until test -e retry.ready; do sleep 5; done
exec bash launch.sh
