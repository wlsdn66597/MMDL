#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
exec >spotcheck.log 2>&1
trap 'echo "$? at line $LINENO" > spotcheck.failed' ERR
until test -f complete; do
    if test -f failed; then echo "Main campaign failed" > spotcheck.failed; exit 1; fi
    sleep 12
done
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0
"$HOME/mmdl_test/.venv/bin/python" -u spotcheck.py
sha256sum spotcheck.json > spotcheck.ok
