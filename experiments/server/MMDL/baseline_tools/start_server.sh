#!/usr/bin/env bash
# Run inside your server's tmux pane, AFTER stopping the previous vLLM with Ctrl+C.
# Optional explicit context change: MAX_MODEL_LEN=32768 bash baseline_tools/start_server.sh
set -Eeuo pipefail
TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$TOOLS_DIR")"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
DATA_PATH="${DATA_PATH:-$HOME/mmdl-data/mmmu_val}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-VL-4B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9048}"
mkdir -p logs
# Do not kill any processes. Refuse to collide with an existing server.
python - <<'PY'
import socket
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1', 8000)) == 0:
        raise SystemExit('Port 8000 is occupied. Stop the existing vLLM in its own terminal with Ctrl+C first.')
PY
python "$TOOLS_DIR/baseline_eval.py" server-record \
  --data-path "$DATA_PATH" \
  --model-path "$MODEL_PATH" \
  --max-model-len "$MAX_MODEL_LEN" \
  --out logs/baseline_server.json
IMAGE_LIMIT="$(python -c 'import json; print(json.load(open("logs/baseline_server.json"))["settings"]["limit_mm_per_prompt"]["image"])')"
STAMP="$(date +%Y%m%d_%H%M%S)"
cp logs/baseline_server.json "logs/baseline_server_${STAMP}.json"
python - <<'PY' > "logs/packages_${STAMP}.txt"
from importlib.metadata import distributions
for name, version in sorted((d.metadata['Name'], d.version) for d in distributions() if d.metadata.get('Name')):
    print(f'{name}=={version}')
PY
printf '\nStarting real weights: image limit=%s, context=%s, engine seed=3407\n' "$IMAGE_LIMIT" "$MAX_MODEL_LEN"
vllm serve "$MODEL_PATH" \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt "{\"image\":${IMAGE_LIMIT},\"video\":0}" \
  --mm-processor-kwargs '{"min_pixels":1003520,"max_pixels":4014080}' \
  --mm-processor-cache-gb 0 \
  --safetensors-load-strategy eager \
  --enforce-eager \
  --seed 3407 \
  --generation-config vllm \
  2>&1 | tee "logs/baseline_server_${STAMP}.log"
