#!/bin/bash
# 차례로 실행한다. 이미 끝난 단계는 건너뛴다 (같은 명령으로 다시 실행 가능).
#   1. 루프 쌍 181개(dpo_r3_loop_best)로 선호 학습 → merge
#   2. MMMU 평가 두 번: 출력 상한 8192, seed 3407과 seed 1
#   3. 새 문항(bank_r2) 샘플링: 베이스 모델, 문항당 8개, 출력 상한 8192. 엔진이 죽으면 이어서 다시 시작한다.
# 실행 (서버): cd ~/mmdl/era_pilot && nohup bash queue_r3.sh > ~/mmdl/train_runs/queue_r3.log 2>&1 < /dev/null &
set -u
RUN=${RUN:-dpo_r3_loop_best_b01_lr1e5}
DATA=${DATA:-$HOME/mmdl/datasets/dpo_r3_loop_best}
OUT=$HOME/mmdl/train_runs/$RUN
BASE=$HOME/mmdl/mmdl_test/outputs/baseline_results.json
DIAG=$HOME/mmdl/mmdl_test/diag
TRAIN_PY=$HOME/mmdl/.venv-train/bin/python
EVAL_PY=$HOME/mmdl/mmdl_test/.venv/bin/python
SAMPLE_OUT=$HOME/mmdl/train_runs/bank_sampling_r2_n8
STATUS=$HOME/mmdl/train_runs/queue_r3.status
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
wait_gpu() { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done; }
cd "$HOME/mmdl/era_pilot" || exit 1

echo "TRAIN" > "$STATUS"
if [ ! -f "$OUT/adapter/adapter_config.json" ]; then
  [ -e "$OUT" ] && { stamp "실패: 끝나지 않은 출력 폴더가 있음 $OUT"; echo "FAILED: 학습 폴더" > "$STATUS"; exit 1; }
  wait_gpu
  stamp "1. 선호 학습 시작: $RUN"
  $TRAIN_PY train_dpo.py --data "$DATA/train.jsonl" --val "$DATA/val.jsonl" --baseline "$BASE" --diag "$DIAG" --out "$OUT" \
    --beta 0.1 --w-gen 1.0 --lr 1e-5 --epochs 4 --grad-accum 16 --save-every 10 > "$OUT.log" 2>&1 \
    || { stamp "실패: 학습"; echo "FAILED: 학습" > "$STATUS"; }
fi
if [ -f "$OUT/adapter/adapter_config.json" ]; then
  if [ ! -f "$OUT/merged/config.json" ]; then
    stamp "1. merge"
    $TRAIN_PY train_dpo.py --merge-only --adapter-name adapter --data x --baseline "$BASE" --diag "$DIAG" --out "$OUT" >> "$OUT.log" 2>&1
  fi
  tail -1 "$OUT/val_log.jsonl" | cut -c1-200
  echo "EVAL" > "$STATUS"
  stamp "2. MMMU 평가 (출력 상한 8192, seed 3407과 1)"
  STATUS=$HOME/mmdl/train_runs/queue_r3_eval.status bash queue_eval.sh \
    "${RUN}_seed3407_max8192:3407:8192:$OUT/merged" "${RUN}_seed1_max8192:1:8192:$OUT/merged"
fi

echo "SAMPLING" > "$STATUS"
mkdir -p "$SAMPLE_OUT"
for try in 1 2 3 4 5 6 7 8; do
  [ -f "$SAMPLE_OUT/samples.summary.json" ] && break
  wait_gpu
  stamp "3. 새 문항 샘플링 (시도 $try)"
  $EVAL_PY sample_bank.py --data "$HOME/mmdl/datasets/bank_r2/bank_r2.jsonl" --images / --baseline "$BASE" --diag "$DIAG" \
    --out "$SAMPLE_OUT/samples.jsonl" --chunk 50 --gpu-mem 0.88 --max-num-seqs 64 >> "$SAMPLE_OUT/run.log" 2>&1
  stamp "샘플링 프로세스 종료 (저장된 문항 $(cut -d'"' -f4 "$SAMPLE_OUT/samples.jsonl" 2>/dev/null | sort -u | wc -l))"
  pkill -f "[s]ample_bank.py"; sleep 20
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
done
[ -f "$SAMPLE_OUT/samples.summary.json" ] && echo "DONE" > "$STATUS" || echo "FAILED: 샘플링 미완료" > "$STATUS"
stamp "모두 끝"
