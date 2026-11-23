#!/usr/bin/env bash
# [ablation (b)] E(seed 3407)와 모든 것이 같고 시각 요소만 뺀다: connector LoRA(--vision-lora), DeepStack gate(--gates), 이미지 조건 손실(--img-pref-types).
# 데이터는 pairs_E.jsonl 원본 그대로. 결과: runs/dpo2_noVis/
set -euo pipefail
source /shared_dir/MMDL_solo/repro/env.sh
export HF_HUB_OFFLINE=1
SEED=3407; W=$ROOT/runs/dpo2_noVis; mkdir -p $W
log() { echo "[$(date '+%F %T')] $*" | tee -a "$W/pipeline.log"; }
if [ ! -f $W/.train.done ]; then
  LAM=$(python3 -c "import json;print(json.load(open('$RUN/hparams.json'))['lambda'])")
  log "[noVis] 학습 시작 (E와 동일하되 시각 요소 제거: LLM LoRA만, 게이트·이미지 손실 없음, lr 1e-5, β 0.1, λ $LAM, replay 1.0, seed $SEED)"
  rm -rf $W/train
  $TRAIN_PY -u $ROOT/scripts/train_dpo_d.py --pairs "$D/pairs_E.jsonl" --images $IMAGES --start-model $BASE \
    --vision-config $VCFG --diag $ROOT/scripts --out "$W/train" --beta 0.1 --lr 1e-5 --epochs 1 \
    --grad-accum 16 --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --max-tokens 8400 --seed $SEED --workers 2 \
    --rpo-alpha "$LAM" --replay-weight 1.0 > "$W/train.log" 2>&1
  [ -f "$W/train/merged/config.json" ] || { log "[noVis] merged 없음 → train.log 확인"; exit 5; }
  log "[noVis] 학습 완료: $(tail -1 $W/train/train_log.jsonl | cut -c1-250)"
  touch $W/.train.done
fi
if [ ! -f $W/.eval.done ]; then
  log "[noVis] MMMU-val 평가 시작"
  rm -rf "$EVAL/outputs/dpo2_noVis"
  (cd "$EVAL" && $EVAL_PY src/evaluate.py --config src/config.json --checkpoint "$W/train/merged" --run-name dpo2_noVis) > "$W/eval.log" 2>&1
  rm -rf $W/eval; cp -r "$EVAL/outputs/dpo2_noVis" $W/eval
  $EVAL_PY $ROOT/scripts/summarize_eval.py --eval-dir $W/eval --out $W/summary_vs_baseline.md --label "DPO2-E - visual components (seed 3407)" >> $W/eval.log 2>&1 || true
  python3 $ROOT/scripts/eval_extra.py --eval-dir $W/eval --out $W/extra_stats.json >> $W/eval.log 2>&1 || true
  touch $W/.eval.done
  log "[noVis] 평가 완료: $(sed -n 3p $W/summary_vs_baseline.md 2>/dev/null)"
fi
