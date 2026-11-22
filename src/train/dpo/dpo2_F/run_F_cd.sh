#!/usr/bin/env bash
# 실험 F (자기증류 → DPO): E와 같은 DPO 학습 앞에, 모델 자신의 ERA 정답 풀이로 "지시 없이 ERA로 쓰기"를 먼저 SFT한다.
#  1) cd_data : base_era 샘플(ERA 지시를 붙여 생성) 중 ERA 3단계 형식 + 정답 + 정상 종료인 것, 문항당 최대 2개.
#               입력은 지시를 뺀 원래 질문(input_prompt), 정답은 그 풀이 원문 (= context distillation).
#  2) cd_sft  : 베이스에서 LoRA SFT (lr 2e-5, 1 epoch, seed) → 병합 모델
#  3) dpo     : 2)의 병합 모델에서 시작, E와 똑같은 쌍(pairs_E)·설정으로 DPO (기준 모델 = 2)의 병합 모델)
#  4) eval    : MMMU-val (평가 프롬프트·설정 고정) + ERA 형식 비율 집계
# 결과: runs/dpo2_F_cd/ (E 결과 runs/dpo2/ 는 건드리지 않는다). 다시 실행하면 .done 이 있는 단계는 건너뛴다.
# 실행: SEED=3407 nohup bash /shared_dir/MMDL_solo/repro/run_F_cd.sh >> /shared_dir/logs/F_cd.out 2>&1 &
set -euo pipefail
source /shared_dir/MMDL_solo/repro/env.sh
EVAL_SRC=src/eval/mmmu_validation
[[ -f "$EVAL/$EVAL_SRC/evaluate.py" ]] || EVAL_SRC=src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 긴 문항에서 단편화로 인한 OOM 방지
SEED=${SEED:-3407}
F=$ROOT/runs/dpo2_F_cd; X=s$SEED; O=$F/$X
mkdir -p "$F/data" "$O"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$F/pipeline.log"; }
is_done() { [ -f "$F/.$1.done" ]; }; done_mark() { touch "$F/.$1.done"; }
wait_gpu() { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 2000 ]; do log "GPU 사용 중 - 5분 후 확인"; sleep 300; done; }
guard_window() { while true; do w=$(python3 $ROOT/repro/blocked_window.py $ROOT/repro/blocked_times.txt "$1"); [ "$w" = 0 ] && return 0; log "[$2] 차단 시간 → ${w}초 대기"; sleep "$w"; done; }

# 1) 자기증류 데이터
if ! is_done cd_data; then
  python3 - "$SFT_DATA" "$D/samples_base_era.jsonl" "$D/labels_era.jsonl" "$F/data/cd_sft.jsonl" "$SEED" <<'EOF'
import json, re, random, sys, collections
sft, samp, lab, out, seed = sys.argv[1:6]
ERA = [re.compile(rf"^\W*{k}\W*:", re.I | re.M) for k in ("Evidence", "Rule", "Application")]
q = {r["id"]: r for r in map(json.loads, open(sft, encoding="utf-8"))}
txt = {}
for s in map(json.loads, open(samp, encoding="utf-8")):
    for k, x in enumerate(s["samples"]): txt[(s["id"], k)] = x["raw"]
ok = collections.defaultdict(list)
for r in map(json.loads, open(lab, encoding="utf-8")):
    t = txt.get((r["id"], r["k"]), "")
    if r["label"] is True and r["finish_reason"] == "stop" and all(x.search(t) for x in ERA) and "\nFinal answer:" in t.strip():
        ok[r["id"]].append(t.strip())
rng = random.Random(int(seed)); rows = []
for i in sorted(ok):
    if i not in q: continue
    for t in rng.sample(ok[i], min(2, len(ok[i]))):
        b = q[i]; rows.append({"id": i, "vic_type": b["vic_type"], "question_type": b["question_type"], "image": b["image"],
                               "input_prompt": b["input_prompt"], "target": t})
rng.shuffle(rows)
with open(out, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
rep = {"rows": len(rows), "questions": len({r['id'] for r in rows}), "by_vic_type": dict(collections.Counter(r["vic_type"] for r in rows))}
json.dump(rep, open(out.replace(".jsonl", ".report.json"), "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False))
EOF
  log "[F] 자기증류 데이터: $(cat $F/data/cd_sft.report.json | tr -d '\n ')"
  done_mark cd_data
fi

# 1b) 길이 필터: train_sft.py는 긴 문항을 건너뛰지 않고 멈추므로 미리 거른다 (입력 토큰 + 정답 토큰 <= 6144)
if ! is_done cd_filter; then
  cp "$F/data/cd_sft.jsonl" "$F/data/cd_sft_all.jsonl"
  (cd "$F/data" && $TRAIN_PY -I - "$BASE" "$D/samples_base_era.jsonl" <<'EOF'
import json, sys, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
inp = {q["id"]: q["input_tokens"] for q in map(json.loads, open(sys.argv[2]))}
rows = [json.loads(l) for l in open("cd_sft_all.jsonl")]
keep = [r for r in rows if inp[r["id"]] + len(tok(r["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]) + 8 <= 6144]
with open("cd_sft.jsonl", "w") as f:
    for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
rep = {"rows": len(keep), "questions": len({r["id"] for r in keep}), "by_vic_type": dict(collections.Counter(r["vic_type"] for r in keep)),
       "dropped_over_6144_tokens": len(rows) - len(keep), "rows_before_length_filter": len(rows)}
json.dump(rep, open("cd_sft.report.json", "w"), ensure_ascii=False, indent=1); print(rep)
EOF
  )
  log "[F] 길이 필터 후: $(cat $F/data/cd_sft.report.json | tr -d '\n ')"
  done_mark cd_filter
fi

# 2) 자기증류 SFT (베이스에서)
if ! is_done cd_sft_$X; then
  guard_window 90 cd_sft; wait_gpu; log "[F $X] 자기증류 SFT 시작 (lr 2e-5, 1 epoch, max-tokens 6144)"
  rm -rf "$O/cd_sft"
  $TRAIN_PY -u $ROOT/scripts/train_sft.py --data "$F/data/cd_sft.jsonl" --images $IMAGES --baseline $VCFG \
    --diag $ROOT/scripts --out "$O/cd_sft" --epochs 1 --lr 2e-5 --grad-accum 16 --warmup-ratio 0.03 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --weight-decay 0.01 --max-grad-norm 1.0 \
    --max-tokens 6144 --save-every 1000000 --seed $SEED --workers 2 --loss-mode token_mean > "$O/cd_sft.log" 2>&1
  [ -f "$O/cd_sft/merged/config.json" ] || { log "[F $X] SFT 병합 모델 없음 → cd_sft.log 확인"; exit 5; }
  log "[F $X] 자기증류 SFT 완료: $(tail -1 $O/cd_sft/train_log.jsonl | cut -c1-200)"
  done_mark cd_sft_$X
fi

# 3) DPO (E와 같은 쌍·설정, 시작·기준 모델 = 자기증류 모델)
if ! is_done dpo_$X; then
  guard_window 150 dpo; wait_gpu
  LAM=$(python3 -c "import json;print(json.load(open('$RUN/hparams.json'))['lambda'])")
  log "[F $X] DPO 시작 (E와 같은 pairs_E, lr 1e-5, λ $LAM, vision LoRA·게이트·이미지 손실)"
  rm -rf "$O/dpo"
  $TRAIN_PY -u $ROOT/scripts/train_dpo_d.py --pairs "$D/pairs_E.jsonl" --images $IMAGES --start-model "$O/cd_sft/merged" \
    --vision-config $VCFG --diag $ROOT/scripts --out "$O/dpo" --beta 0.1 --lr 1e-5 --epochs 1 \
    --grad-accum 16 --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --max-tokens 8400 --seed $SEED --workers 2 \
    --rpo-alpha "$LAM" --vision-lora --gates --gate-lr 1e-3 --img-pref-types V,I --replay-weight 1.0 > "$O/dpo.log" 2>&1
  [ -f "$O/dpo/merged/config.json" ] || { log "[F $X] DPO 병합 모델 없음 → dpo.log 확인"; exit 5; }
  log "[F $X] DPO 완료: $(tail -1 $O/dpo/train_log.jsonl | cut -c1-200)"
  done_mark dpo_$X
fi

# 4) 평가
if ! is_done eval_$X; then
  guard_window 360 eval; wait_gpu; log "[F $X] MMMU-val 평가 시작"
  rm -rf "$EVAL/outputs/dpo2_F_cd_$X"
  (cd "$EVAL" && "$EVAL_PY" "$EVAL_SRC/evaluate.py" --config "$EVAL_SRC/config.json" --checkpoint "$O/dpo/merged" --run-name dpo2_F_cd_$X) > "$O/eval.log" 2>&1
  rm -rf "$O/eval"; cp -r "$EVAL/outputs/dpo2_F_cd_$X" "$O/eval"
  $EVAL_PY $ROOT/scripts/summarize_eval.py --eval-dir "$O/eval" --out "$O/summary_vs_baseline.md" --label "DPO2-F 자기증류→DPO (seed $SEED)" >> "$O/eval.log" 2>&1 || true
  python3 $ROOT/scripts/eval_extra.py --eval-dir "$O/eval" --out "$O/extra_stats.json" >> "$O/eval.log" 2>&1 || true
  ERAF=$(python3 - "$O/eval/results.json" <<'EOF'
import json, re, sys
ERA = [re.compile(rf"^\W*{k}\W*:", re.I | re.M) for k in ("Evidence", "Rule", "Application")]
R = json.load(open(sys.argv[1]))["records"]
f = [all(x.search((r["model_response"] or {}).get("raw") or "") for x in ERA) for r in R]
c = [bool(r["grading"]["final"].get("correct")) for r in R]
n = sum(f); print(f"ERA 형식 {n}/{len(R)}, 그중 정답 {sum(a and b for a, b in zip(f, c))}")
EOF
)
  done_mark eval_$X
  log "[F $X] 평가 완료: $(sed -n 3p "$O/summary_vs_baseline.md" 2>/dev/null) / $ERAF / $(tr -d '\n ' < "$O/extra_stats.json" 2>/dev/null | cut -c1-200)"
fi
log "===== DPO2-F ($X) 완료 ====="
