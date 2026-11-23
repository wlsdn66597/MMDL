#!/bin/bash
# MMMU 평가를 차례로 실행한다. 인자: "tag:seed:출력상한[:모델폴더]" 여러 개. 모델 폴더가 없으면 베이스 모델.
# 이미 채점까지 끝난 tag는 요약만 다시 출력한다.
# 예: nohup bash queue_eval.sh "r2_seed1_max8192:1:8192:$HOME/mmdl/train_runs/dpo_r2_loop_b01_lr1e5/merged" > ~/mmdl/train_runs/queue_eval.log 2>&1 < /dev/null &
set -u
EVAL_PY=$HOME/mmdl/mmdl_test/.venv/bin/python
EVAL_DIR=$HOME/mmdl/mmdl_test/outputs/diag
STATUS=${STATUS:-$HOME/mmdl/train_runs/queue_eval.status}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
wait_gpu() { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done; }

# MMMU 평가: evaluate <tag> <seed> [model_path]
evaluate() {
  local TAG=$1 SEED=$2 MAXTOK=$3 MODEL=${4:-}
  cd "$HOME/mmdl/mmdl_test" || return 1
  if [ ! -s "$EVAL_DIR/$TAG.graded.jsonl" ]; then
    wait_gpu
    stamp "$TAG: 생성 시작 (seed $SEED, 출력 상한 $MAXTOK)"
    $EVAL_PY diag/run_generate.py --baseline outputs/baseline_results.json --out "$EVAL_DIR/$TAG.jsonl" --seed "$SEED" \
      --max-tokens "$MAXTOK" --max-num-seqs 16 ${MODEL:+--model-path "$MODEL"} --tag "$TAG" > "$EVAL_DIR/$TAG.gen.log" 2>&1 || return 1
    wait_gpu
    stamp "$TAG: 채점 보조 시작"
    $EVAL_PY diag/judge.py --baseline outputs/baseline_results.json --pred "$EVAL_DIR/$TAG.jsonl" \
      --out "$EVAL_DIR/$TAG.judge.jsonl" --gpu-mem 0.92 > "$EVAL_DIR/$TAG.judge.log" 2>&1 || return 1
    $EVAL_PY diag/grade.py --baseline outputs/baseline_results.json --pred "$EVAL_DIR/$TAG.jsonl" \
      --judge "$EVAL_DIR/$TAG.judge.jsonl" --out "$EVAL_DIR/$TAG.graded.jsonl" > "$EVAL_DIR/$TAG.grade.log" 2>&1 || return 1
  fi
  $EVAL_PY - "$EVAL_DIR/$TAG" "$MAXTOK" <<'PY' | tee "$EVAL_DIR/$TAG.summary.txt"
import sys, json, ast, statistics
sys.path.insert(0, "diag")
from common import load_baseline
p, cap = sys.argv[1], int(sys.argv[2])
_, R = load_baseline("outputs/baseline_results.json")
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
g = {json.loads(l)["id"]: json.loads(l)["verdict"] == "correct" for l in open(p + ".graded.jsonl")}
gen = {json.loads(l)["id"]: json.loads(l)["model_response"] for l in open(p + ".jsonl")}
bt = {k: (lit(r["model_response"]).get("output_tokens") or 0) for k, r in R.items()}
# 베이스(seed 3407, 32768 실행)에 같은 출력 상한을 적용한 값: 상한을 넘긴 문항은 오답으로 본다
bo = {k: bool(lit(r["grading"])["final"]["correct"]) and bt[k] <= cap for k, r in R.items()}
cut = {k for k, m in gen.items() if m["output_tokens"] >= cap - 1}
bcut = {k for k in bt if bt[k] > cap or bt[k] >= 32767}
print(f"MMMU 정확도 {sum(g.values())}/{len(g)} = {100*sum(g.values())/len(g):.1f}% (베이스 seed 3407에 같은 상한 적용: {sum(bo.values())})")
print(f"베이스만 맞힘 {sum(bo[k] and not g[k] for k in g)}, 이 실행만 맞힘 {sum(g[k] and not bo[k] for k in g)}")
print(f"상한({cap})에서 잘림 {len(cut)} (베이스 {len(bcut)}), 둘 다 잘림 {len(cut & bcut)}")
print(f"잘린 문항 중 정답 {sum(g[k] for k in cut)}")
print(f"출력 토큰 중앙값 {statistics.median(m['output_tokens'] for m in gen.values())} (베이스 277)")
PY
}


for spec in "$@"; do
  IFS=: read -r TAG SEED CAP MODEL <<< "$spec"
  echo "RUNNING $TAG" > "$STATUS"
  evaluate "$TAG" "$SEED" "$CAP" "$MODEL" && stamp "$TAG 끝" || stamp "$TAG 실패"
done
echo "DONE" > "$STATUS"
stamp "모두 끝"
