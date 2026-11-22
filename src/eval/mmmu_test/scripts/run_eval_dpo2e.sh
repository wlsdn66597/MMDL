#!/bin/bash
# 평가를 차례로 실행한다: 생성(4B) → 채점 보조(Qwen3-8B, 형식이 어긋난 응답만) → 채점 → 요약.
#
# 인자: "tag:seed:출력상한:모델폴더:문항파일:루프중단" 여러 개
#   모델폴더   비우면 베이스 모델(Qwen3-VL-4B-Instruct). 326쌍 모델은 merge_adapter.py로 만든 폴더
#   문항파일   build_records.py로 만든 파일
#   루프중단   1이면 같은 말을 반복하는 응답을 중간에 끊는다(오답 처리). 베이스는 32,768까지 반복하는 응답이
#              많아 시간이 오래 걸리므로 1을 권한다. 점수는 끝까지 생성할 때와 같다(val 루프 81개 모두 검출, 오검출 0).
#
# 예 (패키지 폴더에서):
#   PY=.venv/bin/python bash scripts/run_eval.sh \
#     "r4_val:3407:32768:merged:records/mmmu_val.json:0" \
#     "base_val:3407:32768::records/mmmu_val.json:1"
#
# 결과: outputs/<tag>.jsonl(응답), .judge.jsonl, .graded.jsonl(문항별 정오), .summary.txt(점수)
# 중간에 끊겨도 같은 tag로 다시 실행하면 이어서 한다. 채점까지 끝난 tag는 다시 돌리지 않는다.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}
OUT=outputs
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
nlines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

evaluate() {
  local TAG=$1 SEED=$2 MAXTOK=$3 MODEL=$4 REC=$5 LOOP=$6
  [ -f "$REC" ] || { stamp "$TAG: 문항 파일 없음 $REC"; return 1; }
  [ -z "$MODEL" ] || [ -d "$MODEL" ] || { stamp "$TAG: 모델 폴더 없음 $MODEL"; return 1; }
  local TOTAL; TOTAL=$($PY -c "import json;print(len(json.load(open('$REC',encoding='utf-8'))['records']))") || return 1
  if [ ! -s "$OUT/$TAG.graded.jsonl" ]; then
    for try in 1 2 3 4 5 6; do
      [ "$(nlines "$OUT/$TAG.jsonl")" -ge "$TOTAL" ] && break
      stamp "$TAG: 생성 시작 (시도 $try, seed $SEED, 출력 상한 $MAXTOK, 루프중단 $LOOP, 모델 ${MODEL:-베이스}, 저장된 문항 $(nlines "$OUT/$TAG.jsonl")/$TOTAL)"
      $PY scripts/diag/run_generate.py --baseline "$REC" --out "$OUT/$TAG.jsonl" --seed "$SEED" \
        --max-tokens "$MAXTOK" --max-num-seqs "${SEQS:-16}" ${MODEL:+--model-path "$MODEL"} $([ "$LOOP" = 1 ] && echo --loop-stop) \
        --tag "$TAG" >> "$OUT/$TAG.gen.log" 2>&1
      stamp "$TAG: 생성 종료 (exit $?, 저장된 문항 $(nlines "$OUT/$TAG.jsonl")/$TOTAL)"
      sleep 10
    done
    [ "$(nlines "$OUT/$TAG.jsonl")" -ge "$TOTAL" ] || { stamp "$TAG: 생성 미완료, $OUT/$TAG.gen.log 확인"; return 1; }
    stamp "$TAG: 채점 보조 시작"
    $PY scripts/diag/judge.py --baseline "$REC" --pred "$OUT/$TAG.jsonl" \
      --out "$OUT/$TAG.judge.jsonl" --gpu-mem 0.92 > "$OUT/$TAG.judge.log" 2>&1 || { stamp "$TAG: 채점 보조 실패"; return 1; }
    $PY scripts/diag/grade.py --baseline "$REC" --pred "$OUT/$TAG.jsonl" \
      --judge "$OUT/$TAG.judge.jsonl" --out "$OUT/$TAG.graded.jsonl" > "$OUT/$TAG.grade.log" 2>&1 || return 1
  fi
  $PY - "$OUT/$TAG" "$REC" <<'PY' | tee "$OUT/$TAG.summary.txt"
import sys, json, statistics, collections
p, rec = sys.argv[1], sys.argv[2]
R = {r["id"]: r for r in json.load(open(rec, encoding="utf-8"))["records"]}
G = [json.loads(l) for l in open(p + ".graded.jsonl", encoding="utf-8")]
n = len(G); ok = sum(g["verdict"] == "correct" for g in G)
print(f"{rec}: 정확도 {ok}/{n} = {100*ok/n:.2f}%")
tok = [g["output_tokens"] for g in G]
print(f"생성 상태 {dict(collections.Counter(g.get('generation_status') for g in G))}, "
      f"2048 초과 {sum(t > 2048 for t in tok)}, 출력 토큰 중앙값 {statistics.median(tok)}")
print("판정 경로", dict(collections.Counter(g["reason"] for g in G).most_common(8)))
for key, name in (("source_split", "출처 분할별(MMMU-Pro)"), ("question_type", "유형별"), ("subject", "과목별")):
    by = collections.defaultdict(lambda: [0, 0])
    for g in G:
        k = R[g["id"]].get(key)
        if k is None: continue
        by[k][0] += g["verdict"] == "correct"; by[k][1] += 1
    if by:
        print(name)
        for k in sorted(by): print(f"  {k}: {by[k][0]}/{by[k][1]} = {100*by[k][0]/by[k][1]:.1f}%")
PY
}

rc=0
for spec in "$@"; do
  IFS=: read -r TAG SEED MAXTOK MODEL REC LOOP <<< "$spec"
  evaluate "$TAG" "$SEED" "$MAXTOK" "$MODEL" "$REC" "${LOOP:-0}" && stamp "$TAG 끝" || { stamp "$TAG 실패"; rc=1; }
done
stamp "모두 끝"
exit $rc
