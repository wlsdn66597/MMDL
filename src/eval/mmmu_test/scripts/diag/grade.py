"""생성 결과 채점.

사용 예:
  # 1) 채점기 자체 점검: 베이스라인 응답을 다시 채점해 공식 채점과 비교 (GPU 불필요)
  python grade.py --baseline baseline_results.json --self-check

  # 2) 새 실행 결과 채점
  python grade.py --baseline baseline_results.json --pred runs/seed1.jsonl \
                  --out runs/seed1.graded.jsonl [--judge runs/seed1.judge.jsonl]

needs_judge로 남은 항목은 judge.py로 판정한 뒤 --judge로 넘기면 반영된다.
"""
import argparse
import collections
import json

from common import grade_prediction, load_baseline, read_jsonl


def self_check(records):
    agree = disagree = 0
    needs = collections.Counter()
    by_method = collections.Counter()
    mismatches = []
    for rid, rec in records.items():
        pred = {"model_response": rec["model_response"]}
        g = grade_prediction(pred, rec)
        official = rec["grading"]["final"]["correct"]
        method = rec["grading"]["final"]["method"]
        if g["verdict"] == "needs_judge":
            needs[method] += 1
            continue
        mine = g["verdict"] == "correct"
        if mine == official:
            agree += 1
            by_method[method] += 1
        else:
            disagree += 1
            mismatches.append((rid, method, rec["grading"]["final"].get("extracted_answer"), g["answer"], g["reason"]))
    print(f"판정 가능 {agree + disagree}개 중 공식 채점과 일치 {agree}, 불일치 {disagree}")
    print("일치 항목의 공식 채점 경로:", dict(by_method))
    print("needs_judge로 남은 항목 (공식 채점 경로별):", dict(needs), "합계", sum(needs.values()))
    for m in mismatches[:30]:
        print("  불일치:", m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--pred")
    ap.add_argument("--out")
    ap.add_argument("--judge", help="judge.py 출력 jsonl (선택)")
    a = ap.parse_args()

    _, records = load_baseline(a.baseline)
    if a.self_check:
        self_check(records)
        return

    judges = {}
    if a.judge:
        for j in read_jsonl(a.judge):
            judges[j["id"]] = j
    preds = read_jsonl(a.pred)
    cnt = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for p in preds:
            rec = records[p["id"]]
            g = grade_prediction(p, rec, judges.get(p["id"]))
            cnt[g["verdict"]] += 1
            row = {"id": p["id"], "seed": p.get("seed"), **g,
                   "output_tokens": p["model_response"].get("output_tokens"),
                   "generation_status": p["model_response"].get("generation_status")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(preds)
    print(f"{a.pred}: {n}개, 정답 {cnt['correct']} ({cnt['correct']/n:.1%}), "
          f"오답 {cnt['incorrect']}, needs_judge {cnt['needs_judge']}")


if __name__ == "__main__":
    main()
