"""Audit the first explicitly named and valued triple. Does not call a model."""
import hashlib
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from semantic_check import terminal

NUMBER = r"[-+]?\d+(?:\.\d+)?"
FIELD = rf"([A-Za-z][A-Za-z-]*)\s+(?:at\s+)?({NUMBER})"
PREAMBLE = r"^\s*Evidence:\s*((?:(?:Check|Direct(?: read)?|Recheck|Verify)\.\s+)*)"
FIRST = re.compile(PREAMBLE + r"(?:the three (?:shortest|smallest|lowest) (?:bars|values)|the selected bars|my initial candidates) are\s+"
                   + FIELD + r",\s*" + FIELD + r",?\s+and\s+" + FIELD
                   + r"(?=\.(?:\s|$)|[?!](?:\s|$)|\s*$)", re.I)


def extract(raw):
    # A chart, gold values and later sentences are deliberately unavailable here.
    match = FIRST.match(raw)
    if not match:
        return None
    fields = match.groups()[1:]
    return [{"name": n, "value": v} for n, v in zip(fields[::2], fields[1::2])]


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def jsonlines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def review(arm, predictions, gold, old):
    counts = Counter()
    items = {}
    for pred in predictions:
        rid = pred["id"]
        it = gold[rid]
        if it["mode"] != "rank":
            continue
        assert rid not in items and pred["image_sha256"] == it["image_sha256"]
        draft = extract(pred["raw"])
        correct = pred["finish_reason"] == "stop" and terminal(pred["raw"], it["options"]) == it["letter"]
        old_ok = old[rid] is not None
        counts["total"] += 1
        counts["old"] += old_ok
        counts["double_gate"] += bool(re.match(r"^\s*Evidence:\s*Check\.\s+Direct\.", pred["raw"], re.I))
        counts["final_right"] += correct
        item = dict(id=rid, old_extracted=old_ok, draft=draft, final_right=correct)
        if draft:
            counts["parsed"] += 1
            counts["overlap"] += old_ok
            counts["new_only"] += not old_ok
            chart = {k.lower(): Decimal(v) for k, v in it["rows"].items()}
            names = [x["name"].lower() for x in draft]
            vals = [Decimal(x["value"]) for x in draft]
            aligned = all(n in chart and chart[n] == v for n, v in zip(names, vals))
            right = aligned and len(set(names)) == 3 and sorted(vals) == sorted(chart.values())[:3]
            counts["row_value_matches"] += aligned
            counts["right"] += right
            counts["wrong"] += not right
            counts[("right_" if right else "wrong_") + ("right" if correct else "wrong")] += 1
            item.update(source_pairs_match=aligned, initial_right=right)
        items[rid] = item
    assert counts["total"] == len(items) == 180
    return counts, items


def main():
    gold = {x["id"]: x for x in read(ROOT/"holdout/gold.json")}
    inputs = [ROOT/"holdout/gold.json", ROOT/"semantic_check.py", Path(__file__).resolve()]
    stats, results = {}, {}
    for arm in "AB":
        p, q = ROOT/"eval"/arm/"natural/predictions.jsonl", ROOT/"eval"/arm/"self/coverage.json"
        assert read(p.parent/"done.json")["output_sha256"] == sha(p)
        old = {x["id"]: x["draft"] for x in read(q)}
        stats[arm], results[arm] = review(arm, jsonlines(p), gold, old)
        inputs.extend([p, q])
    common = sorted(rid for rid in results["A"] if all(results[a][rid]["draft"] is not None for a in "AB"))
    replay = [{"id": rid, **{a: results[a][rid]["draft"] for a in "AB"}} for rid in common]
    save(OUT/"common_replay.json", replay)
    save(OUT/"audit_results.json", dict(counts=stats, common_n=len(common),
         source_hashes={str(p.relative_to(ROOT)): sha(p) for p in inputs}, scores=results))
    a, b = stats["A"], stats["B"]
    table = [["기존 재검토 추출기에서 초안 사용 가능", a["old"], b["old"]],
             ["이번 이름·값 규칙으로 추출", a["parsed"], b["parsed"]],
             ["이전 규칙과 공통 / 이번 규칙에서만 추출", f'{a["overlap"]} / {a["new_only"]}', f'{b["overlap"]} / {b["new_only"]}'],
             ["추출한 세 항목의 행 이름과 값이 모두 차트에 존재", a["row_value_matches"], b["row_value_matches"]],
             ["그중 최초 세 항목이 실제 최솟값 집합과 일치", a["right"], b["right"]],
             ["처음 맞음 → 최종 정답 / 최종 오답", f'{a["right_right"]} / {a["right_wrong"]}', f'{b["right_right"]} / {b["right_wrong"]}'],
             ["처음 틀림 → 최종 정답 / 최종 오답", f'{a["wrong_right"]} / {a["wrong_wrong"]}', f'{b["wrong_right"]} / {b["wrong_wrong"]}'],
             ["전체에서 E의 시작이 Check. Direct.", a["double_gate"], b["double_gate"]]]
    rows = "\n".join(f"| {label} | {x} | {y} |" for label,x,y in table)
    report = f"""# ERA 1k 기존 응답의 최초 선택 판독

A·B의 무지시 순위 문항 180개씩을 로컬 CPU에서 점검했다. 첫 E에 **이름과 값이 명시된 세 항목**이 있으면 그 문장만으로 복사한다. 양쪽에서 이 기준을 충족하는 **{len(common)}개 문항**을 다음 재검토 실험에 사용할 수 있다. 기존 실행의 점수와 응답은 그대로 보존했다.

| 판독 및 전이 | A | B |
|---|---:|---:|
{rows}

기존 수치는 당시 실제 두 번째 호출에 썼던 숫자 초안 추출기의 범위다. 이번 규칙은 첫 선언에서 명시적으로 연결된 이름과 값 세 쌍을 읽는다. 두 규칙의 범위는 일부만 겹친다. `Check.`와 `Direct.`가 연이어 나오면 둘 다 넘긴 뒤 첫 선언을 읽는다. 행 이름은 영문과 하이픈으로 한정하고 `Heath at 4.9`나 `Heath 4.9` 형태를 허용한다. 전체 표 나열, 모호한 대안, 이후 정정·검산 문장은 추출에 사용하지 않는다. 추출을 완료한 뒤에만 원본 표와 대조해, 존재하는 행·값인지와 실제 최소 세 행인지를 따로 판정했다.

‘처음 틀림 → 최종 정답’은 초기 선택과 최종 보기의 전이다. 보기만 보고 바꿨는지, 실제 누락 행과 비교해 수정했는지는 해당 응답의 E·R·A 검수가 필요하다. 순위 문제나 기존 보고서의 240문항 전체에 추출 가능한 응답의 성공률을 그대로 적용하면 안 된다.

**X007_rreplace.** A와 B는 모두 Grove 9.2, Arbor 15.4, Crest 54.6을 처음 골랐다. 실제 선택은 마지막에 Heath 54.3이 들어가야 한다. A는 제외된 Heath가 54.3이라고 다시 읽고도 유지한다고 썼고, B도 54.6을 유지했다. 두 모델 모두 79.2를 답했다. 정답은 78.9다.

[공통 문항과 모델별 원문 초안](common_replay.json) · [검수 결과와 입력 해시](audit_results.json) · [재현 스크립트](audit.py)
"""
    (OUT/"report.md").write_text(report, encoding="utf8")
    print(json.dumps(dict(counts=stats, common_n=len(common)), indent=2))


if __name__ == "__main__":
    main()
