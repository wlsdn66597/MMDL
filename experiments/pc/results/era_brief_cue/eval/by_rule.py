"""Describe per-rule transitions and inspect new losses after the cue pilot is sealed."""
import json
from pathlib import Path
from collections import defaultdict
from cue_report import ARMS, LABELS, cell, md
from score_cue import sha, rubric

ROOT=Path(__file__).resolve().parents[1]
NAMES={"change":"순변위·총거리","mixture":"동량·전량 혼합","rate":"건수·비율","route":"직행·경유"}


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s]


def build(root):
    review=root/"eval/review"
    summary=json.loads((review/"summary.json").read_text(encoding="utf8"))
    assert sha(review/"scored.jsonl")==summary["proof"]["scored.jsonl"]
    assert sha(review/"manual.jsonl")==summary["proof"]["manual.jsonl"]
    rows=lines(review/"scored.jsonl")
    notes={n["key"]:n["reason"] for n in lines(review/"manual.jsonl")}
    by_arm=defaultdict(dict)
    kind={}
    for row in rows:
        arm,rid=row["arm"],row["id"]
        assert rid not in by_arm[arm]
        by_arm[arm][rid]=row
        if row["dataset"]=="paired":
            assert kind.setdefault(row["group"],row["kind"])==row["kind"]
    assert len(kind)==48 and len(rows)==1152 and all(len(v)==192 for v in by_arm.values())
    content=["# 지시 전후 규칙 유형별 연결","",
             "짝의 두 질문 모두에서 정답과 ERA 연결을 맞힌 그룹 수를 집계했다. 각 규칙 유형은 12그룹이다.","",
             md("유형",*[LABELS[a] for a in ARMS]),"|---|---:|---:|---:|"]
    for k,name in NAMES.items():
        groups=[g for g,t in kind.items() if t==k]
        assert len(groups)==12
        cells=[]
        for arm in ARMS:
            low=summary["groups"][arm+"_before"];high=summary["groups"][arm+"_cue"]
            f=lambda source:{g:source[g]["joint"] for g in groups}
            l,h=f(low),f(high);tr=rubric.transitions(l,h)
            cells.append(f"{cell(rubric.bounds(l.values()))} → {cell(rubric.bounds(h.values()))} (회복 {len(tr['recovered'])}, 손실 {len(tr['lost'])})")
        content.append(md(name,*cells))
    content+=["","## 새로 틀린 문항","","각 모델이 원래 맞혔고 지시 추가 후 실제 제출한 답이 틀린 문항이다. 일반 문항의 단계 판정은 실패한 필드들을 표시한다. 규칙 문항의 최초 오류는 익명 원문 검수값이다.",""]
    for arm in ARMS:
        losses=summary["changes"][arm]["all"]["answer"]["lost"]
        content+=["### "+LABELS[arm],"",f"신규 오답 {len(losses)}개.",""]
        for rid in losses:
            x=by_arm[arm+"_cue"][rid];prev=by_arm[arm+"_before"][rid]
            fields=[f for f in ("e","r","intermediate","winner","a","consistent") if x[f] is False]
            desc=notes.get(x["key"],"기존 확정 판정 재사용 또는 보기 단독 응답.")
            content.append(f"- `{rid}` {prev['choice']} → {x['choice']}. 최초 오류 {x['first_error'] or '일반 문항 별도 분류 없음'}. 실패 필드 {', '.join(fields) or '없음'}. {desc}")
        content.append("")
    return "\n".join(content)+"\n"


if __name__=="__main__":
    (ROOT/"case_stats.md").write_text(build(ROOT),encoding="utf8")
