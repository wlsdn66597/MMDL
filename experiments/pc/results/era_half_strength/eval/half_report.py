"""Compute diagnostic thresholds on the fixed R9 examples and render Korean reports."""
from collections import defaultdict
from pathlib import Path
import json

from diagnostic import OLD, REVIEW, ROOT, SCORED, SUMMARY, immutable, lines, read, rubric, sha

ARMS=("base","ordinary","ordinary50","enhanced","enhanced50")
PAIRED_FIELDS=("answer","r","er","joint","rounding_joint")
LABELS=dict(base="LoRA 전",ordinary="일반 100%",ordinary50="일반 50%",enhanced="A 100%",enhanced50="A 50%")
COMPARISONS=(("base","ordinary"),("base","ordinary50"),("base","enhanced"),("base","enhanced50"),
             ("ordinary","ordinary50"),("enhanced","enhanced50"),("ordinary50","enhanced50"),
             ("ordinary","enhanced50"),("enhanced","ordinary50"))
GATES=dict(paired_kept="LoRA 전이 맞힌 규칙 90문항 중 75개 이상 유지",
           general_answer="일반 96문항에서 95개 이상 정답",
           connection="독립 48쌍에서 22쌍 이상 양쪽 연결",
           general_connection="일반 96문항에서 ERA 연결 92개 이상",
           sentinel="보존 전용 16개 정답 전체 유지")


def subset(rows,ds):
    return [r for r in rows if (r["sentinel"] if ds=="sentinel" else r["dataset"]==ds)]


def compute(rows,fixed_ids):
    by={arm:[r for r in rows if r["arm"]==arm] for arm in ARMS}
    assert len(rows)==len({(r["arm"],r["id"]) for r in rows})==960 and all(len(v)==192 for v in by.values())
    qids={r["id"] for r in by["base"]}
    assert all({r["id"] for r in v}==qids for v in by.values())
    groups,metrics={},{}
    for arm,v in by.items():
        buckets=defaultdict(list)
        for r in v:
            if r["dataset"]=="paired": buckets[r["group"]].append(r)
        assert len(buckets)==48 and all(len(b)==2 and b[0]["kind"]==b[1]["kind"] for b in buckets.values())
        groups[arm]={g:dict(kind=b[0]["kind"],**{k:rubric.every(r[k] for r in b) for k in PAIRED_FIELDS})
                     for g,b in buckets.items()}
        metrics[arm]={}
        for ds in ("paired","general","sentinel"):
            rs=subset(v,ds)
            assert len(rs)==(16 if ds=="sentinel" else 96)
            metrics[arm][ds]={"n":len(rs), **{k:rubric.bounds(r[k] for r in rs) for k in
                                ("answer","format","e","r","er","intermediate","a","winner","joint","rounding_joint")}}
        metrics[arm]["groups"]={"n":len(buckets),**{k:rubric.bounds(b[k] for b in groups[arm].values()) for k in PAIRED_FIELDS}}
    gid=set(groups["base"])
    assert all(set(v)==gid for v in groups.values()) and len(fixed_ids)==29 and set(fixed_ids)<=gid
    transitions={}
    for a,b in COMPARISONS:
        cell=transitions[f"{a}-{b}"]={}
        for ds in ("paired","general","sentinel"):
            l={r["id"]:r for r in subset(by[a],ds)}
            h={r["id"]:r for r in subset(by[b],ds)}
            cell[ds]={k:rubric.transitions({i:x[k] for i,x in l.items()},{i:x[k] for i,x in h.items()})
                      for k in ("answer","joint")}
        cell["groups"]={k:rubric.transitions({g:x[k] for g,x in groups[a].items()},
                                             {g:x[k] for g,x in groups[b].items()}) for k in PAIRED_FIELDS}

    common={}
    for cohort,ids in (("original_fixed29",sorted(fixed_ids)),("half_common_er",
           sorted(g for g in gid if groups["ordinary50"][g]["er"] is True and groups["enhanced50"][g]["er"] is True))):
        common[cohort]=dict(n=len(ids),ids=ids,joint={a:rubric.bounds(groups[a][g]["joint"] for g in ids) for a in ARMS},
                            er={a:rubric.bounds(groups[a][g]["er"] for g in ids) for a in ARMS})
        common[cohort]["half_comparison"]=rubric.transitions({g:groups["ordinary50"][g]["joint"] for g in ids},
                                                            {g:groups["enhanced50"][g]["joint"] for g in ids})
    possible=sorted(g for g in gid if groups["ordinary50"][g]["er"] is not False and groups["enhanced50"][g]["er"] is not False
                    and g not in common["half_common_er"]["ids"])
    common["half_common_er"]["possibly_additional_groups"]=possible
    front=metrics["enhanced50"]
    baseline=transitions["base-enhanced50"]
    assert metrics["base"]["paired"]["answer"] == [90,90] and metrics["base"]["general"]["answer"]==[96,96]
    tests=dict(paired_kept=len(baseline["paired"]["answer"]["kept"])>74,
               general_answer=front["general"]["answer"][0]>94,connection=front["groups"]["joint"][0]>21,
               general_connection=front["general"]["joint"][0]>=92,
               sentinel=len(baseline["sentinel"]["answer"]["kept"])==16)
    return dict(metrics=metrics,groups=groups,transitions=transitions,common=common,tests=tests,advance=all(tests.values()))


def interval(b):
    return str(b[0]) if b[0]==b[1] else f"{b[0]}–{b[1]}"


def mdrow(*values):
    return "| "+" | ".join(map(str,values))+" |"


def render(result):
    m,t,c=result["metrics"],result["transitions"],result["common"]
    tx=["# LoRA 적용 강도 50% 진단","","R9 입력 192개에 일반형과 A 보강형이 각각 한 번 답했다. 기존 LoRA 전·100% 모델의 응답과 채점은 같은 ID에서 재사용했다.",
        "",mdrow("지표",*(LABELS[a] for a in ARMS)),"|---|---:|---:|---:|---:|---:|"]
    for ds,k,label,n in (("groups","joint","짝 양쪽 정답과 ERA 연결",48),("groups","answer","짝 양쪽 정답",48),
                         ("groups","er","짝 양쪽 E·R",48),("paired","answer","규칙 정답",96),
                         ("paired","joint","규칙 정답과 ERA 연결",96),("general","answer","일반 정답",96),
                         ("general","joint","일반 정답과 ERA 연결",96),("sentinel","answer","보존 정답",16)):
        tx.append(mdrow(label,*(f"{interval(m[a][ds][k])}/{n}" for a in ARMS)))
    tx+=["","## LoRA 전 정답 유지","","| 비교 | 규칙 유지 / 90 | 규칙 회복 / 6 | 규칙 손실 | 일반 유지 / 96 | 일반 손실 |",
         "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS[1:]:
        pair,gen=(t["base-"+arm][ds]["answer"] for ds in ("paired","general"))
        tx.append(mdrow(LABELS[arm],len(pair["kept"]),len(pair["recovered"]),len(pair["lost"]),len(gen["kept"]),len(gen["lost"])))
    tx+=["","## 같은 E·R 그룹에서 연결","","| 고정한 그룹 | 그룹 수 | 일반 100% | 일반 50% | A 100% | A 50% |",
         "|---|---:|---:|---:|---:|---:|"]
    for name,label in (("original_fixed29","R9 두 100% 모델 공통 E·R"),("half_common_er","두 50% 모델 공통 E·R")):
        row=c[name]
        tx.append(mdrow(label,row["n"],*(interval(row["joint"][a]) for a in ARMS[1:])))
    tx+=["","## 진단 통과 여부","","| 사전 고정한 A 50% 조건 | 판정 |","|---|---|"]
    for k,label in GATES.items():
        tx.append(mdrow(label,"충족" if result["tests"][k] else "미충족"))
    tx+=["","새 독립 확인 평가 진입 조건을 "+("충족했다." if result["advance"] else "충족하지 못했다.")+
             " 이 보고서는 R9 진단 결과이며 새 독립셋·MMMU 점수는 포함하지 않는다.",
             "","[전체 문항별 변화](transitions.md) · [전체 채점 결과](eval/review/scored.jsonl) · [원장과 집계](eval/review/summary.json)"]
    (ROOT/"report.md").write_text("\n".join(tx)+"\n",encoding="utf8")
    ids=["# R9 비교의 문항별 변화","","유지·회복·손실은 각 행의 왼쪽 모델을 기준으로 한다. ‘양쪽 실패’에는 두 모델 모두 실패한 ID만 포함한다.",""]
    for a,b in COMPARISONS:
        title=LABELS[a]+" → "+LABELS[b]
        ids+=["## "+title,"","| 집합·기준 | 유지 | 회복 | 손실 | 양쪽 실패 | 미정 |","|---|---:|---:|---:|---:|---:|"]
        detail=[]
        for ds,k,label in (("paired","answer","규칙 정답"),("general","answer","일반 정답"),
                           ("general","joint","일반 연결"),("sentinel","answer","보존 정답"),("groups","joint","짝 양쪽 연결")):
            tr=t[f"{a}-{b}"][ds][k]
            ids.append(mdrow(label,*(len(tr[tag]) for tag in ("kept","recovered","lost","both_wrong","unknown"))))
            for tag,trans in (("recovered","회복"),("lost","손실")):
                detail.append(f"- {label} {trans} {', '.join(tr[tag]) or '없음'}")
        ids+=["",*detail,""]
    (ROOT/"transitions.md").write_text("\n".join(ids)+"\n",encoding="utf8")


def summarize(rows):
    fixed=read(OLD/"review/summary.json")["common_er"]["ids"]
    result=compute(rows,fixed)
    immutable("scored.jsonl",rows,rows=True)
    cached=read(REVIEW/"packet.json")
    proof={n:sha(REVIEW/n) for n in ("packet.json","cached.json","judge_complete.json","audit_packet.jsonl","manual.jsonl","scored.jsonl")}
    result.update(proof=proof,r9_scored_sha256=SCORED,r9_summary_sha256=SUMMARY,
                  fresh_calls=384,novel_unique=cached["unique"],cached_unique=cached["cached_keys"])
    immutable("summary.json",result)
    render(result)
    print(json.dumps(dict(tests=result["tests"],advance=result["advance"],fixed29=result["common"]["original_fixed29"]["joint"])),flush=True)
