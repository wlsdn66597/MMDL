"""Summarize intervention transitions and the preregistered general-50 pilot gate."""
from collections import defaultdict
from score_cue import ARMS, HALF_SCORED, HALF_SUMMARY, HERE, REVIEW, ROOT, immutable, read, rubric, sha

METRICS=("answer","joint","e","r","er","intermediate","winner","a","consistent","bare","format")
GROUP_FIELDS=("answer","r","er","joint")
LABELS={"base":"LoRA 전", "ordinary":"일반 ERA 50%", "enhanced":"A 보강 ERA 50%"}
SUBSETS={"all":"전체","general":"일반","paired":"규칙","sentinel":"보존 전용","ChartQA":"ChartQA","ScienceQA":"ScienceQA"}
THRESHOLDS=dict(general_joint=92,general_answer=95,paired_base_kept=90,pair_joint=32)
GATES=dict(general_joint="일반 문항 정답과 연결",general_answer="일반 정답",paired_base_kept="LoRA 전 규칙 정답 유지",
           pair_joint="짝 양쪽 정답과 연결")


def subset(rows,ds):
    if ds=="all":return rows
    if ds=="sentinel":return [r for r in rows if r["sentinel"]]
    if ds in ("paired","general"):return [r for r in rows if r["dataset"]==ds]
    if ds=="ScienceQA":return [r for r in rows if r["id"].startswith("s9-sqa-")]
    if ds=="ChartQA":return [r for r in rows if r["id"].startswith("s9-chart-")]
    raise ValueError(ds)


def census(before,after):
    right={r["id"]:r for r in after}
    ids=sorted(r["id"] for r in before if r["bare"] and r["answer"])
    outcome=dict(eligible=ids,
                 linked=[i for i in ids if right[i]["joint"] is True],
                 answer_kept=[i for i in ids if right[i]["answer"]],
                 answer_lost=[i for i in ids if not right[i]["answer"]],
                 still_bare_correct=[i for i in ids if right[i]["bare"] and right[i]["answer"]],
                 nonbare_correct_partial=[i for i in ids if not right[i]["bare"] and right[i]["answer"] and right[i]["joint"] is False],
                 connection_unknown=[i for i in ids if right[i]["joint"] is None])
    assert set(outcome["answer_kept"])==set(outcome["linked"]+outcome["still_bare_correct"]+
        outcome["nonbare_correct_partial"]+outcome["connection_unknown"])
    return outcome


def compute(rows):
    leads=[a+"_"+side for a in ARMS for side in ("before","cue")]
    by={tag:[r for r in rows if r["arm"]==tag] for tag in leads}
    assert len(rows)==len({(r["arm"],r["id"]) for r in rows})==1152
    ids={r["id"] for r in by["base_before"]}
    assert len(ids)==192 and all({r["id"] for r in rs}==ids for rs in by.values())
    groups,metrics,retained={}, {}, {}
    base_set={r["id"] for r in by["base_before"] if r["dataset"]=="paired" and r["answer"]}
    assert len(base_set)==90
    for tag,rs in by.items():
        groups[tag]={};metrics[tag]={}
        rs_by_group=defaultdict(list)
        for r in subset(rs,"paired"):rs_by_group[r["group"]].append(r)
        assert len(rs_by_group)==48 and all(len(v)==2 for v in rs_by_group.values())
        for g,v in rs_by_group.items():
            groups[tag][g]={k:rubric.every(r[k] for r in v) for k in GROUP_FIELDS}
        for label in SUBSETS:
            s=subset(rs,label)
            metrics[tag][label]={"n":len(s),**{k:rubric.bounds(r[k] for r in s) for k in METRICS}}
        assert [metrics[tag][k]["n"] for k in ("general","paired","sentinel","ScienceQA","ChartQA")]==[96,96,16,64,16]
        metrics[tag]["groups"]={"n":48,**{k:rubric.bounds(r[k] for r in groups[tag].values()) for k in GROUP_FIELDS}}
        passed={r["id"] for r in subset(rs,"paired") if r["answer"]}
        retained[tag]={"kept":sorted(base_set&passed),"lost":sorted(base_set-passed),"recovered":sorted(passed-base_set)}
    cohorts={};changes={}
    for arm in ARMS:
        left,right=arm+"_before",arm+"_cue"
        cohorts[arm]={}
        changes[arm]={}
        for ds in SUBSETS:
            a,b=subset(by[left],ds),subset(by[right],ds)
            cohorts[arm][ds]=census(a,b)
            changes[arm][ds]={f:rubric.transitions({r["id"]:r[f] for r in a},{r["id"]:r[f] for r in b})
                              for f in ("answer","joint")}
        changes[arm]["groups"]={f:rubric.transitions({g:r[f] for g,r in groups[left].items()},
                                                     {g:r[f] for g,r in groups[right].items()}) for f in GROUP_FIELDS}
    assert len(cohorts["ordinary"]["general"]["eligible"])==sum(r["bare"] and r["answer"] for r in subset(by["ordinary_before"],"general"))
    primary=metrics["ordinary_cue"]
    counts=dict(general_joint=primary["general"]["joint"][0],general_answer=primary["general"]["answer"][0],
                paired_base_kept=len(retained["ordinary_cue"]["kept"]),pair_joint=primary["groups"]["joint"][0])
    tests={k:counts[k]>=threshold for k,threshold in THRESHOLDS.items()}
    return dict(metrics=metrics,groups=groups,cohorts=cohorts,changes=changes,base_retained=retained,
                primary="ordinary_cue",counts=counts,thresholds=THRESHOLDS,tests=tests,advance=all(tests.values()))


def md(*s):
    return "| "+" | ".join(map(str,s))+" |"


def cell(x):
    return str(x[0]) if x[0]==x[1] else f"{x[0]}–{x[1]}"


def render(s):
    m=s["metrics"];c=s["cohorts"]
    text=["# 짧은 ERA 지시 단일 호출 실험","",
          "기존 R9의 규칙 48쌍과 일반 96문항에서 각 모델이 한 번씩 답했다. 새로운 576개 응답과 동일한 모델·문항의 지시 전 응답을 비교했다. 일반 50%를 주 모델로 사전 지정했다.","",
          "## 보기만 쓰고 맞혔던 일반 문항","",
          md("모델","기존 대상","지시 후 정답·연결 성공","정답 유지","신규 오답","여전히 보기만 쓰고 정답"),"|---|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        x=c[a]["general"]
        text.append(md(LABELS[a],*[len(x[k]) for k in ("eligible","linked","answer_kept","answer_lost","still_bare_correct")]))
    text+=["","## 같은 문항의 전체 비교","","각 칸은 기존 응답에서 지시 추가 응답으로의 변화다. 미정이 있으면 하한과 상한을 함께 표시한다.","",
           md("지표",*(LABELS[a] for a in ARMS)),"|---|---:|---:|---:|"]
    for ds,k,label,n in (("general","joint","일반 정답과 ERA 연결",96),("general","answer","일반 정답",96),
                         ("groups","joint","짝 양쪽 정답과 ERA 연결",48),("paired","answer","규칙 정답",96),
                         ("groups","r","짝 양쪽 규칙",48),("sentinel","answer","보존 정답",16),
                         ("general","bare","일반 보기만 쓴 응답",96)):
        text.append(md(label,*[f"{cell(m[a+'_before'][ds][k])} → {cell(m[a+'_cue'][ds][k])} / {n}" for a in ARMS]))
    text.append(md("LoRA 전 규칙 정답 유지",*[f"{len(s['base_retained'][a+'_before']['kept'])} → {len(s['base_retained'][a+'_cue']['kept'])} / 90" for a in ARMS]))
    text+=["","## 일반 ERA 50%의 다음 단계 진입 판단","",md("사전 조건","기준","확정 수","판정"),"|---|---:|---:|---|"]
    for k in GATES:text.append(md(GATES[k],s["thresholds"][k],s["counts"][k],"통과" if s["tests"][k] else "미달"))
    text+=["","네 조건을 모두 충족했다. 다음 단계는 새로운 훈련 이미지로 짧은 ERA를 학습한 뒤 지시 없는 독립셋에서 확인하는 것이다."
           if s["advance"] else "네 조건을 모두 충족하지 못했다. 다음 단계는 추가된 지시가 바꾼 문항의 실제 근거·계산·보기 대응을 확인하는 것이다.",
           "","이번 점수는 지시가 있는 진단 성능이다. 공식 MMMU 575/900 유지 여부는 미측정이다.","",
           "[문항별 변화](transitions.md) · [공식 집계](eval/review/summary.json) · [응답별 판정](eval/review/scored.jsonl)"]
    (ROOT/"report.md").write_text("\n".join(text)+"\n",encoding="utf8")
    ids=["# 문항별 정답 및 연결 변화",""]
    for a in ARMS:
        ids.extend(["## "+LABELS[a],"",md("집합·기준","유지","회복","손실","양쪽 실패","미정"),"|---|---:|---:|---:|---:|---:|"])
        detail=[]
        for ds,k,label in (("general","answer","일반 정답"),("general","joint","일반 연결"),("paired","answer","규칙 정답"),
                            ("groups","joint","양쪽 연결"),("sentinel","answer","보존 정답")):
            t=s["changes"][a][ds][k]
            ids.append(md(label,*[len(t[j]) for j in ("kept","recovered","lost","both_wrong","unknown")]))
            for tag,v in (("회복","recovered"),("손실","lost")):
                detail.append(f"- {label} {tag} {', '.join(t[v]) or '없음'}")
        ids.extend(["",*detail,"",md("기존 보기만 쓰고 맞힌 집합","대상","연결 회복","정답 유지","신규 오답"),"|---|---:|---:|---:|---:|"])
        for ds in ("general","paired","ScienceQA","ChartQA","sentinel","all"):
            d=c[a][ds];ids.append(md(SUBSETS[ds],*[len(d[k]) for k in ("eligible","linked","answer_kept","answer_lost")]))
        ids.extend(["",f"- 일반 보기만 쓴 기존 정답 중 새 연결 성공 {', '.join(c[a]['general']['linked']) or '없음'}",
                    f"- 일반 보기만 쓴 기존 정답 중 신규 오답 {', '.join(c[a]['general']['answer_lost']) or '없음'}",""])
    (ROOT/"transitions.md").write_text("\n".join(ids)+"\n",encoding="utf8")


def summarize(rows):
    s=compute(rows)
    immutable("scored.jsonl",rows,rows=True)
    packet=read(REVIEW/"packet.json")
    s.update(proof={f:sha(REVIEW/f) for f in ("packet.json","cached.json","judge_complete.json","audit_packet.jsonl","manual.jsonl","scored.jsonl")},
             new_solver_calls=576,unique_novel=packet["unique"],unique_cached=packet["cached_keys"],
             cue_inputs_sha256=sha(HERE/"spec/inputs.jsonl"),half_scored_sha256=HALF_SCORED,half_summary_sha256=HALF_SUMMARY)
    immutable("summary.json",s)
    render(s)
    print(dict(counts=s["counts"],tests=s["tests"],advance=s["advance"]),flush=True)
