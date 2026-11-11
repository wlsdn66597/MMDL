"""Independently replay the release gate and keep matched item and scene differences."""
import hashlib
import json
from math import comb
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REVIEW=ROOT/"eval/review"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(gain, loss):
    n=gain+loss
    return min(1.,2*sum(comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.


def paired(left,right):
    assert left.keys()==right.keys()
    groups={s:[] for s in ("gained","lost","both_correct","both_wrong","unknown")}
    for rid,a in left.items():
        b=right[rid]
        key="unknown" if None in (a,b) else "both_correct" if a and b else "lost" if a else "gained" if b else "both_wrong"
        groups[key].append(rid)
    return dict(counts={k:len(v) for k,v in groups.items()},gained=groups["gained"],lost=groups["lost"],
                unknown=groups["unknown"],exact_p_among_determinate=exact(len(groups["gained"]),len(groups["lost"])))


def bounds(vec):
    return [sum(v is True for v in vec),sum(v is not False for v in vec)]


def both(a,b):
    if a is False or b is False:return False
    if a is None or b is None:return None
    return True


def main():
    canonical=read(REVIEW/"summary.json")
    for file,expected in canonical["grade_proof"].items():
        assert digest(REVIEW/file)==expected
    rows=[json.loads(x) for x in (REVIEW/"scored_items.jsonl").read_text(encoding="utf8").splitlines()]
    models=("base","ordinary","preserved")
    metrics={};scenes={};retained={};science_hits={};strict={}
    for arm in models:
        items=[r for r in rows if r["arm"]==arm]
        assert len(items)==len({(r["dataset"],r["id"]) for r in items})==115
        science_hits[arm]={r["id"]:r["answer"] for r in items if r["dataset"]=="science"}
        assert len(science_hits[arm])==19
        strict[arm]=sum(r["format"] for r in items)
        charts=[r for r in items if r["dataset"]=="charts"]
        scenes[arm]={}
        for name in {r["scene"] for r in charts}:
            group=[r["joint"] for r in charts if r["scene"]==name]
            assert len(group)==8
            scenes[arm][name]=False if False in group else None if None in group else True
        assert len(scenes[arm])==12 and scenes[arm]==canonical["scene_labels"][arm]
    for arm in models[1:]:
        retained[arm]=bounds([both(science_hits["base"][k],v) for k,v in science_hits[arm].items()])
    change=paired(scenes["ordinary"],scenes["preserved"])
    scene_domain={r["scene"]:r["domain"] for r in rows if r["dataset"]=="charts"}
    gained_domains=sorted({scene_domain[s] for s in change["gained"]})
    tests=dict(natural_total=bounds(science_hits["preserved"].values())[0]>=bounds(science_hits["base"].values())[1],
               retained_original=retained["preserved"][0]>retained["ordinary"][1],
               joint_8_scene_increase=bounds(scenes["preserved"].values())[0]>bounds(scenes["ordinary"].values())[1],
               multi_scene_and_domain=len(change["gained"])>=2 and len(gained_domains)>=2,
               preserved_90pct_strict_format=strict["preserved"]>=104)
    assert tests==canonical["tests"] and all(tests.values())==canonical["gate"]
    for dataset in ("charts","science"):
        metrics[dataset]={}
        for field in ("E","R","A","consistent","answer","joint"):
            table={m:{r["id"]:r[field] for r in rows if r["dataset"]==dataset and r["arm"]==m} for m in models}
            metrics[dataset][field]=paired(table["ordinary"],table["preserved"])
    reason=("All preregistered criteria passed" if all(tests.values()) else
            "Criteria not established: "+", ".join(k for k,v in tests.items() if not v))
    decision=dict(promote=all(tests.values()),reason=reason,tests=tests,
                  score_sha256=digest(REVIEW/"summary.json"),scenes=change,gained_domains=gained_domains,
                  science_retained_bounds=retained,strict=strict,item_comparisons=metrics)
    (ROOT/"decision.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
