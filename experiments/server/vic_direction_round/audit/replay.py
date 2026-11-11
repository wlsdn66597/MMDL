"""Recompute gate and matched changes from the sealed item-level development grades."""
from collections import Counter
import hashlib
import json
from math import comb
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"eval/review"


def load(p):
    return json.loads(p.read_text(encoding="utf8"))


def bounds(values):
    return [sum(v is True for v in values),sum(v is not False for v in values)]


def both(values):
    values=list(values)
    if False in values:return False
    if None in values:return None
    return True


def paired(old,new):
    assert old.keys()==new.keys()
    t={k:[] for k in ("retained","gained","lost","both_wrong","unknown")}
    for rid in sorted(old):
        a,b=old[rid],new[rid]
        k="unknown" if None in (a,b) else "retained" if a and b else "gained" if b else "lost" if a else "both_wrong"
        t[k].append(rid)
    return t


def main():
    scores=load(OUT/"summary.json")
    official=load(ROOT/"decision.json")
    assert hashlib.sha256((OUT/"summary.json").read_bytes()).hexdigest()==official["score_sha256"]
    for f,h in scores["grade_proof"].items():
        assert hashlib.sha256((OUT/f).read_bytes()).hexdigest()==h
    rows=[json.loads(x) for x in (OUT/"scored_items.jsonl").read_text(encoding="utf8").splitlines()]
    arms=("base","ordinary","preserved")
    assert len(rows)==len({(r["arm"],r["dataset"],r["id"]) for r in rows})==345
    by={(a,ds):{r["id"]:r for r in rows if r["arm"]==a and r["dataset"]==ds} for a in arms for ds in ("charts","science")}
    for a in arms:
        assert len(by[a,"charts"])==96 and len(by[a,"science"])==19
    groups={}
    for a in arms:
        chart=list(by[a,"charts"].values())
        groups[a]={}
        for name in sorted({c["scene"] for c in chart}):
            subs=[c["joint"] for c in chart if c["scene"]==name]
            assert len(subs)==8
            groups[a][name]=both(subs)
        assert groups[a]==scores["scene_labels"][a]
    changes=paired(groups["ordinary"],groups["preserved"])
    domain={r["scene"]:r["domain"] for r in rows if r["dataset"]=="charts"}
    gained_domains=sorted({domain[x] for x in changes["gained"]})
    sc={a:{i:r["answer"] for i,r in by[a,"science"].items()} for a in arms}
    ret={a:bounds([both((sc["base"][i],v)) for i,v in sc[a].items()]) for a in arms}
    tests=dict(natural_total=bounds(sc["preserved"].values())[0]>=bounds(sc["base"].values())[1],
               retained_original=ret["preserved"][0]>ret["ordinary"][1],
               joint_8_scene_increase=bounds(groups["preserved"].values())[0]>bounds(groups["ordinary"].values())[1],
               multi_scene_and_domain=len(changes["gained"])>=2 and len(gained_domains)>=2,
               preserved_90pct_strict_format=sum(r["format"] for r in rows if r["arm"]=="preserved")>=104)
    assert tests==scores["tests"] and all(tests.values())==scores["gate"]==official["promote"]
    ng,nl=len(changes["gained"]),len(changes["lost"])
    n=ng+nl
    scene_p=min(1.,2*sum(comb(n,k) for k in range(min(ng,nl)+1))/2**n) if n else 1.
    summary=dict(tests=tests,promote=all(tests.values()),scenes=changes,scene_exact_two_sided_p=scene_p,
                 scene_gained_domains=gained_domains,science_retained_bounds=ret,
                 control_to_reverse_science=paired(sc["ordinary"],sc["preserved"]))
    for a,name in (("ordinary","paraphrase"),("preserved","reversed")):
        summary["base_to_"+name+"_science"]=paired(sc["base"],sc[a])
    for ds in ("science","charts"):
        for field in ("joint","answer"):
            left={i:r[field] for i,r in by["ordinary",ds].items()}
            right={i:r[field] for i,r in by["preserved",ds].items()}
            summary[f"{ds}_control_to_reverse_{field}"]=paired(left,right)
    target=ROOT/"audit/gate_replay.json"
    content=json.dumps(summary,ensure_ascii=False,indent=2)+"\n"
    if target.exists():assert load(target)==summary
    else:target.write_text(content,encoding="utf8")
    print(tests,flush=True)


if __name__=="__main__":main()
