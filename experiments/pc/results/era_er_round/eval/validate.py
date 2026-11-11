"""Independently reconstruct the observed promotion and retention counts."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def read(p):return json.loads(p.read_text(encoding="utf8"))
def lines(p):return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    src=HERE/"review"
    s=read(src/"summary.json")
    assert all(sha(src/name)==h for name,h in s["hashes"].items())
    rows=lines(src/"scored.jsonl")
    assert len(rows)==len({(r["arm"],r["id"]) for r in rows})==921
    armnames=("base","ordinary","enhanced")
    general={a:{r["id"]:r["answer"] for r in rows if r["arm"]==a and r["dataset"]=="general"} for a in armnames}
    assert len(general["base"])==96 and all(set(general["base"])==set(general[a]) for a in armnames)
    old={i for i,v in general["base"].items() if v}
    retained={a:len(old & {i for i,v in general[a].items() if v}) for a in armnames}
    state={}
    for a in armnames:
        pairs=defaultdict(list)
        for r in rows:
            if r["arm"]==a and r["dataset"]=="paired":pairs[(r["kind"],r["group"])].append(r)
        assert len(pairs)==48 and all(len(v)==2 for v in pairs.values())
        for (k,g),v in pairs.items():
            flags=[row[field] for row in v for field in ("answer","e","r","a","consistent")]
            state[a,k,g]=False if False in flags else None if None in flags else True
    groups=sorted({g for a,k,g in state})
    def advantage(kind,exclude=None):
        left=[v for (arm,k,g),v in state.items() if arm=="enhanced" and (k==kind or kind=="all") and g!=exclude]
        right=[v for (arm,k,g),v in state.items() if arm=="ordinary" and (k==kind or kind=="all") and g!=exclude]
        return sum(v is True for v in left)>sum(v is not False for v in right)
    tests={k:advantage(k) for k in ("all","E","R")}
    tests["general_vs_base"]=sum(general["enhanced"].values())>=len(old)
    tests["base_kept_vs_control"]=retained["enhanced"]>=retained["ordinary"]
    sensitive=[g for g in groups if any(tests[k] and not advantage(k,g) for k in ("all","E","R"))]
    assert tests==s["tests"] and set(sensitive)==set(s["sensitive_groups"])
    promote=all(tests.values()) and not sensitive
    assert promote==s["promote"]
    for a in armnames:
        fresh=HERE/"fresh/runs"/("r7-"+a)/"predictions.jsonl"
        v=lines(fresh);assert len(v)==len({r["id"] for r in v})==(192 if a=="base" else 307)
        assert not (fresh.parent/"submitted.json").exists()
        assert read(fresh.parent/"done.json")["output_sha256"]==sha(fresh)
    rec=dict(promote=promote,tests=tests,sensitive=sorted(sensitive),general_retained=retained,
             summary_sha256=sha(src/"summary.json"),scored_sha256=sha(src/"scored.jsonl"),
             unique_fresh={"base":192,"ordinary":307,"enhanced":307},cached_base=115)
    dst=ROOT/"decision.json"
    if dst.exists():assert read(dst)==rec
    else:dst.write_text(json.dumps(rec,indent=2)+"\n",encoding="utf8")
    print(rec)


if __name__=="__main__":main()
