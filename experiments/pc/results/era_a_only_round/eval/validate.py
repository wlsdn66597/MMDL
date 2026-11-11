"""Independent gate reconstruction from per-input grades, journals, and model receipts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding="utf8"))
def rows(p):return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    review=ROOT/"eval/review";summary=load(review/"summary.json")
    assert all(sha(review/k)==h for k,h in summary["proof"].items())
    records=rows(review/"scored.jsonl");arms=["base","ordinary","enhanced"]
    assert len(records)==len({(r["arm"],r["id"]) for r in records})==576
    ids=rows(ROOT/"eval/spec/inputs.jsonl")
    for a in arms:
        journal=ROOT/"eval/fresh/runs"/("r9-"+a)
        p=journal/"predictions.jsonl";d=rows(p)
        assert len(d)==len({r["id"] for r in d})==192 and [r["id"] for r in d]==[r["id"] for r in ids]
        assert all((r["prompt"],r["image_sha256"],r["seed"],r["max_output"])==(i["prompt"],i["image_sha256"],3407,2048) for r,i in zip(d,ids))
        assert sha(p)==load(journal/"done.json")["output_sha256"] and not (journal/"submitted.json").exists()
    by={a:[r for r in records if r["arm"]==a] for a in arms}
    answers={ds:{a:{r["id"] for r in v if r["dataset"]==ds and r["answer"]} for a,v in by.items()} for ds in ("paired","general")}
    sets=answers["general"]
    protected={r["id"] for r in by["base"] if r["sentinel"]}
    assert len(protected)==16
    group_ids={r["group"]:r["kind"] for r in by["base"] if r["dataset"]=="paired"}
    assert len(group_ids)==48
    group_scores={}
    for a in arms:
        for group in group_ids:
            v=[r for r in by[a] if r["dataset"]=="paired" and r["group"]==group]
            assert len(v)==2
            for name,fields in (("r",["r"]),("er",["e","r"]),("joint",["answer","e","r","a","intermediate","winner","consistent"])):
                flags=[r[f] for r in v for f in fields]
                group_scores[a,name,group]=False if False in flags else None if None in flags else True
    def diff(k,remove=None):
        return sum(group_scores["enhanced",k,g] is True for g in group_ids if g!=remove)-sum(group_scores["ordinary",k,g] is not False for g in group_ids if g!=remove)
    gained={group_ids[g] for g in group_ids if group_scores["enhanced","joint",g] is True and group_scores["ordinary","joint",g] is False}
    common=sorted(g for g in group_ids if all(group_scores[a,"er",g] is True for a in ("ordinary","enhanced")))
    possible=[g for g in group_ids if g not in common and all(group_scores[a,"er",g] is not False for a in ("ordinary","enhanced"))]
    net=lambda g:int(group_scores["enhanced","joint",g] is True)-int(group_scores["ordinary","joint",g] is not False)
    common_delta=sum(net(g) for g in common)+sum(min(0,net(g)) for g in possible)
    assert common==summary["common_er"]["ids"] and common_delta==summary["common_er"]["conservative_net"]
    newly_lost=sorted((sets["ordinary"]&sets["base"])-sets["enhanced"])
    assert newly_lost==summary["original_general_lost_ids"]
    paired=answers["paired"]
    tests=dict(rule_pairs_noninferior=diff("r")>=0,er_pairs_noninferior=diff("er")>=0,connected_pairs=diff("joint")>0,
               joint_leave_one=all(diff("joint",g)>0 for g in group_ids),
               common_er_connected=common_delta>0,
               recovered_families=len(gained)>=2,general_at_base=len(sets["enhanced"])>=len(sets["base"]),
               general_baseline_retained=len(sets["enhanced"]&sets["base"])>=len(sets["ordinary"]&sets["base"]),
               paired_baseline_retained=len(paired["enhanced"]&paired["base"])>=len(paired["ordinary"]&paired["base"]),
               original_general_no_new_loss=not newly_lost,
               sentinel_no_new_loss=not ((sets["ordinary"]-sets["enhanced"])&protected))
    assert tests==summary["tests"] and all(tests.values())==summary["promote"]
    out=dict(promote=all(tests.values()),tests=tests,fresh_unique_per_model=192,total_solver_calls=576,
             baseline_kept={a:len(sets["base"]&sets[a]) for a in arms},paired_baseline_kept={a:len(paired["base"]&paired[a]) for a in arms},
             original_general_losses=newly_lost,sentinel_losses=sorted((sets["ordinary"]-sets["enhanced"])&protected),
             summary_sha256=sha(review/"summary.json"),scored_sha256=sha(review/"scored.jsonl"))
    dst=ROOT/"decision.json"
    if dst.exists():assert load(dst)==out
    else:dst.write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print(out)


if __name__=="__main__":main()
