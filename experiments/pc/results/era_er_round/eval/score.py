"""Paired semantic connection, answer retention and group-level sensitivity."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REVIEW=HERE/"review"
ARMS=("base","ordinary","enhanced")
SETS=("paired","general","charts","science")
LIB=ROOT.parent/"era_midstep_round/eval/score.py"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def lines(p):
    return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))


def write(p,content):
    if p.exists():assert p.read_bytes()==content,p
    else:p.write_bytes(content)


def write_json(p,data):
    write(p,(json.dumps(data,ensure_ascii=False,indent=2)+"\n").encode())


def write_lines(p,data):
    write(p,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in data).encode())


def bounds(values):
    v=list(values)
    return [sum(x is True for x in v),sum(x is not False for x in v)]


def every(values):
    v=list(values)
    if False in v:return False
    return None if None in v else True


def review_key(i,raw,finish):
    return hashlib.sha256(json.dumps([i,raw,finish],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def load_rows(manual=False):
    assert sha(LIB)=="43ed42de0b9e5097838d0d93043afa053e8df3aa7a4f39d4dcfa2fde2ce2ecc2"
    s=importlib.util.spec_from_file_location("historic_scorer",LIB);sc=importlib.util.module_from_spec(s);s.loader.exec_module(sc)
    seal=read(REVIEW/"judge_complete.json")
    assert seal["judgments_sha256"]==sha(REVIEW/"judgments.jsonl") and seal["identity_sha256"]==sha(REVIEW/"judge_identity.json")
    assert seal["packet_sha256"]==sha(REVIEW/"packet.jsonl")
    gold={r["id"]:r for r in lines(HERE/"spec/gold.jsonl")}
    packet={r["key"]:r for r in lines(REVIEW/"packet.jsonl")}
    grades={r["key"]:r for r in lines(REVIEW/"judgments.jsonl")}
    assert len(gold)==307 and packet.keys()==grades.keys() and len(grades)==seal["n"]==len(lines(REVIEW/"judgments.jsonl"))
    bare=REVIEW/"answer_only.jsonl"
    if bare.exists():
        for p in lines(bare):
            assert packet[p["key"]]["raw"]==p["raw"]
            grades[p["key"]].update(p["fields"])
    if manual:
        patches=lines(REVIEW/"manual.jsonl")
        assert len(patches)==len({p["key"] for p in patches})
        required={r["key"] for r in lines(REVIEW/"audit_packet.jsonl")}
        assert {p["key"] for p in patches}==required
        for p in patches:
            assert p["reason"].strip()
            for k,v in p["fields"].items():
                assert k in ("e","r","a","consistent","first","corrected","choice")
                assert (v in "ABCDEFGH?" if k=="choice" else v in (True,False,None))
                grades[p["key"]][k]=v
        extra=REVIEW/"extra_manual.jsonl"
        if extra.exists():
            patches=lines(extra)
            assert {p["key"] for p in patches}=={r["key"] for r in lines(REVIEW/"extra_packet.jsonl")}
            assert all(p["reason"].strip() for p in patches)
            for p in patches:grades[p["key"]].update(p["fields"])
    rows=[]
    for arm in ARMS:
        path=HERE/"runs"/arm/"predictions.jsonl"
        assert sha(path)==read(path.parent/"receipt.json")["sha256"]
        for r in lines(path):
            g=gold[r["id"]];k=review_key(r["id"],r["raw"],r["finish_reason"])
            z=grades[k]
            choice=sc.final_letter(r["raw"],g["options"])
            if not choice:
                tail=r["raw"].strip().splitlines()[-1].replace("*","") if r["raw"].strip() else ""
                m=re.fullmatch(r"(?i)\s*(?:final answer|answer)\s*:\s*\(?([A-H])\)?[.:)\s]+(.+)\s*",tail)
                if m and ord(m[1].upper())<65+len(g["options"]):
                    opt=g["options"][ord(m[1].upper())-65]
                    if m[2].strip(" .").casefold()==opt.strip(" .").casefold():choice=m[1].upper()
            fallback=choice is None
            choice=choice or z["choice"]
            answer=(choice==g["gold"]) and r["finish_reason"]=="stop"
            joint=every([answer,z["e"],z["r"],z["a"],z["consistent"]])
            rows.append(dict(arm=arm,key=k,id=r["id"],dataset=g["dataset"],group=g["group"],kind=g["kind"],
                e=z["e"],r=z["r"],a=z["a"],consistent=z["consistent"],first=z["first"],corrected=z["corrected"],
                choice=choice,extraction_fallback=fallback,answer=answer,joint=joint,format=sc.format_ok(r["raw"],r["finish_reason"])))
    assert len(rows)==len({(r["arm"],r["id"]) for r in rows})==921
    return rows,packet


def group_states(rows):
    store=defaultdict(list)
    for r in rows:
        if r["dataset"]=="paired":store[r["group"]].append(r)
    assert len(store)==48 and all(len(v)==2 for v in store.values())
    return {g:(v[0]["kind"],every(x["joint"] for x in v)) for g,v in store.items()}


def audit():
    rows,packet=load_rows()
    groups={a:group_states([r for r in rows if r["arm"]==a]) for a in ARMS}
    # Look at all discordant paired groups and fixed calibration groups even when both arms agree.
    selected={g for g in groups["base"] if groups["ordinary"][g][1]!=groups["enhanced"][g][1] or
              None in (groups["ordinary"][g][1],groups["enhanced"][g][1])}
    for kind in ("E","R","A","choice"):
        candidates=sorted([g for g,(k,v) in groups["base"].items() if k==kind],key=lambda g:hashlib.sha256(("r7-audit/"+g).encode()).digest())
        selected.update(candidates[:2])
    extra=set()
    for ds in ("general","science","charts"):
        ids=sorted({r["id"] for r in rows if r["dataset"]==ds},key=lambda i:hashlib.sha256(("r7-qc/"+i).encode()).digest())
        extra.update(ids[:4])
    keys={r["key"] for r in rows if (r["dataset"]=="paired" and r["group"] in selected) or r["id"] in extra or
          (r["dataset"]=="general" and r["extraction_fallback"]) or r["corrected"] is True}
    cards=[packet[k] for k in sorted(keys)]
    write_lines(REVIEW/"audit_packet.jsonl",cards)
    print(len(cards),"blind audit keys")


def paired_counts(base,new):
    assert base.keys()==new.keys()
    c=Counter()
    for i,x in base.items():
        y=new[i]
        k="unknown" if None in (x,y) else "kept" if x and y else "lost" if x else "gained" if y else "both_wrong"
        c[k]+=1
    return {k:c[k] for k in ("kept","gained","lost","both_wrong","unknown")}


def extra_audit():
    rows,packet=load_rows(manual=True)
    gg={a:group_states([r for r in rows if r["arm"]==a]) for a in ARMS}
    discord={g for g in gg["base"] if gg["ordinary"][g][1]!=gg["enhanced"][g][1] or None in
             (gg["ordinary"][g][1],gg["enhanced"][g][1])}
    checked={r["key"] for r in lines(REVIEW/"audit_packet.jsonl")} | {r["key"] for r in lines(REVIEW/"answer_only.jsonl")}
    keys={r["key"] for r in rows if r["dataset"]=="paired" and r["group"] in discord}-checked
    write_lines(REVIEW/"extra_packet.jsonl",[packet[k] for k in sorted(keys)])
    print(len(keys),"additional keys")


def finish():
    rows,_=load_rows(manual=True)
    assert (REVIEW/"extra_manual.jsonl").exists() and (REVIEW/"answer_only.jsonl").exists()
    allgroups={};details={}
    for arm in ARMS:
        subset=[r for r in rows if r["arm"]==arm]
        allgroups[arm]=group_states(subset)
        details[arm]={}
        for ds in SETS:
            rr=[r for r in subset if r["dataset"]==ds]
            details[arm][ds]={k:bounds(x[k] for x in rr) for k in ("answer","e","r","a","consistent","joint","format")}
            wrong=[r for r in rr if r["first"] is False]
            details[arm][ds].update(n=len(rr),first_errors=len(wrong),first_unknown=sum(x["first"] is None for x in rr),
                repaired=bounds(every([x["corrected"],x["answer"]]) for x in wrong))
        details[arm]["groups"]={k:bounds(v for label,v in allgroups[arm].values() if k=="all" or label==k)
            for k in ("all","E","R","A","choice")}
    changes={}
    for ds in SETS:
        changes[ds]={}
        for old,new in (("base","ordinary"),("base","enhanced"),("ordinary","enhanced")):
            for field in ("answer","joint"):
                prev={r["id"]:r[field] for r in rows if r["dataset"]==ds and r["arm"]==old}
                nxt={r["id"]:r[field] for r in rows if r["dataset"]==ds and r["arm"]==new}
                changes[ds][f"{old}-{new}-{field}"]=paired_counts(prev,nxt)
    def gt(kind,remove=None):
        def bound(arm):
            return bounds(v for g,(k,v) in allgroups[arm].items() if g!=remove and (kind=="all" or kind==k))
        return bound("enhanced")[0]>bound("ordinary")[1]
    tests={k:gt(k) for k in ("all","E","R")}
    tests.update(general_vs_base=details["enhanced"]["general"]["answer"][0]>=details["base"]["general"]["answer"][1],
        base_kept_vs_control=changes["general"]["base-enhanced-answer"]["kept"]>=changes["general"]["base-ordinary-answer"]["kept"])
    single=[g for g in allgroups["base"] if any(not gt(k,g) for k in ("all","E","R") if tests[k])]
    passed=all(tests.values())
    summary=dict(arms=details,changes=changes,groups=allgroups,tests=tests,sensitive_groups=single,
                 primary_pass=passed,promote=passed and not single,
                 hashes={s:sha(REVIEW/s) for s in ("packet.jsonl","judgments.jsonl","audit_packet.jsonl","manual.jsonl","answer_only.jsonl","extra_packet.jsonl","extra_manual.jsonl")})
    write_lines(REVIEW/"scored.jsonl",rows)
    write_json(REVIEW/"summary.json",summary)
    print(tests,"sensitive",len(single),"promote",summary["promote"])


if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("mode",choices=["audit","extra","finish"]);a=cli.parse_args()
    if a.mode=="audit":audit()
    elif a.mode=="extra":extra_audit()
    else:finish()
