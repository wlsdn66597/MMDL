"""Audit blinded responses, compute paired retention, and apply the condition fixed before training."""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from answer_only import only_options

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REVIEW=HERE/"review"
ARMS=("base","ordinary","enhanced")
FIELDS=("e","r","intermediate","winner","a","consistent","first","corrected")
ERRORS={"none","no_grounded_answer","candidate_missing","other_evidence","wrong_rule","operands",
        "arithmetic","loser_rounding","winner_omitted","wrong_winner","wrong_letter","other","uncertain"}
LIB=ROOT.parent/"era_midstep_round/eval/score.py"


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def lines(p):
    return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze(p,data):
    if p.exists():assert p.read_bytes()==data,p
    else:p.write_bytes(data)


def store(p,rows):
    freeze(p,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode())


def obj(p,data):
    freeze(p,(json.dumps(data,ensure_ascii=False,indent=2)+"\n").encode())


def key(r):
    args=[r["id"],r["raw"],r["finish_reason"]]
    return hashlib.sha256(json.dumps(args,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def every(seq):
    seq=list(seq)
    return False if False in seq else None if None in seq else True


def bounds(seq):
    seq=list(seq)
    return [sum(x is True for x in seq),sum(x is not False for x in seq)]


def as_choice(raw,options,sc):
    first=sc.final_letter(raw,options)
    if first:return first
    tail=raw.strip().splitlines()[-1].replace("*","") if raw.strip() else ""
    m=re.fullmatch(r"(?i)\s*(?:(?:final answer|answer)\s*:\s*)?\(?([A-H])\)?[.:)\s]+(.+)\s*",tail)
    if m and ord(m[1].upper())<65+len(options) and m[2].strip(" .").casefold()==options[ord(m[1].upper())-65].strip(" .").casefold():
        return m[1].upper()
    return None


def records(final=False):
    assert sha(LIB)=="43ed42de0b9e5097838d0d93043afa053e8df3aa7a4f39d4dcfa2fde2ce2ecc2"
    lib=importlib.util.spec_from_file_location("frozen_scorer",LIB);sc=importlib.util.module_from_spec(lib);lib.loader.exec_module(sc)
    seal=read(REVIEW/"judge_complete.json");proof=read(REVIEW/"packet.json")
    for fn,k in (("judgments.jsonl","judgments_sha256"),("packet.jsonl","packet_sha256"),("judge_identity.json","identity_sha256")):
        assert sha(REVIEW/fn)==seal[k]
    cards={r["key"]:r for r in lines(REVIEW/"packet.jsonl")}
    grades={r["key"]:r for r in lines(REVIEW/"judgments.jsonl")}
    assert seal["n"]==proof["unique"]==len(grades)==len(cards)==len(lines(REVIEW/"judgments.jsonl")) and cards.keys()==grades.keys()
    bare={k for k,c in cards.items() if only_options(c["raw"],c["options"])}
    if final:
        req={c["key"] for c in lines(REVIEW/"audit_packet.jsonl")}
        patches=lines(REVIEW/"manual.jsonl")
        assert len(patches)==len(req) and req=={p["key"] for p in patches}
        for p in patches:
            assert p["reason"].strip() and set(p["fields"])<=set((*FIELDS,"choice"))
            for f,v in p["fields"].items():
                assert isinstance(v,str) and v in ["?",*"ABCDEFGH"] if f=="choice" else (v is None or type(v) is bool)
            grades[p["key"]].update(p["fields"])
            if cards[p["key"]]["dataset"]=="paired":
                assert p.get("first_error") in ERRORS,p
                grades[p["key"]]["first_error"]=p["first_error"]
    for k in bare:
        grades[k].update(e=False,r=False,a=False,intermediate=False,winner=False,first=None,corrected=None)
    gold={r["id"]:r for r in lines(HERE/"spec/gold.jsonl")}
    scores=[]
    for arm in ARMS:
        folder=HERE/"runs"/arm
        pred=folder/"predictions.jsonl"
        assert sha(pred)==read(folder/"receipt.json")["sha256"]==proof["per_arm"][arm]["sha256"]
        for p in lines(pred):
            k=key(p);z=grades[k];g=gold[p["id"]]
            letter=as_choice(p["raw"],g["options"],sc);choice=letter or z["choice"]
            correct=p["finish_reason"]=="stop" and choice==g["gold"]
            application=every([z["a"],z["intermediate"],z["winner"]])
            joint=every([correct,z["e"],z["r"],application,z["consistent"]])
            first_error="no_grounded_answer" if k in bare else z.get("first_error")
            if first_error=="loser_rounding":
                assert g["dataset"]=="paired" and correct and z["e"] is True and z["r"] is True and z["winner"] is True and z["consistent"] is True and z["intermediate"] is False,p["id"]
            if g["dataset"]=="paired" and final and k not in bare:
                assert (first_error=="none")==bool(joint is True),(p["id"],first_error,joint)
            scores.append(dict(id=p["id"],key=k,arm=arm,group=g["group"],kind=g["kind"],dataset=g["dataset"],sentinel=g["sentinel"],
                  choice=choice,fallback=letter is None,answer=correct,joint=joint,bare=k in bare,
                  format=sc.format_ok(p["raw"],p["finish_reason"]),finish=p["finish_reason"],**{f:application if f=="a" else z[f] for f in FIELDS},a_judgment=z["a"],
                  er=every([z["e"],z["r"]]),first_error=first_error,rounding_joint=True if first_error=="loser_rounding" else joint))
    assert len(scores)==len({(x["arm"],x["id"]) for x in scores})==576
    return scores,cards,grades


def audit():
    scores,cards,grades=records()
    ids=defaultdict(list)
    for r in scores:ids[r["id"]].append(r)
    fixed=sorted([i for i,v in ids.items() if v[0]["dataset"]=="general"],key=lambda i:hashlib.sha256(("r9-general-sample/"+i).encode()).hexdigest())[:8]
    selected=set()
    for i,v in ids.items():
        need=(v[0]["dataset"]=="paired" or v[0]["sentinel"] or i in fixed or any(r["fallback"] or r["corrected"] is True for r in v) or
              len({(r["answer"],r["joint"]) for r in v})>1)
        if need:selected.update(r["key"] for r in v if not r["bare"])
    packet=[dict(cards[k],judge={f:grades[k][f] for f in (*FIELDS,"choice")}) for k in sorted(selected)]
    store(REVIEW/"audit_packet.jsonl",packet)
    store(REVIEW/"bare.jsonl",[dict(key=r["key"],id=r["id"],raw=r["raw"]) for r in cards.values() if only_options(r["raw"],r["options"])])
    obj(REVIEW/"audit_receipt.json",dict(n=len(packet),unique=len(cards),sha256=sha(REVIEW/"audit_packet.jsonl"),bare=sha(REVIEW/"bare.jsonl")))
    print("Anonymous review",len(packet),"out of",len(cards))


def transitions(a,b):
    assert a.keys()==b.keys()
    out={k:[] for k in ("kept","recovered","lost","both_wrong","unknown")}
    for i in sorted(a):
        left,right=a[i],b[i]
        tag="unknown" if None in (left,right) else "kept" if left and right else "recovered" if right else "lost" if left else "both_wrong"
        out[tag].append(i)
    return out


def run():
    scores,_,_=records(final=True)
    metrics={};groups={};changes={}
    for a in ARMS:
        rr=[r for r in scores if r["arm"]==a]
        metrics[a]={}
        for ds in ("paired","general","sentinel"):
            subset=[r for r in rr if (r["sentinel"] if ds=="sentinel" else r["dataset"]==ds)]
            assert len(subset)==(16 if ds=="sentinel" else 96)
            m={k:bounds(x[k] for x in subset) for k in ("answer","joint","e","r","intermediate","winner","a","consistent","format")}
            wrong=[x for x in subset if x["first"] is False]
            m.update(n=len(subset),length=sum(x["finish"]=="length" for x in subset),first_errors=len(wrong),first_unknown=sum(x["first"] is None for x in subset),
                repaired=bounds(every([x["corrected"],x["answer"]]) for x in wrong))
            metrics[a][ds]=m
        pg=defaultdict(list)
        for r in rr:
            if r["dataset"]=="paired":pg[r["group"]].append(r)
        assert len(pg)==48 and all(len(v)==2 and v[0]["kind"]==v[1]["kind"] for v in pg.values())
        groups[a]={g:dict(kind=v[0]["kind"],**{k:every(x[k] for x in v) for k in ("r","er","joint","rounding_joint","answer")}) for g,v in pg.items()}
    for ds in ("paired","general","sentinel"):
        changes[ds]={}
        for a,b in (("base","ordinary"),("base","enhanced"),("ordinary","enhanced")):
            for k in ("answer","r","joint"):
                one=lambda arm:{x["id"]:x[k] for x in scores if x["arm"]==arm and (x["sentinel"] if ds=="sentinel" else x["dataset"]==ds)}
                changes[ds][f"{a}-{b}-{k}"]=transitions(one(a),one(b))
    pairdiff={k:transitions({g:x[k] for g,x in groups["ordinary"].items()},{g:x[k] for g,x in groups["enhanced"].items()}) for k in ("r","er","joint","rounding_joint","answer")}
    def net(k,remove=None):
        h=bounds(v[k] for g,v in groups["enhanced"].items() if g!=remove)
        c=bounds(v[k] for g,v in groups["ordinary"].items() if g!=remove)
        return h[0]-c[1]
    gained_families=sorted({groups["enhanced"][g]["kind"] for g in pairdiff["joint"]["recovered"]})
    common=[g for g in groups["ordinary"] if groups["ordinary"][g]["er"] is True and groups["enhanced"][g]["er"] is True]
    possible=[g for g in groups["ordinary"] if g not in common and None in (groups["ordinary"][g]["er"],groups["enhanced"][g]["er"])
              and False not in (groups["ordinary"][g]["er"],groups["enhanced"][g]["er"])]
    advantage=lambda g:int(groups["enhanced"][g]["joint"] is True)-int(groups["ordinary"][g]["joint"] is not False)
    common_lower=sum(advantage(g) for g in common)+sum(min(0,advantage(g)) for g in possible)
    original_lost=sorted(set(changes["general"]["base-ordinary-answer"]["kept"])&set(changes["general"]["ordinary-enhanced-answer"]["lost"]))
    keep=lambda ds,arm:len(changes[ds][f"base-{arm}-answer"]["kept"])
    tests=dict(rule_pairs_noninferior=net("r")>=0,er_pairs_noninferior=net("er")>=0,connected_pairs=net("joint")>0,
               joint_leave_one=min(net("joint",g) for g in groups["ordinary"])>0,
               recovered_families=len(gained_families)>=2,
               common_er_connected=common_lower>0,
               general_at_base=metrics["enhanced"]["general"]["answer"][0]>=metrics["base"]["general"]["answer"][1],
               general_baseline_retained=keep("general","enhanced")>=keep("general","ordinary"),
               paired_baseline_retained=keep("paired","enhanced")>=keep("paired","ordinary"),
               original_general_no_new_loss=not original_lost,
               sentinel_no_new_loss=not changes["sentinel"]["ordinary-enhanced-answer"]["lost"])
    proof={n:sha(REVIEW/n) for n in ("packet.jsonl","judgments.jsonl","bare.jsonl","audit_packet.jsonl","manual.jsonl")}
    common_report=dict(ids=sorted(common),possible_members=sorted(possible),n=len(common),
          ordinary=bounds(groups["ordinary"][g]["joint"] for g in common),enhanced=bounds(groups["enhanced"][g]["joint"] for g in common),
          conservative_net=common_lower)
    out=dict(metrics=metrics,groups=groups,changes=changes,pair_changes=pairdiff,recovered_families=gained_families,
             common_er=common_report,original_general_lost_ids=original_lost,
             first_error_counts={a:dict(Counter(x["first_error"] for x in scores if x["arm"]==a and x["dataset"]=="paired")) for a in ARMS},
             tests=tests,promote=all(tests.values()),proof=proof)
    store(REVIEW/"scored.jsonl",scores);obj(REVIEW/"summary.json",out)
    print(tests,out["promote"])


if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("mode",choices=["audit","score"])
    if cli.parse_args().mode=="audit":audit()
    else:run()
