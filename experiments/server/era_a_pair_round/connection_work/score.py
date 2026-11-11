"""Blind content-review queue and paired accounting; never load or query a solver."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s]


def put(path, data):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(data,encoding="utf8")
    temp.replace(path)


def save(path, data):
    put(path,json.dumps(data,ensure_ascii=False,indent=2)+"\n")


def save_rows(path,data):
    put(path,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in data))


def final(raw, options):
    # Deliberately handle only explicit, terminal declarations. Other cases go to a blinded reviewer.
    plain=raw.strip().replace("*","").replace("`","").replace("$","").replace("−","-")
    explicit = re.findall(r"(?im)^\s*(?:Final\s+answer|Answer)\s*:\s*([^\n]*)$",plain)
    candidate = explicit[-1] if explicit and plain.endswith(explicit[-1].strip()) else plain.splitlines()[-1] if plain else ""
    candidate = re.sub(r"^\\(?:boxed|text)\{([^{}]+)\}\s*\.?$", r"\1",candidate.strip())
    m = re.fullmatch(r"(?i)\s*(?:option |choice )?[\[(]?([A-D])[.)\]]?\s*\.?",candidate)
    if m:
        return m[1].upper()
    value = re.fullmatch(r"\s*([+-]?\d+)(?:\.0+)?\s*(?:units?(?:\s+per\s+min(?:ute)?)?)?\.?\s*",candidate)
    if value and str(int(value[1])) in options:
        return "ABCD"[options.index(str(int(value[1])))]
    return None


def formatted(raw):
    pattern=r"(?i)\b(Evidence|Rule|Application|Final\s+answer)\s*\**\s*:"
    ms=list(re.finditer(pattern,raw))
    labels=[" ".join(m[1].lower().split()) for m in ms]
    return (labels==["evidence","rule","application","final answer"]
            and all(raw[a.end():b.start()].strip(" *\r\n\t") for a,b in zip(ms,ms[1:]))
            and bool(re.fullmatch(r"(?i)\s*\**final\s+answer\s*\**\s*:\s*\**[^*\s].*",raw.strip().splitlines()[-1])))


def assess(ref, pred, cards, reviews):
    raw=pred["raw"]
    letter=final(raw,ref["options"])
    key=hashlib.sha256((ref["id"]+"\n"+pred["finish_reason"]+"\n"+raw).encode()).hexdigest()[:24]
    answer=False if pred["finish_reason"]=="length" else None if letter is None else letter==ref["gold"]
    joint=False if answer is False else None
    if len(raw.split())<=5 and letter is not None and ("\n" not in raw.strip() or len(raw.split())<4):
        joint=False
    cards[key] = dict(blind_id=key,id=ref["id"],source=ref,finish=pred["finish_reason"],raw=raw)
    stage = {}
    if key in reviews:
        r=reviews[key]
        assert r["id"]==ref["id"] and r["reason"].strip()
        assert all(r[k] is None or type(r[k]) is bool for k in ("E","R","A","answer","consistent"))
        if answer is not None:
            assert r["answer"] is answer, (key,"Contradiction with unambiguous final/length")
        answer=r["answer"]
        stage={k:r[k] for k in ("E","R","A","consistent")}
        terms=[answer,*stage.values()]
        joint=False if False in terms else True if all(v is True for v in terms) else None
    return dict(id=ref["id"],pair=ref["pair"],task=ref["task"],key=key,answer=answer,joint=joint,
                format=bool(formatted(raw)),choice=letter,reviewed=key in reviews,**stage)


def tally(values):
    values=list(values)
    return dict(pass_count=values.count(True),fail_count=values.count(False),unknown=values.count(None),n=len(values))


def both(rows,field):
    return False if any(r[field] is False for r in rows) else True if all(r[field] is True for r in rows) else None


def compare(left,right,field="joint"):
    gains=losses=low=high=0
    for l,r in zip(left,right):
        assert l["id"]==r["id"]
        a,b=l[field],r[field]
        gains+=int(a is False and b is True)
        losses+=int(a is True and b is False)
        if l["key"]!=r["key"]:
            low+=int(b is True)-int(a is not False)
            high+=int(b is not False)-int(a is True)
    return dict(n=len(left),gains=gains,losses=losses,delta_bounds=[low,high],
                coverage=sum(l[field] is not None and r[field] is not None for l,r in zip(left,right)))


def pair_rows(rows, field):
    groups = {}
    for r in rows:
        groups.setdefault(r["pair"],[]).append(r)
    assert len(groups)==24 and all(len(v)==2 for v in groups.values())
    return [dict(id=g,joint=both(members,field),key=tuple(r["key"] for r in members)) for g,members in groups.items()]


def paired_stats(a,b):
    out={}
    for metric in ("answer","joint"):
        out[metric]=compare(a,b,metric)
        left,right=pair_rows(a,metric),pair_rows(b,metric)
        diff=compare(left,right)
        discord=diff["gains"]+diff["losses"]
        if diff["coverage"]==len(left):
            pval=min(1,2*sum(math.comb(discord,k) for k in range(min(diff["gains"],diff["losses"])+1))/(2**discord))
            diff["exact_pair_mcnemar_p"]=pval
        out[metric+"_both_variants"]=diff
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arms",nargs="+",choices=["control","paired","base"],default=["control","paired"])
    ap.add_argument("--runs",type=Path,default=HERE/"runs")
    ap.add_argument("--reviews",type=Path,default=HERE/"review_labels.jsonl")
    ap.add_argument("--output",type=Path,default=HERE/"analysis")
    args=ap.parse_args()
    spec=HERE/"spec"
    manifest=read(spec/"manifest.json")
    assert digest(spec/"gold.json")==manifest["gold_sha256"] and digest(spec/"inputs.json")==manifest["inputs_sha256"]
    gold=read(spec/"gold.json")
    blind={r["id"]:r for r in read(spec/"inputs.json")}
    assert len(gold)==len(blind)==48 and all(digest(spec/r["image"])==r["image_sha256"] for r in gold)
    reviews=lines(args.reviews) if args.reviews.exists() else []
    review_by_key={r["blind_id"]:r for r in reviews}
    assert len(reviews)==len(review_by_key),"Duplicate review"
    args.output.mkdir(parents=True,exist_ok=True)
    allrows={};cards={};summaries={};provenance={};prompts={}
    for name in args.arms:
        folder=args.runs/name
        file=folder/"predictions.jsonl"
        assert read(folder/"done.json")["output_sha256"]==digest(file)
        provenance[name]=digest(file)
        pred=lines(file)
        assert len(pred)==48 and [r["id"] for r in pred]==list(blind)
        pred={r["id"]:r for r in pred}
        rows=[]
        for ref in gold:
            r=pred[ref["id"]]
            assert ref["image_sha256"]==r["image_sha256"] and r["finish_reason"] in ("stop","length")
            assert prompts.setdefault(ref["id"],r["prompt"])==r["prompt"]
            rows.append(assess(ref,r,cards,review_by_key))
        allrows[name]=rows
        summaries[name]={field:tally(r[field] for r in rows) for field in ("answer","joint","format")}
        for field in ("answer","joint"):
            summaries[name][field+"_both_variants"]=tally(r["joint"] for r in pair_rows(rows,field))
        summaries[name]["by_task"]={task:{k:tally(r[k] for r in rows if r["task"]==task) for k in ("answer","joint")} for task in ("panel","series")}
    queue=[cards[key] for key in sorted(cards)]
    save_rows(args.output/"blind_cards.jsonl",queue)
    save_rows(args.output/"scores.jsonl",[dict(arm=name,**r) for name,rr in allrows.items() for r in rr])
    diffs={a+"_to_"+b:paired_stats(allrows[a],allrows[b]) for a,b in (("control","paired"),("base","paired"),("base","control")) if a in allrows and b in allrows}
    manifest=dict(arms=summaries,paired=diffs,unique_blinded_outputs=len(queue),frozen_gold=manifest["gold_sha256"],
                  reviews_recognized=len(set(cards)&set(review_by_key)),scorer_sha256=digest(Path(__file__)),predictions=provenance)
    save(args.output/"summary.json",manifest)
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=="__main__":
    main()
