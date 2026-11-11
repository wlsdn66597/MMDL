"""Freeze anonymous review packets, then aggregate original item/scene IDs without model calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ARMS=("base","ordinary","preserved")
DATASETS=("charts","science")
GOLD={"charts":"0171459c7599cfadb788f8c3950db89b84ff80f7c16058e9126dbd3135fe3cda",
      "science":"cf24d70f750bbde044fc6eff074a4135832175488da2c656bf32db6093755edb"}
INPUTS={"charts":"f1be9e728a836bfecf784451c5b4dbcd5092d4b9519498ece1b9425d7ed16233",
        "science":"4538cf9358debc6eb46d01c3ce63a060784ecb4ce1377748fdd615309394c216"}
FIELDS=("E","R","A","consistent","answer","first_E_correct","image_corrected")


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, payload):
    temp=path.with_suffix(".tmp")
    temp.write_text(payload,encoding="utf8")
    temp.replace(path)


def jsonout(path,payload):
    dump(path,json.dumps(payload,indent=2,ensure_ascii=False)+"\n")


def format_ok(raw, finish):
    if finish!="stop" or not raw.strip():
        return False
    headings=list(re.finditer(r"(?im)^\s*(?:\*\*)?(Evidence|Rule|Application|Final answer)(?:\*\*)?\s*:",raw))
    return (list(m[1].lower() for m in headings)==["evidence","rule","application","final answer"] and
            all(raw[a.end():b.start()].strip(" \t\r\n*") for a,b in zip(headings,headings[1:])) and
            bool(re.fullmatch(r"(?i)(?:\*\*)?Final answer(?:\*\*)?\s*:\s*\S[^\r\n]*",raw.strip().splitlines()[-1].strip())))


def final_letter(raw, options):
    text=raw.strip().replace("*","").replace("`","")
    if not text:
        return None
    tail=text.splitlines()[-1].strip()
    value=re.sub(r"(?i)^(?:final\s+answer|answer)\s*:\s*","",tail).strip()
    value=re.sub(r"^\$?\\(?:boxed|text)\{([^{}]+)\}\$?\.?$",r"\1",value)
    m=re.fullmatch(r"(?i)(?:option\s+|choice\s+)?\(?([A-H])\)?\.?",value)
    if m and ord(m[1].upper())-65<len(options):
        return m[1].upper()
    value=value.strip("$. ").replace("−","-")
    matches=[i for i,x in enumerate(options) if value.casefold()==x.casefold()]
    return chr(65+matches[0]) if len(matches)==1 else None


def cohort(dataset):
    spec=HERE/"spec"/dataset
    assert sha(spec/"inputs.json")==INPUTS[dataset] and sha(spec/"gold.json")==GOLD[dataset]
    inp=load(spec/"inputs.json")
    gold={r["id"]:r for r in load(spec/"gold.json")}
    lookup={}
    if dataset=="science":
        for line in (spec/"visual_review.md").read_text(encoding="utf8").splitlines():
            cells=[s.strip() for s in line.strip("|").split("|")]
            if len(cells)==3 and cells[0].isdigit():
                lookup["sqa-val-"+cells[0]]=" · ".join(cells[1:])
        assert lookup.keys()==gold.keys()
    for row in inp:
        rid=row["id"]; assert sha(spec/row["image"])==row["image_sha256"]
        ref=gold[rid]
        if dataset=="science":
            options=re.findall(r"(?m)^[A-H]\. (.+)$",row["prompt"])
            assert ref["option_text"]==options[ord(ref["option_letter"])-65]
            ref=dict(ref,group="science",gold=ref["option_letter"],options=options,visual_witness=lookup[rid])
        else:
            assert ref["value"]==row["options"][ord(ref["answer"])-65]
            ref=dict(ref,group=ref["domain"],gold=ref["answer"],options=row["options"])
        yield row,ref


def read_runs(dataset):
    cases=list(cohort(dataset));ids=[r["id"] for r,_ in cases];prompts={};manifests={}
    for arm in ARMS:
        folder=HERE/"runs"/f"{dataset}-{arm}"
        pred=folder/"predictions.jsonl"
        done=load(folder/"done.json")
        assert done["n"]==len(ids) and done["output_sha256"]==sha(pred) and done["manifest_sha256"]==sha(folder/"manifest.json")
        manifest=load(folder/"manifest.json")
        assert manifest["inputs_sha256"]==INPUTS[dataset] and manifest["dataset"]==dataset and manifest["arm"]==arm
        assert manifest["max_output"]==2048 and manifest["request_seed"]==3407 and manifest["batch"]==6
        identity=[manifest[k] for k in ("weights","config_sha256","engine_sha256","runner_sha256")]
        if manifests:
            assert all(identity[0]!=v[0] and identity[1:]==v[1:] for v in manifests.values())
        manifests[arm]=identity
        records=lines(pred)
        assert [r["id"] for r in records]==ids
        for r,(input_row,ref) in zip(records,cases):
            assert r["image_sha256"]==input_row["image_sha256"] and r["finish_reason"] in ("stop","length")
            assert r["seed"]==3407 and r["max_output"]==2048 and r["input_tokens"]>0
            assert prompts.setdefault(r["id"],(r["prompt"],r["input_tokens"]))==(r["prompt"],r["input_tokens"])
            provenance={"arm":arm,"dataset":dataset,"id":r["id"],"format":format_ok(r["raw"],r["finish_reason"])}
            key=hashlib.sha256(json.dumps([dataset,r["id"],r["finish_reason"],r["raw"]],ensure_ascii=False).encode()).hexdigest()[:20]
            card=dict(key=key,id=r["id"],domain=ref["group"],image=str(Path("spec")/dataset/input_row["image"]),
                      prompt=r["prompt"],finish=r["finish_reason"],reference=ref,raw=r["raw"])
            yield provenance,card


def both(flags):
    return False if any(v is False for v in flags) else None if any(v is None for v in flags) else True


def bounds(flags):
    return [sum(v is True for v in flags),sum(v is not False for v in flags)]


def tally(flags):
    flags=list(flags)
    return dict(yes=flags.count(True),no=flags.count(False),unknown=flags.count(None),n=len(flags),bounds=bounds(flags))


def grade(row,card,label):
    for field in FIELDS:
        assert field in label and (type(label[field]) is bool or label[field] is None),(card["key"],field)
    assert label["id"]==card["id"] and label["reason"].strip()
    letter=final_letter(card["raw"],card["reference"]["options"])
    auto=False if card["finish"]=="length" else None if letter is None else letter==card["reference"]["gold"]
    if auto is not None:
        assert label["answer"] is auto, (card["id"],card["key"],"clear terminal answer conflicts",auto)
    if label["first_E_correct"] is not False:
        assert label["image_corrected"] is not True
    if label["image_corrected"] is True:
        assert label["answer"] is True
    ref=card["reference"]
    return dict(**row,key=card["key"],**{f:label[f] for f in FIELDS},joint=both([label[f] for f in ("E","R","A","consistent","answer")]),
                domain=ref["group"],scene=ref.get("scene"),variant=ref.get("variant"),
                rule_index=ref.get("rule_index"),order=ref.get("order"))


def scene_stats(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["scene"]].append(r)
    assert len(groups)==12 and all(len(v)==8 for v in groups.values())
    scenes={s:both([row["joint"] for row in group]) for s,group in groups.items()}
    pairs={}
    for factor in ("variant","rule_index","order"):
        dims=["scene","variant","rule_index","order"]
        units=defaultdict(list)
        for row in rows:
            units[tuple(row[d] for d in dims if d!=factor)].append(row)
        assert len(units)==48 and all(len(g)==2 for g in units.values())
        pairs[factor]={f:tally(both([r[f] for r in g]) for g in units.values()) for f in ("joint","answer")}
    return scenes,pairs


def summarize(rows):
    metrics={f:tally(r[f] for r in rows) for f in ("answer","joint","format","E","R","A","consistent")}
    errors=[r for r in rows if r["first_E_correct"] is False]
    metrics["initial_E"]=dict(wrong=len(errors),unknown=sum(r["first_E_correct"] is None for r in rows),
                               corrected=tally(r["image_corrected"] for r in errors))
    return metrics


def outcome_comparison(base,other,field):
    assert len(base)==len(other)
    transitions=Counter()
    for left,right in zip(base,other):
        assert left["id"]==right["id"]
        a,b=left[field],right[field]
        key=("unknown" if None in (a,b) else "retained" if a and b else "gained" if b else "lost" if a else "both_wrong")
        transitions[key]+=1
    return {k:transitions[k] for k in ("retained","gained","lost","both_wrong","unknown")}


def summary(allrows):
    arms={};scene={};byid={}
    for arm in ARMS:
        armrows=[r for r in allrows if r["arm"]==arm];assert len(armrows)==115
        per={}
        for dataset in DATASETS:
            records=[r for r in armrows if r["dataset"]==dataset]
            byid[arm,dataset]=sorted(records,key=lambda r:r["id"])
            per[dataset]=summarize(records)
            if dataset=="charts":
                scene[arm],per[dataset]["factor_pairs"]=scene_stats(records)
                per[dataset]["scenes"]=tally(scene[arm].values())
                per[dataset]["domain_joint"]={d:tally(r["joint"] for r in records if r["domain"]==d) for d in sorted({r["domain"] for r in records})}
        arms[arm]=dict(**per,strict_era_total=sum(r["format"] for r in armrows))
    changes={}
    for left,right in (("base","ordinary"),("base","preserved"),("ordinary","preserved")):
        changes[left+"_to_"+right]={d:{metric:outcome_comparison(byid[left,d],byid[right,d],metric) for metric in ("answer","joint")} for d in DATASETS}
    c,p=scene["ordinary"],scene["preserved"]
    recovered=[s for s in c if c[s] is False and p[s] is True]
    lost=[s for s in c if c[s] is True and p[s] is False]
    domain={r["scene"]:r["domain"] for r in allrows if r["scene"]}
    b=byid["base","science"]
    ret={}
    for arm in ("ordinary","preserved"):
        ret[arm]=bounds([both([left["answer"],right["answer"]]) for left,right in zip(b,byid[arm,"science"])])
    tests=dict(natural_total=arms["preserved"]["science"]["answer"]["bounds"][0]>=arms["base"]["science"]["answer"]["bounds"][1],
               retained_original=ret["preserved"][0]>ret["ordinary"][1],
               joint_8_scene_increase=bounds(p.values())[0]>bounds(c.values())[1],
               multi_scene_and_domain=len(recovered)>=2 and len({domain[s] for s in recovered})>=2,
               preserved_90pct_strict_format=arms["preserved"]["strict_era_total"]>=104)
    return dict(arms=arms,paired=changes,scene_labels=scene,scene_recovered=recovered,scene_lost=lost,
                science_retained_bounds=ret,tests=tests,gate=all(tests.values()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["packet","score"])
    parser.add_argument("--labels",nargs="*",type=Path,default=[])
    args=parser.parse_args()
    target=HERE/"review";target.mkdir(exist_ok=True)
    records=[];cards={}
    for dataset in DATASETS:
        for row,card in read_runs(dataset):
            assert cards.setdefault(card["key"],card)==card
            records.append((row,card["key"]))
    packets=defaultdict(list)
    for k in sorted(cards):
        c=cards[k]
        packets[c["domain"]].append(c)
    if args.mode=="packet":
        proof={}
        for domain,packet in packets.items():
            dst=target/(domain+".jsonl")
            data="".join(json.dumps(c,ensure_ascii=False)+"\n" for c in packet)
            if dst.exists():assert dst.read_text(encoding="utf8")==data
            else:dump(dst,data)
            proof[domain]=dict(n=len(packet),sha256=sha(dst))
        jsonout(target/"packet.json",dict(inputs=INPUTS,gold=GOLD,domains=proof,n=len(records),unique=len(cards)))
        return
    labels={}
    for file in args.labels:
        for label in lines(file):
            key=label["key"]
            assert key not in labels
            labels[key]=label
    assert labels.keys()==cards.keys(),(len(labels),len(cards),sorted(cards.keys()-labels.keys())[:8])
    allrows=[grade(r,cards[key],labels[key]) for r,key in records]
    dump(target/"scored_items.jsonl","".join(json.dumps(r,ensure_ascii=False)+"\n" for r in allrows))
    results=summary(allrows)
    results["n_responses"]=len(records);results["unique_reviews"]=len(labels)
    results["grade_proof"]={str(p.name):sha(p) for p in args.labels}
    results["scorer_sha256"]=sha(Path(__file__))
    results["inputs_sha256"]=INPUTS;results["gold_sha256"]=GOLD
    jsonout(target/"summary.json",results)
    print(json.dumps(dict(gate=results["gate"],tests=results["tests"])),flush=True)


if __name__=="__main__":
    main()
