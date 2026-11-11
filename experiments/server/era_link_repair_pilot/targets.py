"""Derive paired supervision exclusively from train labels and the starting model's train rollouts."""
from collections import Counter
from decimal import Decimal
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf8"))
def write(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fmt(x):return format(Decimal(x).normalize(),"f")
def section(text,name):
    m=re.search(r"(?ims)^\s*(?:\*\*)?"+name+r"(?:\*\*)?\s*:(?:\*\*)?\s*(.*?)(?=^\s*(?:\*\*)?(?:Evidence|Rule|Application|Final answer)(?:\*\*)?\s*:|\Z)",text)
    return m[1].strip() if m else ""
def nums(text):
    text=re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))","",text)
    return [Decimal(s) for s in re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?",text)]
def first_set(txt,vs):
    """Only admit an unambiguous first selection, never infer the draft from the final option."""
    line=txt.splitlines()[0] if txt else ""
    line=re.split(r"(?i)\.\s|;\s*(?:their sum|these add|summing)",line)[0]
    values={v for v in nums(line) if v in vs.values()}
    names={v for n,v in vs.items() if re.search(r"\b"+re.escape(n)+r"\b",line,re.I)}
    if len(values)==3 and (not names or names==values):return sorted(values)
    if len(names)==3 and values.issubset(names):return sorted(names)
    return None
def vs_text(v):return ", ".join(map(fmt,v))
def make(it,initial):
    vs={n:Decimal(v) for n,v in it["rows"].items()};label={v:n for n,v in vs.items()}
    if it["mode"]=="direct":
        n=it["direct_name"];val=it["truth"]
        e=f"Direct read. The {n} bar is labeled {val}."
        return e,e,"Read the value printed beside the named bar.",f"The requested value is {val}, option {it['letter']}.","direct"
    picked=set(initial); assert len(picked)==3 and picked<=set(vs.values())
    goal=set(map(Decimal,it["truth_vals"]));outcome="keep" if picked==goal else "replace"
    intro="Check. My initial candidates are "+vs_text(sorted(picked))+"."
    es,eq=[intro],[intro]
    for step in range(4):
        ceiling=max(picked); challenger=min(set(vs.values())-picked)
        es.append(f"Compare the smallest excluded bar with {fmt(ceiling)}.")
        eq.append(f"Is any excluded bar smaller than {fmt(ceiling)}?")
        if challenger<ceiling:
            obs=f"The lowest excluded bar is {label[challenger]} at {fmt(challenger)}, below {label[ceiling]} at {fmt(ceiling)}. Replace {fmt(ceiling)} with {fmt(challenger)}."
            picked.remove(ceiling);picked.add(challenger)
        else:
            obs=f"The lowest excluded bar is {label[challenger]} at {fmt(challenger)}, above {label[ceiling]} at {fmt(ceiling)}. Keep the selection."
        es.append(obs);eq.append(obs)
        if challenger>=ceiling:break
    assert picked==goal
    end="Verified values: "+vs_text(sorted(picked))+"."
    es.append(end);eq.append(end)
    rule="Add the three smallest chart values after verifying the selection."
    app=" + ".join(map(fmt,sorted(picked)))+f" = {it['truth']}, option {it['letter']}."
    return " ".join(es)," ".join(eq),rule,app,outcome
def main():
    items=read(ROOT/"train/gold.json");rows=list(map(json.loads,(ROOT/"eval/collect/predictions.jsonl").read_text().splitlines()))
    pred={r["id"]:r for r in rows};assert len(rows)==len(pred)==72
    targets=[]; audit=[];parts=Counter()
    from random import Random
    for it in items:
        rid=it["id"];vs={n:Decimal(v) for n,v in it["rows"].items()};draft=None;raw_e=None
        if it["mode"]=="rank":
            r=pred[rid];assert r["image_sha256"]==it["image_sha256"]
            raw_e=section(r["raw"],"Evidence");draft=first_set(raw_e,vs)
            if draft is None:
                audit.append(dict(id=rid,skipped=True,original_E=raw_e));continue
        if it["mode"]=="draft":draft=list(map(Decimal,it["draft"]))
        a,b,rule,app,action=make(it,draft)
        parts[it["mode"]+"/"+action]+=1
        prompt="Question: "+it["question"]+"\nOptions:\n"+"".join(f"{chr(65+i)}. {v}\n" for i,v in enumerate(it["options"]))+'Please select the correct answer from the options above.\nOn the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'
        def tgt(e):return f"Evidence: {e}\nRule: {rule}\nApplication: {app}\nFinal answer: {it['letter']}"
        targets.append(dict(sample_id=rid,input_prompt=prompt,image="train/"+it["image"],image_sha256=it["image_sha256"],
            statement=tgt(a),question=tgt(b),action=action,mode=it["mode"],draft=list(map(fmt,draft)) if draft else None))
        audit.append(dict(id=rid,skipped=False,original_E=raw_e,action=action,source="observed_first_E" if raw_e else it["mode"]))
    assert parts["draft/replace"]==parts["draft/keep"]==parts["direct/direct"]==24 and parts["rank/keep"]+parts["rank/replace"]>=36
    Random(536).shuffle(targets)
    f=ROOT/"paired_train.jsonl";f.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in targets),encoding="utf8")
    for arm in ("statement","question"):
        (ROOT/(arm+".jsonl")).write_text("".join(json.dumps(dict(id=x["sample_id"],image=x["image"],input_prompt=x["input_prompt"],target=x[arm]),ensure_ascii=False)+"\n" for x in targets),encoding="utf8")
    write(ROOT/"draft_audit.json",audit);write(ROOT/"target_receipt.json",dict(included=len(targets),counts=dict(parts),
        omitted=len(items)-len(targets),paired_sha256=sha(f),rollout_sha256=sha(ROOT/"eval/collect/predictions.jsonl")))
if __name__=="__main__":main()
