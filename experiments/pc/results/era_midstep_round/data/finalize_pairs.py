"""Emit ordinary and intermediate-preserving ERA examples after all source/teacher reviews."""
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil

from inspect_teacher import EXCLUSIONS, decision, numset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CAP=48


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def file_sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, data):
    payload=(json.dumps(data,ensure_ascii=False,indent=2)+"\n").encode()
    if path.exists() and path.read_bytes()!=payload:
        raise ValueError(f"Frozen file would change: {path}")
    path.write_bytes(payload)


def fmt(n):
    return format(Decimal(n).normalize(),"f")


def last_answer(text):
    choices=re.findall(r'(?im)^\s*(?:\*\*)?final\s+answer(?:\*\*)?\s*:\s*[* (]*(?:\\boxed\{)?([A-E])\b',text)
    return choices[-1] if choices else None


def seg(raw):
    # Preserve an exact passage from the one generated response; remove only whitespace.
    # Keep full reasoning if option discussions are interspersed.
    anchor=re.search(r"(?im)^\s*(?:Looking at|Checking|Comparing|Given) (?:the |our )?(?:answer )?(?:options|choices)\b",raw)
    if anchor and anchor.start()>120:
        return raw[:anchor.start()].strip()
    return raw.strip()


def term_evidence(row):
    terms=row["terms"]
    if row["schema"]=="read_label":
        label,value=next(iter(terms.items()))
        return f'The printed reading associated with "{label}" is {value}.'
    if row["schema"] in ("sum_then_subtract","mean_then_subtract"):
        return ", ".join(f'"{k}" reads {v}' for k,v in terms.items())+"."
    return row["evidence"]


def quantitative_targets(row,proof):
    kind=row["schema"]; option=row["answer"]; y=row["answer_text"]
    terms=row["terms"]
    evidence,rule=term_evidence(row),row["rule"]
    if kind=="read_label":
        name=next(iter(terms))
        control=f'Reading "{name}" gives {y}, which corresponds to option {option}.'
        return evidence,rule,control,control,[]
    if kind in ("sum_then_subtract","mean_then_subtract"):
        (an,a),(bn,b),(cn,c)=terms.items()
        mid=row["intermediate"][0]
        if kind=="sum_then_subtract":
            control=f'The required total from "{an}" and "{bn}", less the "{cn}" reading, is ({a} + {b}) - {c} = {y}, option {option}.'
            expanded=f'Adding "{an}" and "{bn}" gives {a} + {b} = {mid}. Subtracting "{cn}" gives {mid} - {c} = {y}, option {option}.'
        else:
            control=f'The equally weighted mean for "{an}" and "{bn}", less the "{cn}" reading, is ({a} + {b}) / 2 - {c} = {y}, option {option}.'
            expanded=f'The mean for "{an}" and "{bn}" is ({a} + {b}) / 2 = {mid}. Subtracting "{cn}" gives {mid} - {c} = {y}, option {option}.'
        return evidence,rule,control,expanded,[mid]
    f1,f2,t1,t2,m,v=[Decimal(terms[k]) for k in ("first_force","second_force","first_end","second_end","mass","initial_speed")]
    if proof.get("route")=="acceleration":
        a1,a2=f1/m,f2/m
        rule="Divide each signed force by mass to find its acceleration. Accumulate the two velocity changes over their respective durations."
        control=f'The final velocity is {v} + ({f1}/{m}) × {t1} + ({f2}/{m}) × ({t2} - {t1}) = {y} m/s, option {option}.'
        expanded=f'The accelerations are {f1}/{m} = {fmt(a1)} and {f2}/{m} = {fmt(a2)} m/s². The final velocity is {v} + ({fmt(a1)}) × {t1} + ({fmt(a2)}) × ({t2} - {t1}) = {y} m/s, option {option}.'
        return evidence,rule,control,expanded,[fmt(a1),fmt(a2)]
    j1,j2=f1*t1,f2*(t2-t1)
    rule="Each plateau supplies signed force times duration of impulse. Adding their contributions divided by mass to the initial velocity gives the final velocity."
    control=f'Using both signed force intervals, v = {v} + [({f1}) × {t1} + ({f2}) × ({t2} - {t1})] / {m} = {y} m/s, option {option}.'
    expanded=f'The signed impulses are ({f1}) × {t1} = {j1} N·s and ({f2}) × ({t2} - {t1}) = {j2} N·s. Thus v = {v} + [({j1}) + ({j2})] / {m} = {y} m/s, option {option}.'
    if proof.get("preserved_application"):
        expanded=proof["preserved_application"].strip()+" This matches option "+option+"."
    return evidence,rule,control,expanded,[fmt(j1),fmt(j2)]


def make_era(e,r,a,final):
    fields=[s.strip() for s in (e,r,a,final)]
    assert all(fields) and all("\n" not in s and "\r" not in s for s in fields)
    return f"Evidence: {fields[0]}\nRule: {fields[1]}\nApplication: {fields[2]}\nFinal answer: {fields[3]}"


def select(eligible):
    by=defaultdict(list)
    for row in eligible:
        by[row["schema"]].append(row)
    for group in by.values():
        group.sort(key=lambda x:hashlib.sha256(("r5-vetted-selection-v1/"+x["id"]).encode()).digest())
    keys=sorted(by)
    selected=[]
    while len(selected)<CAP and any(by.values()):
        for k in keys:
            if by[k] and len(selected)<CAP:
                selected.append(by[k].pop(0))
    return selected


def main():
    frozen=read(HERE/"teacher_receipt.json")
    assert frozen["input_sha256"]==file_sha(HERE/"teacher_inputs.json")
    assert frozen["gold_sha256"]==file_sha(HERE/"teacher_gold.json")
    inputs={r["id"]:r for r in read(HERE/"teacher_inputs.json")}
    gold={r["id"]:r for r in read(HERE/"teacher_gold.json")}
    pred=HERE/"teacher_predictions.jsonl"
    proof=read(HERE/"teacher_done.json")
    assert file_sha(pred)==proof["output_sha256"]
    raws={r["id"]:r for r in (json.loads(s) for s in pred.read_text(encoding="utf8").splitlines())}
    checks={}
    for name in ("review_native","review_quant_trainer","review_quant_data"):
        chunk=read(HERE/(name+".json"))
        assert not set(chunk)&set(checks), name
        checks.update(chunk)
    assert len(checks)==len(raws)==192 and set(inputs)==set(gold)==set(raws)==set(checks)
    verdicts=[]
    eligible=[]
    for sid, r in raws.items():
        x=gold[sid];inp=inputs[sid];v=checks[sid];raw=r["raw"]
        assert file_sha(HERE/inp["image"])==inp["image_sha256"]
        reason=v["reason"]
        ok=bool(v["accept"])
        if sid in EXCLUSIONS:
            ok=False;reason="source_image_disagreement: "+EXCLUSIONS[sid]
        if ok:
            assert last_answer(raw)==x["answer"] and r["finish_reason"]=="stop",(sid,x['answer'],last_answer(raw))
            quote=v["teacher_quote"]
            assert quote and quote in raw,(sid,"Invalid source quote")
            if x["schema"]=="science_native":
                e,rule,ac=v["evidence"],v["rule"],v["application"]
                enriched=ac;expected=[]
            else:
                e,rule,ac,enriched,expected=quantitative_targets(x,v)
            assert all(Decimal(t) in numset(quote) for t in expected),(sid,expected)
            teacher_only="Show the image readings and reasoning needed to justify your answer.\n"
            assert inp["prompt"].count(teacher_only)==1
            training_prompt=inp["prompt"].replace(teacher_only,"")
            record=dict(id=sid,prompt=training_prompt,image=inp["image"],image_sha256=inp["image_sha256"],
                        control_response=make_era(e,rule,ac,x["answer"]),midstep_response=make_era(e,rule,enriched,x["answer"]),
                        stratum=x["stratum"],schema=x["schema"],source=x["source"],source_group=x["group"],review_id=sid,
                        teacher_raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),teacher_quote=quote,
                        teacher_quote_start=raw.index(quote),teacher_quote_end=raw.index(quote)+len(quote),
                        teacher_supported_intermediate=expected)
            eligible.append(record)
        verdicts.append(dict(id=sid,stratum=x["stratum"],schema=x["schema"],source=x["source"],
                             correct_letter=last_answer(raw)==x["answer"],finish_reason=r["finish_reason"],
                             verified=ok,reason=reason,reviewer=v.get("reviewer"),teacher_raw_sha256=hashlib.sha256(raw.encode()).hexdigest()))
    selected=[]
    for stratum in ("direct","multi"):
        selected.extend(select([x for x in eligible if x["stratum"]==stratum]))
    chosen={r["id"] for r in selected}; selected.sort(key=lambda x:x["id"])
    for row in verdicts:row["selected"]=row["id"] in chosen
    save(HERE/"teacher_verdicts.json",verdicts)
    target=HERE/"train_pairs.jsonl"
    payload="".join(json.dumps(x,ensure_ascii=False)+"\n" for x in selected).encode()
    if target.exists():assert target.read_bytes()==payload
    target.write_bytes(payload)
    receipt=dict(verified=True,n=len(selected),train_sha256=file_sha(target),teacher_prediction_sha256=file_sha(pred),
                 teacher_input_sha256=frozen["input_sha256"],teacher_gold_sha256=frozen["gold_sha256"],verdict_sha256=file_sha(HERE/"teacher_verdicts.json"),
                 n_teacher_calls=192,n_reviewed=len(verdicts),n_eligible=len(eligible),all_correct_letters=sum(v["correct_letter"] for v in verdicts),
                 eligible_by_schema=dict(Counter(r["schema"] for r in eligible)),selected_by_schema=dict(Counter(r["schema"] for r in selected)),
                 selected_by_stratum=dict(Counter(r["stratum"] for r in selected)),identical_targets=sum(r["control_response"]==r["midstep_response"] for r in selected),
                 rule="Review scope as rubric. Each stratum targets 48; schema round-robin with SHA256('r5-vetted-selection-v1/' + id) order within each.",
                 teacher_witness="Candidate A is an editorial paraphrase with source-checked intermediate values, not a claim of copying the exact sentence. Supporting original quote stored with offsets.")
    save(HERE/"train_receipt.json",receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=="__main__":
    main()
