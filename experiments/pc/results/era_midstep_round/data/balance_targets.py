"""Pre-model length matching with equivalent, verified ERA renderings of frozen V1 examples."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import median

from transformers import AutoTokenizer

from finalize_pairs import HERE, read, file_sha

V1="a53132d8d007754ab9a36ef2a1b0f0ffa981533491c55abb6e28c03111ddb24c"
BASE=Path.home()/"mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"


def rebuild(full,a):
    before,_,tail=full.partition("\nApplication: ")
    body,split,end=tail.rpartition("\nFinal answer: ")
    assert before and body and split and end and "\n" not in a
    return before+"\nApplication: "+a+"\nFinal answer: "+end


def render(row,reference):
    terms=reference["terms"]; y=reference["answer_text"]; opt=reference["answer"]
    v=reference["intermediate"]; label=lambda t: f'"{t}"'
    if reference["schema"]=="impulse":
        f1=terms["first_force"]; f2=terms["second_force"];t1=terms["first_end"];t2=terms["second_end"]
        m=terms["mass"];v0=terms["initial_speed"];j1,j2=v[:2]
        expr=f"{v0} + [({f1}) × {t1} + ({f2}) × ({t2} − {t1})] / {m}"
        std=[
            row["control_response"].split("\nApplication: ")[1].split("\nFinal answer:")[0],
            f"The initial velocity plus the impulse per unit mass is {expr} = {y} m/s, option {opt}.",
            f"The first plateau runs from 0 to {t1} s, then the second ends at {t2} s. Their signed areas divided by mass give v = {expr} = {y} m/s, option {opt}.",
            f"Using force times elapsed time for both plateaus and adding the velocity change to the initial velocity yields v = {expr} = {y} m/s, option {opt}.",
        ]
        mid=[
            f"The impulses are {f1}×{t1}={j1} and {f2}×({t2}−{t1})={j2} N·s. Then v={v0}+[{j1}+({j2})]/{m}={y} m/s, option {opt}.",
            f"The first impulse is {f1}×{t1}={j1} N·s; the second is {f2}×({t2}−{t1})={j2} N·s. Thus v={v0}+[{j1}+({j2})]/{m}={y} m/s, option {opt}.",
            f"The signed impulses are {f1} × {t1} = {j1} and {f2} × ({t2} − {t1}) = {j2} N·s. The final velocity is {v0} + [{j1} + ({j2})] / {m} = {y} m/s, option {opt}.",
        ]
    else:
        (an,a),(bn,b),(cn,c)=terms.items()
        op="/" if reference["schema"]=="mean_then_subtract" else ""
        expr=f"({a} + {b})"+(" / 2" if op else "")
        rule="arithmetic mean" if op else "sum"
        std=[
            row["control_response"].split("\nApplication: ")[1].split("\nFinal answer:")[0],
            f"The {rule} of the first two readings less {label(cn)} is {expr} − {c} = {y}, option {opt}.",
            f"For {label(an)} and {label(bn)} take their {rule}; subtract the {label(cn)} reading: {expr} − {c} = {y}, option {opt}.",
            f"Using the readings in the requested order gives {expr} − {c} = {y}, option {opt}.",
        ]
        s=v[0]
        mid=[
            f"The {rule} is {expr} = {s}. Subtract {c}: {s} − {c} = {y}, option {opt}.",
            f"The {rule} of {label(an)} and {label(bn)} is {expr} = {s}. Subtracting {label(cn)} gives {s} − {c} = {y}, option {opt}.",
            f"{expr} = {s}; {s} − {c} = {y}, option {opt}.",
            f"The first two readings give {expr} = {s}; subtracting the third leaves {s} − {c} = {y}, option {opt}.",
        ]
    return ([rebuild(row["control_response"],s) for s in std],[rebuild(row["midstep_response"],p) for p in mid])


def main():
    folder=HERE/"target_v1"
    folder.mkdir(exist_ok=True)
    for name in ("train_pairs.jsonl","train_receipt.json"):
        target=folder/name
        if not target.exists():target.write_bytes((HERE/name).read_bytes())
    assert file_sha(folder/"train_pairs.jsonl")==V1
    candidates=[json.loads(s) for s in (folder/"train_pairs.jsonl").read_text(encoding="utf8").splitlines()]
    gold={r["id"]:r for r in read(HERE/"teacher_gold.json")}
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    counts=lambda s:len(tok.encode(s,add_special_tokens=False))+1
    rows=[]
    for row in candidates:
        if row["stratum"]=="direct":
            assert row["control_response"]==row["midstep_response"]
            continue
        std,mid=render(row,gold[row["id"]])
        options=[]
        target_delta=1 if int(hashlib.sha256(row["id"].encode()).hexdigest(),16)%2 else -1
        current=counts(row["control_response"])
        for i,s in enumerate(std):
            for j,p in enumerate(mid):
                ns,np=counts(s),counts(p)
                # Give each item similar total supervision. Ties favor a natural standard length near V1.
                rank=(abs((np-ns)-target_delta),abs(ns-current),i+j,i,j)
                options.append((rank,s,p,ns,np))
        _,s,p,ns,np=min(options,key=lambda v:v[0])
        row["control_response"]=s;row["midstep_response"]=p
        rows.append(dict(id=row["id"],schema=row["schema"],standard=ns,preserved=np,delta=np-ns))
    out=HERE/"target_v2"
    out.mkdir(exist_ok=True)
    raw="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in candidates).encode()
    (out/"train_pairs.jsonl").write_bytes(raw)
    receipt=read(folder/"train_receipt.json")
    receipt.update(train_sha256=file_sha(out/"train_pairs.jsonl"),target_version=2,version1_sha256=V1,
                   amendment="Prior to student training, choose equivalent genuine numerical substitutions and shorter explicit intermediate renderings using the pinned tokenizer. Input, E, R, Final, selected IDs, source-review outcomes and teacher intermediate witnesses are fixed.")
    (out/"train_receipt.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    def stats(rs):
        ds=[r["delta"] for r in rs]
        return dict(n=len(rs),standard=sum(r["standard"] for r in rs),preserved=sum(r["preserved"] for r in rs),
                    median=median(ds),min=min(ds),max=max(ds),longer=sum(d>0 for d in ds),shorter=sum(d<0 for d in ds))
    summary={k:stats([r for r in rows if r["schema"]==k]) for k in sorted({r["schema"] for r in rows})}
    summary["all_changed"]=stats(rows)
    (out/"match_audit.json").write_text(json.dumps(dict(summary=summary,samples=rows),indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary))


if __name__=="__main__":
    main()
