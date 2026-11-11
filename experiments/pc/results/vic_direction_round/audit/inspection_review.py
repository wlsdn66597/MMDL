"""Blind inspection-form review: inspect complete novel answers, then seal manual labels."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/"eval/review/inspection.jsonl"
DEC=ROOT/"audit/inspection_decisions.txt"
FIELDS=("E","R","A","consistent","answer","first_E_correct","image_corrected")


def cards():
    return sorted([json.loads(s) for s in PACK.read_text(encoding="utf8").splitlines() if s],
                  key=lambda r:(r["reference"]["scene"],r["reference"]["variant"],r["reference"]["rule_index"],r["key"]))


def display(rows,scene=None,rule=None):
    for i,card in enumerate(rows):
        ref=card["reference"]
        if scene is not None and scene not in ref["scene"] or rule is not None and rule!=ref["rule_index"]:continue
        src=ref["source"]
        a,b,n,m=(src[k] for k in ("a","b","n1","n2"))
        pct=100*(Fraction(a+b,n+m) if ref["rule_index"]==0 else (Fraction(a,n)+Fraction(b,m))/2)
        print(i,card["key"][:12],ref["scene"],ref["variant"],ref["rule_index"],src,ref["options"],ref["gold"],ref["value"],float(pct),card["finish"],"\n",card["raw"],"\n")


def seal(rows):
    by={c["key"][:12]:c for c in rows}
    assert len(by)==len(rows)
    grades={}
    for line in DEC.read_text(encoding="utf8").splitlines():
        if not line.strip() or line.startswith("#"):continue
        k,bits,why=line.split("|")
        c=by[k]
        assert len(bits)==7 and why.strip() and c["key"] not in grades
        vals={f:{"0":False,"1":True,"?":None}[v] for f,v in zip(FIELDS,bits)}
        grades[c["key"]]=dict(key=c["key"],id=c["id"],**vals,reason=why)
    assert len(grades)==len(rows),(len(grades),len(rows))
    out=PACK.with_name("inspection_labels.jsonl")
    data=("".join(json.dumps(grades[k],ensure_ascii=False)+"\n" for k in sorted(grades))).encode()
    if out.exists():assert out.read_bytes()==data
    out.write_bytes(data)
    print(len(rows),hashlib.sha256(data).hexdigest())


def transcribe(rows):
    """Materialize the manual decisions in packet order, before unmasking."""
    assert hashlib.sha256(PACK.read_bytes()).hexdigest()=="bd4b395687fbaf5c4f747cc5738f1a3c201b422266a47231237b6ef3189c1f71"
    correct={0,2,3,4,5,6,9,10,11,12,13,14,15,16,21,22,23,28,29,30,32,33,34,35,
             37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52}
    bad_binding={1,7,8,31,36}
    premature_round={17,18,19,20}
    bad_rate={24,25,26,27}
    assert len(rows)==53==len(correct)+len(bad_binding)+len(premature_round)+len(bad_rate)
    assert correct|bad_binding|premature_round|bad_rate==set(range(53))
    lines=[]
    for i,c in enumerate(rows):
        key,ref=c["key"],c["reference"]
        src=ref["source"]
        a,b,n1,n2=(src[k] for k in ("a","b","n1","n2"))
        m=re.search(r"(?im)Final answer:\s*([A-D])\.?\s*$",c["raw"].rstrip().splitlines()[-1])
        assert c["finish"]=="stop" and m and m[1]==ref["gold"]
        if i in correct:
            bits="111111?"
            if ref["rule_index"]==0:
                reason=f"Correctly pools passes ({a}+{b}) and inspections ({n1}+{n2}), obtaining {ref['value']} and its mapped choice."
            else:
                reason=f"Matches both source fractions {a}/{n1} and {b}/{n2}, weights them equally and selects the mapped {ref['value']} option."
        elif i in bad_binding:
            bits="0110100"
            reason=f"E explicitly calls the inspected counts {n1},{n2} passing units. A uses the correct passed counts {a},{b} without resolving the binding error from the image."
        elif i in premature_round:
            bits="110011?"
            reason="A uses rounded batch percentages and obtains 63.6%. It selects the 63.5% option without computing the unrounded mean of 10/23 and 61/73."
        else:
            bits="110011?"
            reason="Raw cells are correct, but 61/73 is 83.5616%, rounded to 83.6%. A relies on the misstated 83.5% and reaches the correct 72.2% choice."
        lines.append(f"{key[:12]}|{bits}|{reason}")
    data="\n".join(lines)+"\n"
    if DEC.exists():assert DEC.read_text(encoding="utf8")==data
    DEC.write_text(data,encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["view","seal"]);p.add_argument("--scene");p.add_argument("--rule",type=int)
    args=p.parse_args();rows=cards()
    if args.mode=="view":display(rows,args.scene,args.rule)
    else:
        transcribe(rows)
        seal(rows)
