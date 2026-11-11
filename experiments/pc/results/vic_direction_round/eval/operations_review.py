"""Present complete masked operations responses in review order and encode checked decisions."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FIELDS=("E","R","A","consistent","answer","first_E_correct","image_corrected")

def cards():
    return [json.loads(line) for line in (ROOT/"review/operations.jsonl").read_text(encoding="utf8").splitlines()]

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=["show","labels"])
    cli.add_argument("--scene")
    cli.add_argument("--start",type=int,default=0)
    cli.add_argument("--count",type=int,default=8)
    opt=cli.parse_args()
    cases=cards()
    if opt.mode=="show":
        rows=sorted(cases,key=lambda c:(c["reference"]["scene"],c["reference"]["variant"],c["reference"]["rule_index"],c["reference"]["order"],c["key"]))
        if opt.scene:rows=[c for c in rows if c["reference"]["scene"].endswith(opt.scene)]
        for c in rows[opt.start:opt.start+opt.count]:
            ref=c["reference"]
            print(c["key"],ref["scene"],ref["variant"],ref["rule_index"],ref["order"],ref["value"],ref["answer"],c["finish"],"\n",c["raw"],"\n")
        print(f"{len(rows)} cards in view.")
    else:
        lookup={c["key"]:c for c in cases}
        labels={}
        for line in (ROOT/"review/operations_decisions.txt").read_text(encoding="utf8").splitlines():
            if not line or line.startswith("#"):continue
            key,bits,reason=line.split("|")
            card=lookup[key]
            assert key not in labels and len(bits)==len(FIELDS) and reason.strip()
            vals={f:{"?":None,"1":True,"0":False}[v] for f,v in zip(FIELDS,bits)}
            labels[key]=dict(key=key,id=card["id"],**vals,reason=reason)
        assert labels.keys()==lookup.keys(),sorted(lookup.keys()-labels.keys())
        payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in labels.values())
        file=ROOT/"review/operations_labels.jsonl"
        if file.exists():assert file.read_text(encoding="utf8")==payload
        else:file.write_text(payload,encoding="utf8")

if __name__=="__main__":
    main()
