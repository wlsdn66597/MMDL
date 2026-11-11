"""Review only new anonymous ScienceQA texts using the same source-witness rubric as R5."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/"eval/review"
FIELDS=("E","R","A","consistent","answer","first_E_correct","image_corrected")


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s]


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=("ids","show","labels"))
    cli.add_argument("--ids",nargs="+")
    a=cli.parse_args()
    cards=rows(REVIEW/"science.jsonl")
    if a.mode=="ids":
        print([(s,sum(r["id"]==s for r in cards)) for s in sorted({r["id"] for r in cards})])
        return
    if a.mode=="show":
        for r in sorted(cards,key=lambda c:(c["id"],c["key"])):
            if not a.ids or r["id"].split("-")[-1] in a.ids:
                print(r["key"][:12],r["id"],r["finish"],r["reference"]["gold"],r["reference"]["visual_witness"])
                print(r["raw"],"\n")
        return
    lookup={r["key"][:12]:r for r in cards}
    assert len(lookup)==len(cards)
    result={}
    for line in (ROOT/"audit/science_notes.txt").read_text(encoding="utf8").splitlines():
        if not line.strip() or line.startswith("#"):continue
        prefix,bits,reason=line.split("|")
        card=lookup[prefix]
        assert card["key"] not in result and len(bits)==len(FIELDS) and reason.strip()
        flags={f:{"1":True,"0":False,"?":None}[c] for f,c in zip(FIELDS,bits)}
        result[card["key"]]=dict(key=card["key"],id=card["id"],**flags,reason=reason)
    assert len(result)==len(cards)
    path=REVIEW/"science_labels.jsonl"
    content="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in result.values()).encode()
    if path.exists():assert path.read_bytes()==content
    else:path.write_bytes(content)
    print(len(result),hashlib.sha256(content).hexdigest())


if __name__=="__main__":main()
