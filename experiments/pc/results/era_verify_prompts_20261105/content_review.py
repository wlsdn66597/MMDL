"""Attach first-E judgments and boundary-check notes from reading the ERA outputs."""
import csv
from decimal import Decimal
import json
import re
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
def read(path): return json.loads(path.read_text(encoding="utf-8"))
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def first_evidence(raw):
    found=re.search(r"(?im)^\s*Evidence:\s*([^\r\n]+)",raw)
    return found[1].strip() if found else ""

def main():
    rows={r["id"]:r for r in read(HERE/"comparisons.json")}
    with (HERE/"content_notes.tsv").open(encoding="utf-8") as stream:
        notes={r["id"]:r for r in csv.DictReader(stream,delimiter="\t")}
    assert notes.keys()==rows.keys()
    out=[];summary={}
    for arm in ("generic","boundary"):
        preds={r["id"]:r for r in map(json.loads,(HERE/arm/"predictions.jsonl").read_text(encoding="utf8").splitlines())}
        counts=Counter()
        for rid,item in rows.items():
            first_e=first_evidence(preds[rid]["raw"])
            code=notes[rid][arm+"_E"]
            assert code in ("rank","lower","other","unclear")
            if code in ("rank","lower"):
                group=item["truth" if code=="rank" else "bottom"]
                values=set(map(Decimal,group["values"]))
                seen={Decimal(v) for v in re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)",first_e)}
                assert values<=seen,(rid,first_e,values-seen)
                assert code!="lower" or not item["aligned"]
            ok=code=="rank"
            final=item[arm]["correct"]
            check=notes[rid]["check"] if arm=="boundary" else "not_assigned"
            counts["E_correct"]+=int(ok)
            counts["initial_E_correct_final_wrong"]+=int(ok and not final)
            counts["initial_E_incorrect_final_correct"]+=int(not ok and final)
            counts[check]+=1
            out.append(dict(id=rid,arm=arm,initial_E=first_e,selection=code,final_correct=final,
                            initial_selection_correct=ok,check=check,note=notes[rid]["note"] if arm=="boundary" else ""))
        summary[arm]=dict(counts)
    dump(HERE/"content_review.json",out);dump(HERE/"content_summary.json",summary)

if __name__=="__main__": main()
