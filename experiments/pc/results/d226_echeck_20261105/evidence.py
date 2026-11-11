"""Identify explicitly stated first-E selections. Leave ambiguous or implicit selections for review."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf8"))
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def values_mentioned(text):
    # Remove thousands-grouping commas only; keep separate numbers in lists distinct.
    text=re.sub(r"(?<=\d),(?=\d{3}(?!\d))","",text)
    return {Decimal(x) for x in re.findall(r"(?<![\w.])(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])",text.replace(". ", " ").rstrip("."))}

def main():
    rows=read(ROOT/"comparisons.json");items=read(ROOT/"fresh/items.json")
    cells={c["id"]:c for i in items for c in i["cells"].values()}
    path=ROOT/"evidence_overrides.json"
    overrides=read(path) if path.exists() else {}
    observations=[];stats={}
    for arm in ("baseline","standard","checked"):
        counts=Counter()
        for row in rows:
            rid=row["id"];cell=cells[rid];txt=row[arm]["evidence"]
            pool=set(map(Decimal,cell["rows"].values())); listed=pool & values_mentioned(txt)
            truth=set(map(Decimal,cell["truth"]["values"]));bottom=set(map(Decimal,cell["bottom"]["values"]))
            if len(listed)!=3: state="unclear"
            elif listed==truth:state="rank"
            elif listed==bottom:state="bottom"
            else:state="other"
            key=arm+"/"+rid
            if key in overrides:state=overrides[key][0]
            assert state in {"rank","bottom","other","unclear"}
            counts[state]+=1; counts["E_question_marks"]+=row[arm]["E_question_mark"]
            counts["right_E_wrong_answer"]+=int(state=="rank" and not row[arm]["correct"])
            observations.append(dict(arm=arm,id=rid,status=state,source="manual" if key in overrides else "number_match",
                matched_values=sorted(map(str,listed)),evidence=txt,answer_correct=row[arm]["correct"],
                note=overrides.get(key,["",""])[1]))
        stats[arm]=dict(counts)
    dump(ROOT/"evidence_audit.json",observations);dump(ROOT/"evidence_stats.json",stats)
if __name__=="__main__":main()
