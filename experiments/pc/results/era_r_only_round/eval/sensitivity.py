"""Secondary checks for wording ambiguity and precision without modifying official grades."""
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/"eval/review"


def count(values):
    return [sum(v is True for v in values),sum(v is not False for v in values)]


def both(values):
    return False if False in values else None if None in values else True


def main():
    items=list(map(json.loads,(ROOT/"paired/items.jsonl").read_text(encoding="utf8").splitlines()))
    ambiguous={r["group"] for r in items if "including its return trips" in r["question"].casefold()}
    assert len(ambiguous)==4
    # Masked reviews explicitly isolated these small rounding discrepancies, all with the correct substantive winner.
    rounded_prefixes={"cd800ac5b1bc","b044fa8a55ac","23dcd63fc051","ceafa60f55d7","a3e786aab6b6","ebbf89fafdb5"}
    rows=[r for r in map(json.loads,(REVIEW/"scored.jsonl").read_text(encoding="utf8").splitlines()) if r["dataset"]=="paired"]
    assert rounded_prefixes=={r["key"][:12] for r in rows if r["key"][:12] in rounded_prefixes}
    output={}
    for name,exclude,relax in (("official",False,False),("omit_return_trips",True,False),("rounding_tolerant",False,True),("both",True,True)):
        by_arm={}
        for arm in ("base","ordinary","enhanced"):
            groups=defaultdict(list)
            for r in rows:
                if r["arm"]!=arm or (exclude and r["group"] in ambiguous):continue
                joint=r["joint"]
                if relax and r["key"][:12] in rounded_prefixes:
                    assert r["answer"] and r["e"] and r["r"] and r["winner"] and r["consistent"]
                    joint=True
                groups[r["group"]].append((r["r"],joint))
            by_arm[arm]={}
            for i,k in enumerate(("r","joint")):
                values=[both([t[i] for t in v]) for v in groups.values()]
                by_arm[arm][k]=count(values)
            by_arm[arm]["n"]=len(groups)
        output[name]=by_arm
    result=dict(excluded_groups=sorted(ambiguous),rounding_key_prefixes=sorted(rounded_prefixes),scenarios=output)
    (REVIEW/"sensitivity.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    for n,rs in output.items():print(n,{a:(v["r"],v["joint"],v["n"]) for a,v in rs.items()})
if __name__=="__main__":main()
