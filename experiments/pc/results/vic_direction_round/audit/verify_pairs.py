"""Independent gold check using readings transcribed directly from the 24 ZIP figures."""
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def values(i):
    ordered={"v_6":("D","G"),"v_15":(36.4,55.5),"v_36":(400000,900000),"v_45":(16.6,16.8),
             "v_57":(12.1,29.7),"v_106":(400,500),"v_147":(62.1,3.8),"v_198":(60,38)}
    if i in ordered:
        a,b=ordered[i]
        return (a,b),(b,a)
    ratios={"c_27":(100-26-64,64),"c_131":(8+10,14+6),"c_285":(6+16,38+32),
            "c_380":(10*11//2,11**2),"c_683":(35,20)}
    if i in ratios:
        r=Fraction(*ratios[i])
        return (r.numerator,r.denominator),(r.denominator,r.numerator)
    if i=="c_38":
        return ((8-5)*100/5,),((5-8)*100/8,)
    if i=="v_8":
        return (3,3),(-3,-3)
    if i=="c_239":
        return (4-1,-1-2),(1-4,2+1)
    signed={"v_13":Decimal("36.8")-Decimal("39.5"),"v_49":31-27,"v_58":7-4,
            "c_158":70-70*30/35,"c_292":51+10-53,"c_816":-40-25,
            "c_936":Decimal("6.8362")-Decimal("6.72"),"c_962":-128-134}
    return (signed[i],),(-signed[i],)


def selected(row, arm):
    opts=dict(re.findall(r"(?m)^([A-D])\. (.+)$",row[arm+"_prompt"]))
    label=row[arm+"_response"].splitlines()[-1].split(":")[-1].strip()
    val=opts[label].replace("−","-")
    tokens=re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", val)
    if tokens:return tuple(Decimal(v.replace(",","")) for v in tokens)
    return tuple(re.findall("[DG]",val))


def main():
    path=ROOT/"data/direction_pairs.jsonl"
    rows=[json.loads(l) for l in path.read_text(encoding="utf8").splitlines()]
    assert len(rows)==48
    seen=Counter()
    for r in rows:
        i=re.search(r"[vc]_\d+",r["group_id"])[0]
        forward,reverse=values(i)
        for arm,expected in (("control",forward),("reverse",reverse if r["role"]=="alternate" else forward)):
            target=tuple(Decimal(str(v)) if isinstance(v,(int,float)) else v for v in expected)
            assert selected(r,arm)==target,(i,r["role"],arm,selected(r,arm),target)
        seen[i]+=1
    assert seen=={r["id"]:2 for r in json.loads((ROOT/"audit/source_rows.json").read_text(encoding="utf8"))}
    proof=dict(groups=len(seen),arm_targets_checked=2*len(rows),data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               scope="Manual ZIP image readings plus independent ordered pair, sign, percentage or ratio recomputation; final option checked for each arm and row.")
    (ROOT/"audit/pair_check.json").write_text(json.dumps(proof,indent=2)+"\n",encoding="utf8")
    print(proof)


if __name__=="__main__":
    main()
