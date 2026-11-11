"""Check the delivered manifest, option permutations, and gold against independent formulas."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(file):
    return json.loads((ROOT/file).read_text(encoding="utf8"))


def digest(file):
    return hashlib.sha256((ROOT/file).read_bytes()).hexdigest()


def recompute(row):
    s,domain,k = row["source"],row["domain"],row["rule_index"]
    if domain=="inspection":
        if k==0:
            frac=Fraction(s["a"]+s["b"],s["n1"]+s["n2"])
        else:
            frac=(Fraction(s["a"],s["n1"])+Fraction(s["b"],s["n2"]))/2
        tenths=int(1000*frac + Fraction(1,2))
        return f"{tenths//10}.{tenths%10}%"
    if domain=="operations":
        revenue=s["units1"]*s["price1"]+s["units2"]*s["price2"]
        variable=s["units1"]*s["cost1"]+s["units2"]*s["cost2"]
        return str(revenue-variable-(s["fixed"] if k else 0))
    points=[s["start"],s["turn"],s["end"]]
    assert (points[1]-points[0])*(points[2]-points[1])<0
    return str(sum(abs(b-a) for a,b in zip(points,points[1:]))) if k else str(points[-1]-points[0])


def main():
    manifest=read("manifest.json")
    inp, gold=read("inputs.json"),read("gold.json")
    assert manifest["input_sha256"]==digest("inputs.json") and manifest["gold_sha256"]==digest("gold.json")
    assert manifest["generator_sha256"]==digest("generate.py") and manifest["contact_sha256"]==digest("representatives.png")
    by_id={r["id"]:r for r in inp}
    assert len(by_id)==len(inp)==len(gold)==96 and set(by_id)=={g["id"] for g in gold}
    assert len({(r["image"],r["question"],tuple(r["options"])) for r in inp})==96
    assert len({r["image_sha256"] for r in inp})==24
    scenes=defaultdict(list)
    for g in gold:
        row=by_id[g["id"]]
        assert row["image_sha256"]==digest(row["image"])
        assert set(row)=={"id","image","image_sha256","question","options"}
        assert g["value"]==recompute(g)==row["options"]["ABCD".index(g["answer"])]
        scenes[g["scene"]].append(g)
    for gr in scenes.values():
        assert len(gr)==8 and len({g["value"] for g in gr})==4
        semantics={tuple(sorted(by_id[g["id"]]["options"])) for g in gr}
        assert len(semantics)==1 and set(next(iter(semantics)))=={g["value"] for g in gr}
        options={tuple(by_id[g["id"]]["options"]) for g in gr}
        assert len(options)==2
        a,b=list(options)
        assert any(a[i:]+a[:i]==b for i in (1,2,3))
        assert len({by_id[g["id"]]["image"] for g in gr})==2
        assert len({by_id[g["id"]]["question"] for g in gr})==2
        lookup={(r["variant"],r["rule_index"],r["order"]):r for r in gr}
        s,t=lookup[0,0,0]["source"],lookup[1,0,0]["source"]
        assert [k for k in s if s[k]!=t[k]]==[gr[0]["changed_field"]]
        for v in (0,1):
            for k in (0,1):
                x,y = lookup[v,k,0],lookup[v,k,1]
                assert x["value"]==y["value"] and x["answer"]!=y["answer"]
                assert x["evidence"]==y["evidence"]
                assert by_id[x["id"]]["question"]==by_id[lookup[1-v,k,0]["id"]]["question"]
        assert Counter(r["answer"] for r in gr)=={k:2 for k in "ABCD"}
    assert len(scenes)==12
    print("Verified 96 IDs and distinct prompts; 12 scenes, 24 unique image byte strings, one changed source field, four exact values and two rotated option orders per scene; 24 correct per answer letter.")


if __name__=="__main__":
    main()
