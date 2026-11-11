import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf8"))


def verify():
    checks=read("receipt.json")
    checks.update(read("freeze.json"))
    for name,h in checks.items():
        path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and hashlib.sha256(path.read_bytes()).hexdigest()==h,name
    gold={i["id"]:i for i in read("holdout/gold.json")}
    with (ROOT/"scores.csv").open(encoding="utf-8-sig",newline="") as f:
        scores={(r["arm"],r["id"]):r for r in csv.DictReader(f)}
    assert len(scores)==216
    judge=read("spotcheck.json")
    assert len(judge)==24
    out=[]
    for j in judge:
        r=scores[j["arm"],j["id"]]; it=gold[j["id"]]; d=it["mode"]=="draft"
        if d:
            gold_rep=r["supported"]=="True" and j["id"].endswith("dreplace")
        else:
            gold_rep=r["repair"]=="True"
        target=dict(repair=gold_rep,keep=r["retain"]=="True")
        if not d:target["joint"]=r["joint"]=="True"
        pj=j["parsed"]
        good=isinstance(pj,dict) and all(pj.get(k) in (0,1) for k in ("joint","repair","keep"))
        agreement={k:bool(pj[k])==v for k,v in target.items()} if good else {}
        out.append(dict(**j,reference=target,valid=good,agreement=agreement))
    (ROOT/"spotcheck_review.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for key in ("joint","repair","keep"):
        vals=[r["agreement"][key] for r in out if key in r["agreement"]]
        print(key,sum(vals),len(vals))
    print("verified files",len(checks),"responses",len(scores),"judge",sum(r["valid"] for r in out),"/ 24")


if __name__=="__main__":verify()
