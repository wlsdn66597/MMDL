"""Fetch verified-revision train tables; download images for eligible single-series charts."""
import csv
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
from io import StringIO
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[3]
DATA=Path(__file__).resolve().parent
CACHE=ROOT/"tmp/era_r5_chartqa_train"
REV="044eabfc306abfe9340c5741f0093aefc5973d06"
ORIGIN=f"https://raw.githubusercontent.com/vis-nlp/ChartQA/{REV}/ChartQA%20Dataset/train/"


def get(relative,local):
    if not local.exists():
        request=Request(ORIGIN+relative,headers={"User-Agent":"research-training-corpus"})
        with urlopen(request,timeout=18) as response:
            payload=response.read()
        temp=local.with_suffix(local.suffix+".partial")
        temp.write_bytes(payload)
        temp.replace(local)
    return local.read_bytes()


def handle(name):
    try:
        table=CACHE/(name+".csv")
        table_data=get("tables/"+name+".csv",table)
        rows=list(csv.reader(StringIO(table_data.decode("utf-8-sig"))))
        if not 4<=len(rows)<=12 or any(len(v)!=2 for v in rows):
            return
        if any(not v[0].strip() or v[0].strip().lower() in {"none","nan"} for v in rows[1:]):
            return
        labels=[v[0].strip() for v in rows[1:]]
        if len(labels)!=len(set(labels)) or max(map(len,labels))>34:
            return
        vals=[Decimal(v[1].strip().replace(",","").rstrip("%")) for v in rows[1:]]
        if len(vals)!=len(set(vals)) or any(not v.is_finite() or v<0 or v>100000 or v.as_tuple().exponent < -2 for v in vals):
            return
        get("png/"+name+".png",CACHE/(name+".png"))
        return name
    except Exception:
        return


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    file=ROOT/"reports/era_dataset_examples_20261014/chartqa_train_human.json"
    assert hashlib.sha256(file.read_bytes()).hexdigest()=="4eaaa03e406dbbbe43caff925dbec9930b87fadba960f1c2ab30a5def287c384"
    names={Path(row["imgname"]).stem for row in json.loads(file.read_text(encoding="utf8"))}
    names=sorted(names,key=lambda n:hashlib.sha256(("r5-source/"+n).encode()).digest())
    accepted=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for start in range(0,len(names),72):
            accepted.extend(n for n in pool.map(handle,names[start:start+72]) if n)
            print(f"{start+72} attempted, {len(accepted)} tables/images verified",flush=True)
            if len(accepted)>=130:
                break
            if start>=144 and not accepted:
                raise RuntimeError("Source unavailable")
    (CACHE/"eligible.json").write_text(json.dumps(accepted),encoding="utf8")

if __name__=="__main__":
    main()
