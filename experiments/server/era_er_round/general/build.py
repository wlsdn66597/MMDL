"""Freeze 72 image-bearing ScienceQA validation and 24 ChartQA human validation items."""
import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
import random
import re
import sys
from urllib.request import Request, urlopen

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=ROOT/"tmp/er_r7_sources"
sys.path[:0]=[str(ROOT/".runtime/era_review_deps"),str(ROOT/"results/era_midstep_round/data")]
from build_roster import block_scope, collides, fingerprint, normalized, signatures
from native_dev import VAL_SHA, quality_exclusion
import pyarrow.parquet as pq

REV="044eabfc306abfe9340c5741f0093aefc5973d06"
BASE=f"https://raw.githubusercontent.com/vis-nlp/ChartQA/{REV}/ChartQA%20Dataset/val/"
QUOTAS={"natural science":50,"social science":14,"language science":8}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def key(s):
    return sha(("r7-independent-general-v1/"+str(s)).encode())


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def fetch(rel):
    dst=CACHE/("chart-"+rel.replace("/","_"))
    if not dst.is_file():
        data=urlopen(Request(BASE+rel,headers={"User-Agent":"research-evaluation"}),timeout=18).read()
        tmp=dst.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dst)
    return dst


def charts():
    raw=json.loads(fetch("val_human.json").read_text(encoding="utf8"))
    assert raw==read(ROOT/"tmp/era_pairs_chartqa_val_20261105/val_human.json")
    # Prefer direct numeric answers recorded in the source table. Disallow noisy operations/subjective rounding.
    opts=sorted((r for r in raw if re.fullmatch(r"\d+(?:\.\d+)?",str(r["label"])) and
                 not re.search(r"difference|ratio|subtract|average|sum of|how many|rounded|median|times|divide",r["query"],re.I)),
                key=lambda r:key(r["imgname"]+r["query"]))
    unique={}
    for r in opts:unique.setdefault(r["imgname"],r)
    picked=list(unique.values())[:80]
    def get(r):
        n=Path(r["imgname"]).stem
        return r,fetch("png/"+r["imgname"]),fetch("tables/"+n+".csv")
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(get,picked))


def main():
    CACHE.mkdir(exist_ok=True,parents=True)
    (HERE/"images").mkdir(exist_ok=True)
    blocked_q,blocked=block_scope()
    for r in (ROOT/"results/vic_era_2000_review/overlap/vic_image_hashes.jsonl").read_text(encoding="utf8").splitlines():
        v=json.loads(r);blocked.add((v["pixel"],v["dh"],v["ph"]));blocked_q.add(v["normalized_question"])
    for rel in ["results/era_midstep_round/data/teacher_inputs.json","results/era_midstep_round/eval/spec/charts/inputs.json","results/era_midstep_round/eval/spec/science/inputs.json"]:
        src=ROOT/rel
        for v in read(src):
            blocked.add(signatures(fingerprint(src.parent/v["image"])))
            blocked_q.add(normalized(v.get("question") or v["prompt"].split("Question: ")[-1].split("\nOptions:")[0]))
    result=[]
    def accept(r,data):
        name=r["id"]+".png"
        pic=CACHE/"probe.png";pic.write_bytes(data)
        sig=signatures(fingerprint(pic))
        if normalized(r["question"]) in blocked_q or collides(sig,blocked):return False
        path=HERE/"images"/name
        path.write_bytes(data)
        blocked_q.add(normalized(r["question"]));blocked.add(sig)
        result.append(r|dict(image="general/images/"+name,image_sha256=sha(data)))
        return True

    native=CACHE/"scienceqa_val.parquet"
    assert sha(native.read_bytes())==VAL_SHA
    source=ROOT/"reports/era_dataset_examples_20261014"
    metadata=read(source/"scienceqa_problems.json")
    sids=read(source/"scienceqa_splits.json")["val"]
    rows=pq.read_table(native).to_pylist()
    assert len(rows)==len(sids)==4241
    counts=Counter();skills=Counter()
    for i in sorted(range(len(rows)),key=lambda i:key(sids[i])):
        r=rows[i];sid=str(sids[i]);sub=r["subject"];m=metadata[sid]
        if counts[sub]>=QUOTAS.get(sub,0) or skills[r["skill"]]>=6 or not r["image"] or not r["solution"].strip() or quality_exclusion(r):continue
        assert m["split"]=="val" and all(r[f]==m[f] for f in ("question","hint","choices","answer","solution"))
        if accept(dict(id="sqa-"+sid,kind=sub,source="ScienceQA/validation",source_id=sid,question=r["question"],hint=r["hint"],
                       options=r["choices"],gold=chr(65+r["answer"]),reference=dict(skill=r["skill"],source_solution=r["solution"])),
                  r["image"]["bytes"]):
            counts[sub]+=1;skills[r["skill"]]+=1
    assert dict(counts)==QUOTAS,counts
    # Historical source review found ambiguously specified years/stacks or table/image disagreements.
    exclude_charts={"4083","multi_col_20326","11756","5436","multi_col_40608","16804"}
    for row,pic,table in charts():
        n=Path(row["imgname"]).stem
        if n in exclude_charts:continue
        tab=list(csv.reader(io.StringIO(table.read_text(encoding="utf-8-sig"))))
        q=row["query"];answer=Decimal(str(row["label"]));unit=Decimal(1).scaleb(answer.as_tuple().exponent)
        values=[]
        for rowdata in tab[1:]:
            for v in rowdata[1:]:
                try:
                    value=Decimal(v.replace(",","").strip().rstrip("%"))
                    if value.is_finite():values.append(value)
                except Exception:pass
        if answer not in values:continue
        alternatives=sorted({v for v in values if v!=answer},key=lambda x:(abs(x-answer),x))
        for dist in (1,-1,2,3):
            v=answer+dist*unit
            if v>=0 and v!=answer and v not in alternatives:alternatives.append(v)
        opts=[format(x.normalize(),"f") for x in [answer,*alternatives[:3]]]
        if len(opts)!=4:continue
        goldpos=counts["ChartQA"]%4
        swap=opts[0];opts[0]=opts[goldpos];opts[goldpos]=swap
        if accept(dict(id="chart-"+n,kind="ChartQA",source="ChartQA/val-human",source_id=n,question=q,hint="",options=opts,
                       gold=chr(65+goldpos),reference=dict(table=tab,native_answer=str(answer),source_table_sha256=sha(table.read_bytes()))),pic.read_bytes()):
            counts["ChartQA"]+=1
        if counts["ChartQA"]==24:break
    assert len(result)==96 and counts["ChartQA"]==24,counts
    result.sort(key=lambda r:key(r["id"]))
    out=HERE/"items.jsonl"
    out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in result),encoding="utf8")
    receipt=dict(n=len(result),counts=dict(counts),science_revision="f18b0a70359ebfb41f658fd564208d0355b013f4",science_sha256=VAL_SHA,
                 chart_revision=REV,chart_source_sha256=sha(fetch("val_human.json").read_bytes()),skills=dict(skills),
                 items_sha256=sha(out.read_bytes()),excluded_chart_ids=sorted(exclude_charts),
                 selection="fixed source-ID hash before model responses; exact pixel or dHash<=4 or pHash<=6 screen")
    (HERE/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    print(receipt)


if __name__=="__main__":main()
