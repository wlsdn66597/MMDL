"""Verify source-native fields and publish the pre-outcome general/sentinel roster."""
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/".runtime/era_review_deps"))
import pyarrow.parquet as pq
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf8"))
def lines(p):return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines()]
def seal(path,data):
    if path.exists():assert path.read_bytes()==data,path
    else:path.write_bytes(data)
def main():
    cc=lines(HERE/"candidates.jsonl")
    manual=read(HERE/"source_review.json")
    approved=manual["chart_witness"]
    assert len(approved)==16
    keep=[]
    chart_order=list(approved)
    sids=read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_splits.json")["val"]
    source_file=ROOT/"tmp/er_r7_sources/scienceqa_val.parquet"
    assert sha(source_file.read_bytes())=="e66b5215ce71e748ade4ee629bc14fbf86762071d355a4fcd831581cc04d72d8"
    source=pq.read_table(source_file).to_pylist()
    src={str(i):r for i,r in zip(sids,source,strict=True)}
    corpus=read(ROOT/"tmp/er_r7_sources/chart-val_human.json")
    seen=set()
    for row in cc:
        if row["kind"]=="ChartQA" and row["source_id"] not in approved:continue
        pic=HERE.parent/row['image'];raw=pic.read_bytes()
        assert sha(raw)==row['image_sha256']
        with Image.open(pic) as im:
            fp=(im.size,sha(im.convert("RGB").tobytes()))
            assert fp not in seen
            seen.add(fp)
        if row['source']=="ScienceQA/validation":
            s=src[row["source_id"]]
            assert row["question"]==s["question"] and row["hint"]==s["hint"] and row["options"]==s["choices"] and row["gold"]==chr(65+s["answer"])
            assert raw==s["image"]["bytes"] and row["reference"]["source_solution"]==s["solution"]
        elif row["kind"]=="ChartQA":
            sid=row['source_id'];witness=approved[sid]
            assert any(r['imgname']==sid+".png" and r['query']==row['question'] and str(r['label'])==row['reference']['native_answer'] for r in corpus)
            # Width of the decimal representation can differ; numeric value is fixed.
            from decimal import Decimal
            old=ord(row["gold"])-65
            assert Decimal(row["options"][old])==Decimal(row['reference']['native_answer'])==Decimal(witness[0])
            new=chart_order.index(sid)%4
            row['options'][old],row['options'][new]=row['options'][new],row['options'][old];row['gold']=chr(65+new)
            row['reference']['visual_witness']=witness[1]
        keep.append(row)
    assert len(keep)==len(seen)==96 and sum(r["sentinel"] for r in keep)==16
    assert Counter(r["gold"] for r in keep if r["sentinel"])==dict(zip("ABCD",[4]*4))
    assert Counter(r["gold"] for r in keep if r["kind"]=="ChartQA")==dict(zip("ABCD",[4]*4))
    assert len({r['id'] for r in keep})==96
    payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in keep).encode()
    seal(HERE/"items.jsonl",payload)
    rec=read(HERE/"candidate_receipt.json")|dict(n=96,n_images=96,items_sha256=sha(payload),source_review_sha256=sha((HERE/"source_review.json").read_bytes()),
        kinds=dict(Counter(r["kind"] for r in keep)),chart_reviewed_16=True,source_native_verified=80,
        sentinel_fixed_before_solver=True)
    seal(HERE/"receipt.json",(json.dumps(rec,ensure_ascii=False,indent=2)+"\n").encode())
    print(rec)
if __name__=="__main__":main()
