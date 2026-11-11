"""Construct new general evaluation without reading predictions from any checkpoint."""
from collections import Counter
import csv
from decimal import Decimal,InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=ROOT/"tmp/er_r7_sources"
sys.path[:0]=[str(ROOT/".runtime/era_review_deps"),str(ROOT/"results/era_midstep_round/data")]
from build_roster import block_scope,collides,fingerprint,normalized,signatures
from native_dev import VAL_SHA, quality_exclusion
import pyarrow.parquet as pq
from sentinels import records

def sha(blob):return hashlib.sha256(blob).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rows(p):return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))
def key(v):return sha(("r8-independent-native-guard-v1/"+str(v)).encode())
def write(p,b):
    if p.exists() and p.read_bytes()!=b:raise ValueError(f"Different sealed bytes: {p}")
    if not p.exists():p.write_bytes(b)
def signature_bytes(blob):
    with tempfile.NamedTemporaryFile(suffix=".png",delete=False) as f:
        f.write(blob);temp=Path(f.name)
    try:return signatures(fingerprint(temp))
    finally:temp.unlink()

def native_questions():
    ids=read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_splits.json")["val"]
    metadata=read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_problems.json")
    parquet=CACHE/"scienceqa_val.parquet";assert sha(parquet.read_bytes())==VAL_SHA
    source=pq.read_table(parquet).to_pylist()
    assert len(ids)==len(source)==4241
    pool=[]
    for i in sorted(range(len(ids)),key=lambda i:key(ids[i])):
        sid=str(ids[i]);r=source[i];m=metadata[sid]
        # Holoparasitic plant versus a generic plant rule; jar with glass body but no glass option.
        if sid in {"12229","12721"}:continue
        if not r["image"] or not r["solution"].strip() or quality_exclusion(r):continue
        assert m["split"]=="val" and all(m[f]==r[f] for f in ("question","hint","choices","answer","solution"))
        if len(set(r['choices']))!=len(r['choices']) or len(r['choices'])<2:continue
        item=dict(id="s8-sqa-"+sid,kind=r["subject"],source="ScienceQA/validation",source_id=sid,
                  question=r["question"],hint=r["hint"],options=r["choices"],gold=chr(65+r["answer"]),
                  sentinel=False,reference=dict(skill=r["skill"],source_solution=r["solution"]))
        pool.append((item,r["image"]["bytes"]))
    return pool

def scope():
    qs,sigs=block_scope()
    vic=ROOT/"results/vic_era_2000_review/overlap/vic_image_hashes.jsonl"
    for v in rows(vic):
        sigs.add((v["pixel"],v["dh"],v["ph"]));qs.add(v["normalized_question"])
    def include(p,root):
        src=rows(p) if p.suffix==".jsonl" else read(p)
        for v in src:
            f=root/v["image"];sigs.add(signatures(fingerprint(f)))
            q=v.get("question") or v.get("prompt","").split("Question: ")[-1].split("\nOptions:")[0];qs.add(normalized(q))
    for rel in ["results/era_midstep_round/data/teacher_inputs.json","results/era_midstep_round/eval/spec/charts/inputs.json","results/era_midstep_round/eval/spec/science/inputs.json"]:
        p=ROOT/rel;include(p,p.parent)
    r7=ROOT/"results/era_er_round"
    for f in ("general/items.jsonl","data/rows.jsonl","paired/items.jsonl"):
        include(r7/f,r7)
    return qs,sigs

CHART_EXCLUDE={"4083","multi_col_20326","11756","5436","multi_col_40608","16804",
    "2562","2595","4576","12442699004228","12784129003060","00484591006451","10075413003231",
    "two_col_22481","multi_col_60981","two_col_5941"}
def chart_pool():
    src=CACHE/"chart-val_human.json"
    assert sha(src.read_bytes())=="e297ce5b38b4ab79bd0e22a93eb19a65cb89f9bd07ce886cb78f2e333fb0ba14"
    charts=read(src);first={}
    for r in charts:first.setdefault(r["imgname"],r)
    for fn in sorted(first,key=key):
        n=Path(fn).stem
        file=CACHE/("chart-png_"+fn);table=CACHE/("chart-tables_"+n+".csv")
        if not file.exists():
            ext=ROOT/"tmp/er_r8_chartqa"
            file=ext/("chart-png_"+fn);table=ext/("chart-tables_"+n+".csv")
        if n in CHART_EXCLUDE or not file.is_file() or not table.is_file():continue
        r=first[fn]
        try:answer=Decimal(str(r["label"]))
        except InvalidOperation:continue
        if answer<0 or not answer.is_finite() or any(v in r["query"].lower() for v in ("ratio","average","maximum change","how many times","combined percentage","mention","peak point","how many colors","which year mentioned")):continue
        tab=list(csv.reader(io.StringIO(table.read_text(encoding="utf-8-sig"))))
        cells=[]
        for rr in tab[1:]:
            for j,v in enumerate(rr):
                try:
                    val=Decimal(v.strip().replace(",","").rstrip("%"))
                    if val.is_finite():cells.append((val,rr[0],tab[0][j] if j<len(tab[0]) else "cell"))
                except InvalidOperation:pass
        if not any(val==answer for val,k,v in cells):continue
        # Present viable in-chart distractors; if a table has one or two numbers, complete with adjacent scale values.
        alts=sorted({v for v,k,l in cells if v!=answer and (v>1900)==(answer>1900)},key=lambda v:(abs(v-answer),v))
        step=Decimal(1).scaleb(answer.as_tuple().exponent)
        for distance in (1,-1,2,3):
            alt=answer+distance*step
            if alt>=0 and alt!=answer and alt not in alts:alts.append(alt)
        opts=[format(v.normalize(),"f") for v in [answer,*alts[:3]]]
        if len(opts)!=4:continue
        yield dict(id="s8-chart-"+n,kind="ChartQA",source="ChartQA/val-human",source_id=n,question=r["query"],hint="",options=opts,
                   gold="A",sentinel=False,reference=dict(table=tab,native_answer=str(r["label"]),
                   source_table_sha256=sha(table.read_bytes()))),file.read_bytes()

def main():
    imgdir=HERE/"images";imgdir.mkdir(parents=True,exist_ok=True)
    qs,sigs=scope();seen_q=set();seen_sig=set();out=[];cnt=Counter();skills=Counter()
    def accept(item,blob):
        q=normalized(item["question"]);sig=signature_bytes(blob)
        if q in qs|seen_q or collides(sig,sigs) or collides(sig,seen_sig):return False
        seen_q.add(q);seen_sig.add(sig)
        fname=item["id"]+".png";write(imgdir/fname,blob)
        out.append(item|dict(image="general/images/"+fname,image_sha256=sha(blob)))
        return True
    for r,blob in records():
        assert accept(r,blob),r["id"]
    pool=native_questions()
    for subject,cap in [("social science",10),("language science",8),("natural science",64),("social science",64),("language science",64)]:
        for r,blob in pool:
            if sum(cnt.values())==64 or cnt[subject]>=cap:break
            if r["kind"]!=subject or skills[r["reference"]["skill"]]>=7:continue
            if accept(r,blob):cnt[subject]+=1;skills[r["reference"]["skill"]]+=1
    assert sum(cnt.values())==64,cnt
    n=0
    for r,b in chart_pool():
        pos=n%4;r['options'][pos],r['options'][0]=r['options'][0],r['options'][pos];r['gold']=chr(65+pos)
        if accept(r,b):n+=1
        if n==32:break
    assert n>=16,n
    out.sort(key=lambda r:key(r["id"]))
    assert len(out)==len({r['id'] for r in out})==80+n
    payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in out).encode()
    write(HERE/"candidates.jsonl",payload)
    receipt=dict(n=len(out),n_images=len(out),n_sentinels=16,sentinel_groups=dict(Counter(v["sentinel_type"] for v in out if v["sentinel"])),
                 sentinel_ids=[v["id"] for v in out if v["sentinel"]],kinds=dict(Counter(v["kind"] for v in out)),science_source_sha256=VAL_SHA,
                 chart_source_sha256=sha((CACHE/"chart-val_human.json").read_bytes()),chart_revision="044eabfc306abfe9340c5741f0093aefc5973d06",
                 science_revision="f18b0a70359ebfb41f658fd564208d0355b013f4",items_sha256=sha(payload),
                 scope=dict(image_sigs=len(sigs),q_stems=len(qs),thresholds="exact/dHash<=4/pHash<=6"),
                 source_selection="Versioned ID-based order and review of source drawing, before solver outputs.")
    write(HERE/"candidate_receipt.json",(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n").encode())
    print(receipt)
if __name__=="__main__":main()
