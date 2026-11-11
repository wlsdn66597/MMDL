"""Select unused validation images without consulting predictions, then freeze reviewed rows."""
from collections import Counter
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
R9=ROOT/"results/era_a_only_round"
sys.path.insert(0,str(R9/"general"))
spec=importlib.util.spec_from_file_location("r9_general",R9/"general/build.py")
r9=importlib.util.module_from_spec(spec);spec.loader.exec_module(r9)
legacy=r9.legacy
CACHE=ROOT/"tmp/half_confirm_source_review"
sha=legacy.sha


def key(s):
    return sha(("half-strength-confirm-40271/"+str(s)).encode())


def registry():
    qs,sigs=r9.scope()
    for rel in ("general/items.jsonl","paired/items.jsonl","data/rows.jsonl"):
        for r in legacy.rows(R9/rel):
            qs.add(legacy.normalized(r["question"]))
            sigs.add(legacy.signatures(legacy.fingerprint(R9/r["image"])))
    return qs,sigs


def preview(records,prefix):
    for i in range(0,len(records),8):
        canvas=Image.new("RGB",(1360,856),"white");draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(records[i:i+8]):
            x=(j%4)*340;y=(j//4)*428
            with Image.open(CACHE/(r["id"]+".png")) as im:
                image=ImageOps.contain(im.convert("RGB"),(330,309))
                canvas.paste(image,(x+(330-image.width)//2,y+25))
            draw.text((x+5,y+4),r["id"],fill="black",font=ImageFont.truetype("arial.ttf",14))
            import textwrap
            title=r["question"]+" ["+r["options"][ord(r["gold"])-65]+"]"
            for t,line in enumerate(textwrap.wrap(title,40)[:4]):
                draw.text((x+4,y+337+t*17),line,fill="black",font=ImageFont.truetype("arial.ttf",13))
        canvas.save(CACHE/f"{prefix}_{i//8}.jpg",quality=94)


def candidates():
    CACHE.mkdir(parents=True,exist_ok=True)
    questions,signatures=registry()
    accepted_q=set();accepted_s=set()
    def accept(r,blob):
        q=legacy.normalized(r["question"]);sig=legacy.signature_bytes(blob)
        if q in questions|accepted_q or legacy.collides(sig,signatures) or legacy.collides(sig,accepted_s):
            return False
        accepted_q.add(q);accepted_s.add(sig)
        (CACHE/(r["id"]+".png")).write_bytes(blob)
        return True
    excluded=legacy.read(R9/"general/science_exclusions.json")
    pool=[];counts=Counter()
    def extra_train():
        from native_dev import quality_exclusion
        ids=legacy.read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_splits.json")["train"]
        metadata=legacy.read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_problems.json")
        parquet=ROOT/"tmp/era_prereview_20261021/scienceqa_train.parquet"
        import pyarrow.parquet as pq
        table=pq.ParquetFile(parquet)
        assert table.metadata.num_rows==len(ids)==12726
        index=0;native=[]
        for batch in table.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                sid=str(ids[index]);index+=1
                if row["subject"]!="natural science" or not row["image"] or not row["solution"].strip() or quality_exclusion(row):continue
                m=metadata[sid];assert m["split"]=="train" and all(m[f]==row[f] for f in ("question","hint","choices","answer","solution"))
                if len(set(row["choices"]))!=len(row["choices"]) or len(row["choices"])<2:continue
                rec=dict(id="train-"+sid,source_id=sid,source="ScienceQA/train (project-held-out)",kind=row["subject"],question=row["question"],
                         hint=row["hint"],options=row["choices"],gold=chr(65+row["answer"]),sentinel=False,
                         reference=dict(skill=row["skill"],source_solution=row["solution"]))
                native.append((rec,row["image"]["bytes"]))
        return native
    streams=[sorted(legacy.native_questions(),key=lambda pair:key(pair[0]["id"])), sorted(extra_train(),key=lambda pair:key(pair[0]["id"]))]
    for r,b in streams[0]+streams[1]:
        if r["source_id"] in excluded["ids"] or r["reference"]["skill"] in excluded["skills"]:continue
        if r["kind"]!="natural science" or counts[r["reference"]["skill"]]>=18:continue
        row=r|dict(id="hc-sqa-"+r["source_id"])
        if accept(row,b):
            pool.append(row);counts[r["reference"]["skill"]]+=1
            if len(pool)==100:break
    charts=[]
    for r,b in r9.chart_candidates():
        t=r["source_id"]
        r=r|dict(id="hc-chart-"+t)
        if accept(r,b):charts.append(r)
    for tag,rs in (("native",pool),("charts",charts)):
        (CACHE/(tag+".json")).write_text(json.dumps(rs,ensure_ascii=False),encoding="utf8")
        preview(rs,tag)
    print("eligible",len(pool),"native and",len(charts),"charts",counts,flush=True)


def freeze():
    import importlib.util
    spec=importlib.util.spec_from_file_location("confirm_guards",HERE/"anchors.py")
    guards=importlib.util.module_from_spec(spec);spec.loader.exec_module(guards)
    review=legacy.read(HERE/"selected.json")
    final=list(guards.make())
    for cat,limit in (("native",64),("charts",16)):
        witnesses=review[cat]
        candidates={r["id"]:r for r in legacy.read(CACHE/(cat+".json"))}
        assert len(witnesses)==limit
        for n,(id,witness) in enumerate(witnesses.items()):
            r=candidates[id].copy();r["reference"]=dict(r["reference"],visual_witness=witness)
            if cat=="charts":
                opts=list(r["options"]);j=(n+2)%4
                opts[0],opts[j]=opts[j],opts[0];r["options"]=opts;r["gold"]="ABCD"[j]
            final.append((r,(CACHE/(id+".png")).read_bytes()))
    assert len(final)==96
    (HERE/"images").mkdir(exist_ok=True)
    rows=[]
    for r,b in sorted(final,key=lambda x:key(x[0]["id"])):
        file=HERE/"images"/(r["id"]+".png")
        legacy.write(file,b);rows.append(r|dict(image="general/images/"+file.name,image_sha256=sha(b)))
    assert len(set(x["image_sha256"] for x in rows))==96
    assert sum(r["sentinel"] is True for r in rows)==16
    assert len(set(legacy.normalized(r["question"]) for r in rows))==96
    raw="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode()
    legacy.write(HERE/"items.jsonl",raw)
    receipt=dict(n=96,n_images=96,n_sentinels=16,kinds=dict(Counter(r["kind"] for r in rows)),
                 sentinel_groups=dict(Counter(r["sentinel_type"] for r in rows if r["sentinel"])),
                 science_source_sha256=legacy.VAL_SHA,items_sha256=sha(raw),
                 chart_revision="044eabfc306abfe9340c5741f0093aefc5973d06",science_revision="f18b0a70359ebfb41f658fd564208d0355b013f4",
                 selection_sha256=sha((HERE/"selected.json").read_bytes()))
    legacy.write(HERE/"receipt.json",(json.dumps(receipt,indent=2)+"\n").encode())
    print(receipt)


if __name__=="__main__":
    if sys.argv[1]=="candidates":candidates()
    elif sys.argv[1]=="freeze":freeze()
