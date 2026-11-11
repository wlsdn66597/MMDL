"""Select an independent general roster before model outputs, using public validation sources."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from guards import records

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
R8=ROOT/"results/era_r_only_round"
sys.path.append(str(R8/"general"))
spec=importlib.util.spec_from_file_location("r8_sources",R8/"general/build.py")
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
sha=legacy.sha
read=legacy.read
rows=legacy.rows

def key(value):
    return sha(("r9-general-frozen-v1/"+str(value)).encode())

def scope():
    questions,sigs=legacy.scope()
    for name in ("general/items.jsonl","paired/items.jsonl","data/rows.jsonl"):
        for item in rows(R8/name):
            questions.add(legacy.normalized(item["question"]))
            sigs.add(legacy.signatures(legacy.fingerprint(R8/item["image"])))
    return questions,sigs

def fmt(v):
    return format(v.normalize(),"f")

def chart_candidates():
    candidates={}
    for folder in (ROOT/"tmp/er_r7_sources",ROOT/"tmp/er_r8_chartqa",ROOT/"tmp/era_pairs_chartqa_val_20261105"):
        for file in folder.glob("*.png"):
            sid=file.stem.removeprefix("chart-png_")
            table=folder/(("chart-tables_"+sid if file.name.startswith("chart-png_") else sid)+".csv")
            if table.exists():
                candidates[sid]=(file,table)
    for sid in sorted(candidates,key=key):
        file,table=candidates[sid]
        data=list(csv.reader(table.read_text(encoding="utf-8-sig").splitlines()))
        if not data or not 2<=len(data[0])<=5:continue
        cells=[]
        for row in data[1:]:
            if len(row)!=len(data[0]) or not row[0] or len(row[0])>60:continue
            for j,val in enumerate(row[1:],1):
                if not val:continue
                try: num=Decimal(val.replace(",","").rstrip("%").strip())
                except InvalidOperation:continue
                if not num.is_finite() or num<0 or -num.as_tuple().exponent>2:continue
                cells.append((row[0],data[0][j],num))
        for row,col,gold in sorted(cells,key=lambda x:key((sid,x))):
            if any(r==row and c==col and n!=gold for r,c,n in cells):continue
            other=sorted({n for _,_,n in cells if n!=gold},key=lambda n:(abs(n-gold),n))
            step=max(Decimal("1"),gold/10).quantize(Decimal("1"))
            for dr in (1,-1,2,-2,3):
                v=gold+dr*step
                if v>=0 and v!=gold and v not in other:other.append(v)
            if len(other)<3:continue
            if len(data[0])==2:
                q=f'What value is shown for "{row}" on the plotted scale?'
            elif col.strip() and col.strip()!=row.strip():
                q=f'Which value does the chart show for "{row}" in the "{col}" series or year?'
            else:continue
            yield dict(id="s9-chart-"+sid,kind="ChartQA",source="ChartQA/val, authored lookup",source_id=sid,sentinel=False,
                       hint="",question=q,options=list(map(fmt,[gold,*other[:3]])),gold="A",reference=dict(table=data,
                       visual_witness=f"The source table gives {row}, {col}: {fmt(gold)} on the chart's scale.",
                       source_table_sha256=sha(table.read_bytes()))),file.read_bytes()
            break

def main():
    qs,sigs=scope()
    n_prior=len(sigs)
    accepted=[];image_sigs=set();new_q=set()
    def accept(record,blob):
        q=legacy.normalized(record["question"])
        sig=legacy.signature_bytes(blob)
        if q in qs or q in new_q or legacy.collides(sig,sigs) or legacy.collides(sig,image_sigs):
            return False
        image_sigs.add(sig);new_q.add(q)
        accepted.append((record,blob))
        return True
    for r,blob in records():
        if not accept(r,blob):
            raise ValueError(f"sentinel collision {r['id']}")
    science=sorted(legacy.native_questions(),key=lambda pair:key(pair[0]["id"]))
    counts=Counter();skills=Counter()
    excluded=read(HERE/"science_exclusions.json")
    rejected=set(excluded["ids"])
    exclude_skills=set(excluded["skills"])
    for subject,cap in (("social science",10),("language science",8),("natural science",64),("language science",64),("social science",64)):
        for r,blob in science:
            if sum(counts.values())>=64 or counts[subject]>=cap:break
            skill=r["reference"]["skill"]
            if r["source_id"] in rejected or skill in exclude_skills or r["kind"]!=subject or skills[skill]>=12:continue
            row=dict(r,id="s9-sqa-"+r["source_id"])
            if accept(row,blob):counts[subject]+=1;skills[row["reference"]["skill"]]+=1
    if sum(counts.values())!=64:
        raise ValueError(f"native ScienceQA shortage: {counts}")
    # Local review workspace is excluded from both solver and reviewer releases.
    preview=ROOT/"tmp/r9_general_source_review";preview.mkdir(exist_ok=True)
    (preview/"native.json").write_text(json.dumps([r for r,b in accepted[16:]],ensure_ascii=False,indent=2),encoding="utf8")
    for r,b in accepted[:80]:(preview/(r["id"]+".png")).write_bytes(b)
    # Require individually confirmed visible row, series and scale before admitting a native chart.
    candidates={}
    for r,blob in chart_candidates():
        if accept(r,blob):
            candidates[r["source_id"]]=(r,blob)
    if not (HERE/"chart_selection.json").exists() or not (HERE/"native_review.json").exists():
        folder=preview/"chart_candidates";folder.mkdir(exist_ok=True)
        for r,b in candidates.values():(folder/(r["source_id"]+".png")).write_bytes(b)
        (folder/"roster.json").write_text(json.dumps([r for r,b in candidates.values()],indent=2,ensure_ascii=False),encoding="utf8")
        print("Review before sealing:",len(candidates),"chart candidates,",counts,flush=True)
        return
    approved=read(HERE/"chart_selection.json")
    visual=read(HERE/"native_review.json")
    assert sorted(visual["witness"])==sorted(r["source_id"] for r,_ in accepted[16:80])
    for r,_ in accepted[16:80]:
        r["reference"]["visual_witness"]=visual["witness"][r["source_id"]]
    assert len(approved)==16 and all(i in candidates for i in approved)
    final=accepted[:80]
    for i,(sid,witness) in enumerate(approved.items()):
        r,b=candidates[sid]
        pos=i%4
        r["options"][0],r["options"][pos]=r["options"][pos],r["options"][0]
        r["gold"]=chr(65+pos)
        r["reference"]["visual_witness"]=witness
        final.append((r,b))
    assert len(final)==96
    image_folder=HERE/"images";image_folder.mkdir(exist_ok=True)
    manifest=[]
    for r,blob in sorted(final,key=lambda x:key(x[0]["id"])):
        file=image_folder/(r["id"]+".png")
        legacy.write(file,blob)
        manifest.append(r|dict(image="general/images/"+file.name,image_sha256=sha(blob)))
    assert len({r["image_sha256"] for r in manifest})==96
    assert len({r["id"] for r in manifest})==96 and sum(r["sentinel"] for r in manifest)==16
    payload="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in manifest).encode()
    legacy.write(HERE/"items.jsonl",payload)
    receipt=dict(n=96,n_images=96,n_sentinels=16,kinds=dict(Counter(r["kind"] for r in manifest)),
        sentinel_ids=[r["id"] for r in manifest if r["sentinel"]],
        sentinel_groups=dict(Counter(r["sentinel_type"] for r in manifest if r["sentinel"])),
        science_source_sha256=legacy.VAL_SHA,chart_revision="044eabfc306abfe9340c5741f0093aefc5973d06",
        science_revision="f18b0a70359ebfb41f658fd564208d0355b013f4",items_sha256=sha(payload),
        prior_questions=len(qs),prior_images=n_prior,duplicate_filter="exact normalized question / RGB / dHash <= 4 or pHash <= 6",
        chart_selection_sha256=sha((HERE/"chart_selection.json").read_bytes()),
        native_review_sha256=sha((HERE/"native_review.json").read_bytes()),
        chart_questions="Authored per held-out original validation chart and its source table",
        science_questions="Original questions, choices, solutions, gold and source image bytes")
    legacy.write(HERE/"receipt.json",(json.dumps(receipt,indent=2)+"\n").encode())
    print(receipt)

if __name__=="__main__":main()
