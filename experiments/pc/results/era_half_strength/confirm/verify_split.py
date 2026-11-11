"""Audit release integrity, external question/image overlap and separate solver inputs."""
import hashlib
import importlib.util
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys

from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
R9=PROJECT/"results/era_a_only_round"
sys.path[:0]=[str(R9/"paired"),str(PROJECT/".runtime/era_review_deps"),str(PROJECT/"results/vic_era_2000_review/overlap")]
from check_sources import enroll
from check_overlap import load_refs,fingerprint,signature,match,qbody


def rows(f):
    return list(map(json.loads,f.read_text("utf8").splitlines()))


def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    refs,manifests=load_refs()
    vic=PROJECT/"results/vic_era_2000_review/overlap/vic_image_hashes.jsonl"
    refs["VIC2000"]=[(r["id"],r["normalized_question"],[(r["image"],r)]) for r in rows(vic)]
    manifests["VIC2000"]={vic.relative_to(PROJECT).as_posix():digest(vic)}
    direction=PROJECT/"results/vic_direction_round"
    enroll(refs,manifests,"round6_direction",direction/"data/direction_pairs.jsonl",direction/"data")
    for rnd in ("era_er_round","era_r_only_round","era_a_only_round"):
        directory=PROJECT/"results"/rnd
        for rel in ("data/rows.jsonl","paired/items.jsonl","general/items.jsonl"):
            enroll(refs,manifests,rnd+"/"+rel,directory/rel,directory)
    enroll(refs,manifests,"R9 complete 296",R9/"train/manifests/ordinary.jsonl",PROJECT)
    pool={}; qids={}
    for tag,ref in refs.items():
        for rid,q,pics in ref:
            qids.setdefault(q,[]).append((tag,rid))
            for f,fp in pics:pool.setdefault(fp["pixel"],(tag,rid,f,fp))
    cards=rows(ROOT/"paired/items.jsonl")+rows(ROOT/"general/items.jsonl")
    assert len(cards)==len(set(r["id"] for r in cards))==192
    pics={};texts={};exact=[];near=[];qdupes=[]
    for c in cards:
        q=qbody(c["question"])
        assert q not in texts,q
        texts[q]=c["id"]
        if q in qids:qdupes.append((c["id"],qids[q]))
        if c["image"] not in pics:
            pic=signature(fingerprint(ROOT/c["image"]))
            pics[c["image"]]=pic
            for tag,rid,f,old in pool.values():
                hit=match(pic,old)
                if hit:
                    result=dict(image=c["image"],pool=tag,id=rid,old=f,**hit)
                    (near if hit["kind"]=="near" else exact).append(result)
    assert len(pics)==len({v["pixel"] for v in pics.values()})==144
    internal=[]
    for (pa,a),(pb,b) in combinations(pics.items(),2):
        hit=match(a,b)
        if hit:internal.append(dict(left=pa,right=pb,**hit))
    report=dict(counts={tag:len(r) for tag,r in refs.items()},manifest_sha256=manifests,
                unique_prior_pixels=len(pool),unique_prior_normalized_questions=len(qids),
                exact=exact,near=near,duplicate_text=qdupes,
                internal_near_by_type=dict(Counter(x["left"].split("/")[0]+"/"+x["right"].split("/")[0] for x in internal)),
                internal_near=internal)
    path=ROOT/"overlap.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    assert not exact and not qdupes,(exact,qdupes)
    print("external near",len(near),"unique prior pixels",len(pool),"question stems",len(qids),flush=True)
    spec=importlib.util.spec_from_file_location("r9_separation",R9/"eval/prepare.py")
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    prepare.ROOT=ROOT;prepare.HERE=ROOT/"eval";prepare.main()
    receipts={p.relative_to(ROOT).as_posix():digest(p) for p in [ROOT/"paired/receipt.json",ROOT/"general/receipt.json",ROOT/"paired/items.jsonl",ROOT/"general/items.jsonl",ROOT/"paired/sources.jsonl",ROOT/"eval/spec/receipt.json",ROOT/"overlap.json"]}
    train=PROJECT/"tmp/era_prereview_20261021/scienceqa_train.parquet"
    gen=rows(ROOT/"general/items.jsonl")
    receipt=dict(paired_groups=48,paired=96,general=96,sentinel=16,source_counts=dict(Counter(x["source"] for x in gen)),
                 unique_images=144,external_exact_image=len(exact),external_near=len(near),external_exact_question=len(qdupes),
                 scienceqa_train_parquet_sha256=digest(train),proof=receipts)
    (ROOT/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    anchor=[r for r in gen if r["sentinel"]]
    for i in range(0,16,4):
        out=Image.new("RGB",(1080,696),"white");d=ImageDraw.Draw(out)
        for j,r in enumerate(anchor[i:i+4]):
            pic=ImageOps.contain(Image.open(ROOT/r["image"]).convert("RGB"),(528,320))
            x=(j%2)*540;y=(j//2)*348
            out.paste(pic,(x,y));d.text((x+5,y+320),r["id"]+" "+r["gold"],fill="black")
        out.save(ROOT/"general"/f"guard_sheet_{i//4}.jpg",quality=93)
    print(receipt)


if __name__=="__main__":main()
