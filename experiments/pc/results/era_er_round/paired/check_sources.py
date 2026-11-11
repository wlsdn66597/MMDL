"""Inspect held-out images against project training/evaluation registries, without model calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"results/vic_era_2000_review/overlap"))
from check_overlap import fingerprint, load_refs, match, qbody, signature
from verify import read, sha


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--train",type=Path,help="R7 training row manifest; images relative to round root")
    cli.add_argument("--general",type=Path,help="R7 general holdout row manifest; images relative to round root")
    a=cli.parse_args()
    data,manifests=load_refs()
    vic=ROOT/"results/vic_era_2000_review/overlap/vic_image_hashes.jsonl"
    data["VIC2000"]=[(r["id"],r["normalized_question"],[(r["image"],r)]) for r in read(vic)]
    manifests["VIC2000"]={vic.relative_to(ROOT).as_posix():sha(vic)}
    for name,path in (("round7_train",a.train),("round7_general",a.general)):
        if path is None:continue
        path=path.resolve()
        rows=read(path); cache={};group=[]
        for r in rows:
            rel=r["image"]; im=HERE.parent/rel
            if im not in cache:cache[im]=signature(fingerprint(im))
            group.append((r["id"],qbody(r["question"]),[(rel,cache[im])]))
        data[name]=group;manifests[name]={path.relative_to(ROOT).as_posix():sha(path)}
    cards=read(HERE/"items.jsonl")
    query={k:qbody(r["question"]) for r in cards for k in [r["id"]]}
    fp={r["group"]:signature(fingerprint(HERE.parent/r["image"])) for r in cards[::2]}
    exact=[];near=[]; stems=[];counts={}
    for name,records in data.items():
        pics={};qs={}
        for rid,q,img in records:
            qs.setdefault(q,rid)
            for rel,z in img: pics.setdefault(z["pixel"],(rid,rel,z))
        counts[name]=dict(records=len(records),unique_images=len(pics))
        for rid,q in query.items():
            if q in qs:stems.append(dict(dev=rid,source=name,ref=qs[q]))
        for group,ours in fp.items():
            for rid,rel,z in pics.values():
                hit=match(ours,z)
                if hit is None:continue
                pair=dict(group=group,source=name,ref=rid,image=rel,**hit)
                (near if hit["kind"]=="near" else exact).append(pair)
    assert not exact and not stems,(exact,stems)
    result=dict(checked_groups=48,unique_images=48,reference_sets=counts,
                exact_file_or_pixel_matches=exact,exact_question_matches=stems,
                near_candidates=near,near_counts=dict(Counter(x["source"] for x in near)),reference_hashes=manifests,
                interpretation="Perceptual hashes propose low-resolution resemblance, not identity; sources of all held-out scenes are newly generated labelled numeric data.")
    suffix="final" if a.train and a.general else "train" if a.train else "general" if a.general else "history"
    file=HERE/f"overlap_{suffix}.json"
    file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(file.name,"sets",len(data),"near",len(near),"sha",sha(file))


if __name__=="__main__":main()
