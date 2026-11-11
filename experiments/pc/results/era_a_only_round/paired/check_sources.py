"""Screen this held-out roster against the project's pinned question and visual registries."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

from verify import rows,digest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"results/vic_era_2000_review/overlap"))
from check_overlap import fingerprint,load_refs,match,qbody,signature


def enroll(refs,manifests,tag,file,base):
    prints={};records=[]
    for r in rows(file):
        image=r.get("image") or r["image_path"]
        if image not in prints:
            prints[image]=signature(fingerprint(base/image))
        if "input_prompt" in r:
            questions=[r["input_prompt"]]
        elif "question" in r:
            questions=[r["question"]]
        else:
            questions=[r["control_prompt"],r["reverse_prompt"]]
        for q in questions:
            records.append((r.get("id") or r["sample_id"],qbody(q),[(image,prints[image])]))
    refs[tag]=records
    manifests[tag]={file.relative_to(ROOT).as_posix():digest(file)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--live",action="store_true",help="Include the R9 training and general manifests after release")
    args=parser.parse_args()
    refs,manifests=load_refs()
    vic=ROOT/"results/vic_era_2000_review/overlap/vic_image_hashes.jsonl"
    refs["VIC2000"]=[(r["id"],r["normalized_question"],[(r["image"],r)]) for r in rows(vic)]
    manifests["VIC2000"]={vic.relative_to(ROOT).as_posix():digest(vic)}
    direction=ROOT/"results/vic_direction_round"
    enroll(refs,manifests,"round6_direction",direction/"data/direction_pairs.jsonl",direction/"data")
    for tag,dirname in (("round7","era_er_round"),("round8","era_r_only_round")):
        rnd=ROOT/"results"/dirname
        for ds,file in [("train","data/rows.jsonl"),("paired","paired/items.jsonl"),("general","general/items.jsonl")]:
            enroll(refs,manifests,f"{tag}_{ds}",rnd/file,rnd)
    if args.live:
        for ds,file in [("train","data/rows.jsonl"),("general","general/items.jsonl")]:
            enroll(refs,manifests,f"round9_{ds}",HERE.parent/file,HERE.parent)
        enroll(refs,manifests,"round9_full_train",HERE.parent/"train/manifests/ordinary.jsonl",ROOT)
    cards=rows(HERE/"items.jsonl")
    questions={r["id"]:qbody(r["question"]) for r in cards}
    prints={r["group"]:signature(fingerprint(HERE.parent/r["image"])) for r in cards[::2]}
    exact=[]; near=[]; duplicate=[]; counts={}
    for tag,pool in refs.items():
        text={q:rid for rid,q,_ in pool}
        pictures={f["pixel"]:(rid,filename,f) for rid,_,files in pool for filename,f in files}
        counts[tag]={"records":len(pool),"unique_images":len(pictures)}
        duplicate.extend([i,tag,text[q]] for i,q in questions.items() if q in text)
        for g,new in prints.items():
            for rid,path,prev in pictures.values():
                hit=match(new,prev)
                if hit:
                    (near if hit["kind"]=="near" else exact).append(dict(group=g,source=tag,id=rid,image=path,**hit))
    inner=[]
    for (i,x),(j,y) in combinations(prints.items(),2):
        hit=match(x,y)
        if hit:
            assert hit["kind"]=="near"
            inner.append(dict(left=i,right=j,**hit))
    assert not exact and not duplicate,(exact,duplicate)
    report=dict(n_groups=48,items_sha256=digest(HERE/"items.jsonl"),sources_sha256=digest(HERE/"sources.jsonl"),
                source_receipt_sha256=digest(HERE/"receipt.json"),reference_sets=counts,reference_hashes=manifests,
                exact_file_or_pixel_matches=exact,exact_question_matches=duplicate,
                near_candidates=near,near_counts=dict(Counter(h["source"] for h in near)),internal_low_resolution_candidates=inner,
                note="Perceptual candidates are triage signals. Internal groups have intentionally repeated layout families.")
    dest="overlap_final.json" if args.live else "overlap_history.json"
    (HERE/dest).write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+"\n").encode())
    print(dest,"references",len(refs),"external near",len(near),"internal near",len(inner))


if __name__=="__main__":
    main()
