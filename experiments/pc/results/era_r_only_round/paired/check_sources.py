"""Compare held-out source pixels and normalized questions to protected reference groups."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"results/vic_era_2000_review/overlap"))
from check_overlap import fingerprint, load_refs, match, qbody, signature
from verify import read,sha


def register(data,manifests,name,file,base):
    pics={};records=[]
    for r in read(file):
        rel=r["image"]
        if rel not in pics:pics[rel]=signature(fingerprint(base/rel))
        records.append((r["id"],qbody(r["question"]),[(rel,pics[rel])]))
    data[name]=records;manifests[name]={file.relative_to(ROOT).as_posix():sha(file)}


def main():
    p=ArgumentParser();p.add_argument("--train",type=Path);p.add_argument("--general",type=Path)
    arg=p.parse_args()
    data,manifests=load_refs()
    vic=ROOT/"results/vic_era_2000_review/overlap/vic_image_hashes.jsonl"
    data["VIC2000"]=[(v["id"],v["normalized_question"],[(v["image"],v)]) for v in read(vic)]
    manifests["VIC2000"]={vic.relative_to(ROOT).as_posix():sha(vic)}
    r7=ROOT/"results/era_er_round"
    for name,rel in (("round7_train","data/rows.jsonl"),("round7_pairs","paired/items.jsonl"),("round7_general","general/items.jsonl")):
        register(data,manifests,name,r7/rel,r7)
    for name,file in (("round8_train",arg.train),("round8_general",arg.general)):
        if file:register(data,manifests,name,file.resolve(),HERE.parent)
    cards=read(HERE/"items.jsonl")
    queries={c["id"]:qbody(c["question"]) for c in cards}
    fingers={c["group"]:signature(fingerprint(HERE.parent/c["image"])) for c in cards[::2]}
    exact=[];near=[];qdup=[];counts={}
    for name,records in data.items():
        pics={};texts={}
        for rid,q,ims in records:
            texts.setdefault(q,rid)
            for rel,f in ims:pics.setdefault(f["pixel"],(rid,rel,f))
        counts[name]=dict(records=len(records),unique_images=len(pics))
        qdup.extend(dict(id=i,source=name,ref=texts[q]) for i,q in queries.items() if q in texts)
        for g,f in fingers.items():
            for rid,rel,fy in pics.values():
                hit=match(f,fy)
                if hit:
                    (near if hit["kind"]=="near" else exact).append(dict(group=g,ref=rid,image=rel,source=name,**hit))
    intern=[]
    groups=list(fingers.keys())
    for i,g in enumerate(groups):
        for j in groups[i+1:]:
            hit=match(fingers[g],fingers[j])
            if hit:intern.append(dict(group1=g,group2=j,**hit))
    assert not exact and not qdup,(exact,qdup)
    assert all(h["kind"]=="near" for h in intern)
    out=dict(n_groups=48,items_sha256=sha(HERE/"items.jsonl"),sources_sha256=sha(HERE/"sources.jsonl"),
             source_receipt_sha256=sha(HERE/"receipt.json"),reference_sets=counts,reference_hashes=manifests,
             exact_file_or_pixel_matches=exact,exact_question_matches=qdup,near_candidates=near,near_counts=dict(Counter(x["source"] for x in near)),
             internal_low_resolution_candidates=intern,internal_near_counts_by_family=dict(Counter((x["group1"].split("-")[1]) for x in intern)),
             note="48 distinct source images; layouts recur within a rule family, so visual similarity within this evaluation set is reported separately.")
    suffix="final" if arg.train and arg.general else "train" if arg.train else "general" if arg.general else "history"
    file=HERE/f"overlap_{suffix}.json";file.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(file.name,"external",len(near),"internal",len(intern),"sha",sha(file))


if __name__=="__main__":main()
