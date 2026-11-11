"""Check R9 training pictures and question text against archived training/evaluation inputs."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"results/vic_era_2000_review/overlap"))
from check_overlap import load_refs, match, signature, fingerprint, qbody


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s]


def archive_inputs():
    refs,_=load_refs()
    vic=lines(ROOT/"results/vic_era_2000_review/overlap/vic_image_hashes.jsonl")
    refs["VIC2000"]=[(v["id"],v["normalized_question"],[(v["image"],v)]) for v in vic]
    for rnd in ("era_er_round","era_r_only_round"):
        folder=ROOT/"results"/rnd
        for path in ("data/rows.jsonl","paired/items.jsonl","general/items.jsonl"):
            records=[];previous={}
            for row in lines(folder/path):
                img=folder/row["image"]
                if img not in previous:previous[img]=signature(fingerprint(img))
                records.append((row["id"],qbody(row["question"]),[(row["image"],previous[img])]))
            refs[f"{rnd}/{path}"]=records
    return refs


def main():
    records=lines(HERE/"rows.jsonl")
    imgs={r["group"]:signature(fingerprint(HERE.parent/r["image"])) for r in records[::2]}
    qs={qbody(r["question"]):r["id"] for r in records}
    assert len(qs)==len(records)==48
    exact=[];near=[];same_q=[];tables={}
    for dataset,rows in archive_inputs().items():
        unique_images={h["file"]:(ident,img,h) for ident,q,pics in rows for img,h in pics}
        tables[dataset]=dict(rows=len(rows),images=len(unique_images))
        for ident,q,_ in rows:
            if q in qs:same_q.append([qs[q],dataset,ident])
        for img_id,h in imgs.items():
            for ident,img,other in unique_images.values():
                finding=match(h,other)
                if finding:
                    (near if finding["kind"]=="near" else exact).append(dict(group=img_id,dataset=dataset,id=ident,image=img,**finding))
    internal=[]
    for (a,x),(b,y) in combinations(imgs.items(),2):
        finding=match(x,y)
        if finding:internal.append(dict(first=a,second=b,**finding))
    assert not exact and not same_q and all(z["kind"]=="near" for z in internal)
    out=dict(compared=tables,cross_exact_pixels_or_file=exact,cross_exact_question=same_q,
             cross_perceptual_candidates=near,within_training_perceptual_candidates=internal,
             thresholds={"dHash_max_hamming":4,"pHash_max_hamming":6})
    (HERE/"overlap.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(len(tables),len(exact),len(same_q),len(near),dict(Counter(z["dataset"] for z in near)),len(internal))


if __name__=="__main__":
    main()
