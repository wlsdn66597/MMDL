"""Compare R8 training pictures against archived sources and R7 inputs without predictions."""
import json
from collections import Counter
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"results/vic_era_2000_review/overlap"))
from check_overlap import load_refs, signature, fingerprint, match, qbody


def lines(p):return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))


def main():
    groups,manifest=load_refs()
    vic=ROOT/"results/vic_era_2000_review/overlap/vic_image_hashes.jsonl"
    groups["VIC2000"]=[(v["id"],v["normalized_question"],[(v["image"],v)]) for v in lines(vic)]
    for rnd,sub,file in (("era_er_round","data","rows.jsonl"),("era_er_round","paired","items.jsonl"),("era_er_round","general","items.jsonl")):
        base=ROOT/"results"/rnd;seen={}
        group=[]
        for r in lines(base/sub/file):
            path=base/r["image"]
            if path not in seen:seen[path]=signature(fingerprint(path))
            group.append((r["id"],qbody(r["question"]),[(r["image"],seen[path])]))
        groups[rnd+"_"+sub]=group
    rr=lines(HERE/"rows.jsonl")
    ours={r["group"]:signature(fingerprint(HERE.parent/r["image"])) for r in rr[::2]}
    ownq={r["id"]:qbody(r["question"]) for r in rr}
    exact=[];near=[];qs=[];tables={}
    for tag,records in groups.items():
        pictures={fp["file"]:(rid,path,fp) for rid,q,ls in records for path,fp in ls}
        questions={q:rid for rid,q,ls in records};tables[tag]=dict(rows=len(records),images=len(pictures))
        for q,rid in questions.items():
            for current,text in ownq.items():
                if q==text:qs.append(dict(current=current,source=tag,rid=rid))
        for k,fp in ours.items():
            for rid,path,z in pictures.values():
                m=match(fp,z)
                if m is not None:
                    record=dict(current=k,source=tag,rid=rid,other_image=path,**m)
                    (near if m["kind"]=="near" else exact).append(record)
    assert not exact and not qs,(exact,qs)
    internal=[]
    for n,(i,x) in enumerate(ours.items()):
        for j,y in list(ours.items())[n+1:]:
            m=match(x,y)
            if m:internal.append(dict(first=i,second=j,**m))
    assert all(x["kind"]=="near" for x in internal)
    out=dict(compared=tables,archive_source_count=len(groups),cross_exact_pixels_or_file=exact,cross_exact_question=qs,
             cross_perceptual_candidates=near,within_training_perceptual_candidates=internal,
             thresholds={"dHash_max_hamming":4,"pHash_max_hamming":6},
             interpretation="Within-template similarities are recorded separately; repeated layouts occur in independent training groups.")
    (HERE/"overlap.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(len(groups),len(exact),len(near),len(qs),len(internal))


if __name__=="__main__":main()
