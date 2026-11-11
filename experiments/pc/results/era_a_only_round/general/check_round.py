"""Compare held-out general image/question identities with R9 trained and paired data."""
from pathlib import Path
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
ROUND=HERE.parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/".runtime/era_review_deps"),str(ROOT/"results/era_midstep_round/data")]
from build_roster import normalized,fingerprint,signatures

def load(p):return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    current=HERE/"items.jsonl"
    gen=load(current)
    fprints={r["id"]:signatures(fingerprint(ROUND/r["image"])) for r in gen}
    out={}
    for name in ("data/rows.jsonl","paired/items.jsonl"):
        p=ROUND/name;pool=load(p);fps={r["image"]:signatures(fingerprint(ROUND/r["image"])) for r in pool}
        exact_qs={normalized(r["question"]) for r in pool}&{normalized(r["question"]) for r in gen}
        near=[]
        for rid,(pix,dh,ph) in fprints.items():
            for im,(pix2,dh2,ph2) in fps.items():
                dd=(dh^dh2).bit_count();pp=(ph^ph2).bit_count()
                if pix==pix2 or dd<=4 or pp<=6:near.append(dict(general=rid,input_image=im,same_rgb=pix==pix2,dhash=dd,phash=pp))
        out[name]=dict(sha256=sha(p),input_images=len(fps),input_items=len(pool),
                       near=near,same_questions=sorted(exact_qs))
    receipt=dict(n=len(fprints),general_sha256=sha(current),references=out)
    assert len(fprints)==96 and all(not x["near"] and not x["same_questions"] for x in out.values()),out
    (HERE/"round_overlap.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print({n:(len(x["near"]),len(x["same_questions"])) for n,x in out.items()})
if __name__=="__main__":main()
