"""Extend the frozen diagnostic overlap check to the donor 200 and shared prior 449."""
from pathlib import Path
import json
import re
import sys

from generate import OUT, ROOT, sha, write
sys.path.insert(0,str(ROOT/"scripts/visual_groups"))
from visual_groups import fingerprint, is_candidate


INPUT_SHA="f1be9e728a836bfecf784451c5b4dbcd5092d4b9519498ece1b9425d7ed16233"
GOLD_SHA="0171459c7599cfadb788f8c3950db89b84ff80f7c16058e9126dbd3135fe3cda"


def norm(q):
    return " ".join(re.findall(r"\w+|[%+−-]",q.lower()))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def donor_rows():
    root=ROOT/"code/training/era_decoder_lora/data/pilot200"
    path=root/"exports/train_era.jsonl"
    archive=read(root/"SHA256SUMS.json")
    assert sha(path)==archive["exports/train_era.jsonl"]=="6c19160e0a60b568854a8377d330a04300ffab52484d6143da4b3f12aa1afa53"
    rows=[]
    for line in path.read_text(encoding="utf8").splitlines():
        row=json.loads(line)
        question=re.search(r"(?:^|\n)Question:\s*(.*?)\nOptions:",row["input_prompt"],flags=re.DOTALL)
        assert question and question[1].strip()
        image=row["image_path"]
        rows.append((row["sample_id"],question[1].strip(),root/image,archive[image]))
    return path,rows


def previous_rows():
    root=ROOT/"results/era_single_call_campaign/data"
    path=root/"train.jsonl"
    assert sha(path)==read(root/"receipt.json")["splits"]["train"]["sha256"]
    rows=[]
    for line in path.read_text(encoding="utf8").splitlines():
        r=json.loads(line)
        rows.append((r["id"],r["question"],root/r["image"],r["image_sha256"]))
    return path,rows


def compare(path,rows,inputs,fp_test,cache):
    question_hits=[]
    image_hits=[]
    input_questions={norm(r["question"]) for r in inputs}
    for rid,q,image,expected in rows:
        if norm(q) in input_questions:
            question_hits.append(rid)
        assert sha(image)==expected
        if expected not in cache:
            cache[expected]=fingerprint(image)
        fp=cache[expected]
        for diagnostic,test_fp in fp_test.items():
            if fp["pixel_sha256"]==test_fp["pixel_sha256"] or is_candidate(fp,test_fp):
                image_hits.append([rid,diagnostic])
    return dict(train_sha256=sha(path), rows=len(rows), unique_ids=len({r[0] for r in rows}),
                unique_images=len({r[-1] for r in rows}), image_candidates=image_hits, normalized_question_hits=question_hits)


def main():
    assert sha(OUT/"inputs.json")==INPUT_SHA and sha(OUT/"gold.json")==GOLD_SHA
    items=read(OUT/"inputs.json")
    unique={r["image"]:r["image_sha256"] for r in items}
    fps={key:fingerprint(OUT/key) for key in unique}
    assert all(fps[k]["file_sha256"]==v for k,v in unique.items())
    out=dict(input_sha256=INPUT_SHA,gold_sha256=GOLD_SHA,diagnostic_images=len(fps),
             matching_rule="Identical normalized question or exact RGBA pixel fingerprint or dHash <=4 or pHash <=6.")
    cache={}
    for name,source,n in (("donor200",donor_rows,200),("prior449",previous_rows,449)):
        path,rows=source()
        assert len(rows)==n and len({r[0] for r in rows})==n
        out[name]=compare(path,rows,items,fps,cache)
    write(OUT/"train_overlap.json",out)
    print(json.dumps(out))


if __name__=="__main__":
    main()
