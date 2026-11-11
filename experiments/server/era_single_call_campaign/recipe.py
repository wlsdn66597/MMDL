"""Freeze non-MMMU ERA examples and their distinct-source connection-validation partition."""
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
DATA=HERE/"data"
PILOT=REPO/"code/training/era_decoder_lora/data/pilot200"
ORIGINAL=REPO/"results/era_pilot_200_20261028_r2"
DSET=REPO/"results/d226_echeck_20261105/data"
REF=REPO/"reports/era_abcd_review_20261020/mmmu_reference_fingerprints.json"
OFFICIAL=REPO/"results/mmmu_first_baseline_20261009/baseline_seed3407/results.json"
HEAD=re.compile(r"^Evidence: (.+?)\nRule: (.+?)\nApplication: (.+?)\nFinal answer: (.+?)$",re.S)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,content):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(content,encoding="utf8")
    tmp.replace(path)


def normalize(text):
    return " ".join(re.findall(r"\w+|[%+−-]",text.casefold()))


def sections(target):
    m=HEAD.fullmatch(target.strip())
    assert m and all(v.strip() for v in m.groups()),target
    return m.groups()


def clean_open(app):
    app=re.sub(r"(?i)[,;]?\s*(?:so\s+)?(?:(?:matching|matches|match|select(?:ing)?|choose|corresponding to)\s+)?(?:answer\s+)?option\s+[A-Z](?=[\s.,;)]|$)","",app)
    if re.search(r"(?i)\boption\b|\bchoices?\b|select\s+[A-Z]\b",app):return None
    app=re.sub(r"\s*[.,;]+\s*$",".",app.strip())
    return app if any(c.isalpha() for c in app) else None


def prepare():
    sys.path.insert(0,str(REPO/"scripts/visual_groups"))
    from visual_groups import fingerprint
    sums=read(PILOT/"SHA256SUMS.json")
    assert sha(PILOT/"exports/train_era.jsonl")==sums["exports/train_era.jsonl"]
    assert sha(OFFICIAL)==read(ORIGINAL/"overlap_report.json")["reference_sha256"]
    questions={normalize(x["question"]) for x in read(OFFICIAL)["records"]}
    refs=read(REF)
    assert len(refs)==982 and len(read(OFFICIAL)["records"])==900
    image_checks={}
    rejected={}
    rows=[]
    dest=DATA/"images";dest.mkdir(exist_ok=True,parents=True)
    def add(base,src):
        original=src.resolve();digest=sha(original)
        if digest not in image_checks:
            fp=fingerprint(original)
            hits=[x["id"] for x in refs if x["pixel_sha256"]==fp["pixel_sha256"] or
                  (int(x["dhash"],16)^int(fp["dhash"],16)).bit_count()<=4 or
                  (int(x["phash"],16)^int(fp["phash"],16)).bit_count()<=6]
            image_checks[digest]=(fp["pixel_sha256"],hits)
        pixel,hits=image_checks[digest]
        if hits or normalize(base["question"]) in questions:
            rejected[base["id"]]=dict(mmmu_image_ids=hits,question_overlap=normalize(base["question"]) in questions)
            return
        file=dest/(digest+original.suffix.lower())
        if not file.exists():shutil.copyfile(original,file)
        assert sha(file)==digest
        e,r,a,ans=sections(base["target"])
        rows.append(dict(**base,image=file.relative_to(DATA).as_posix(),image_sha256=digest,pixel_sha256=pixel,
                         evidence=e,rule=r,application=a,answer=ans))
    source={x["sample_id"]:x for x in lines(ORIGINAL/"source_records.jsonl")}
    for x in lines(PILOT/"exports/train_era.jsonl"):
        orig=source[x["sample_id"]]
        assert orig["dataset"] in ("ChartQA","ScienceQA") and orig["image_path"].startswith("train/")
        pic=PILOT/x["image_path"]
        assert sha(pic)==sums[x["image_path"]]
        base=dict(id=x["sample_id"],dataset=orig["dataset"],group=x["visual_group_id"],source_id=orig["source_id"],
                  source_revision=orig["source_revision"],question=orig["question"],prompt=x["input_prompt"],target=x["response"],kind="mc")
        add(base,pic)
        if orig["dataset"]=="ChartQA":
            e,r,a,letter=sections(x["response"])
            assert re.search(rf"(?m)^{letter}\. .+$",x["input_prompt"])
            answer=str(orig["original_answer"]).strip();app=clean_open(a)
            if not app or re.search(r"(?i)\b(?:option|choice)\b",e+" "+r):continue
            prompt="Question: "+orig["question"]+'\nOn the last line, write "Final answer: ANSWER", replacing ANSWER with a concise final answer.'
            target=f"Evidence: {e}\nRule: {r}\nApplication: {app}\nFinal answer: {answer}"
            add(dict(base,id=base["id"]+"_open",prompt=prompt,target=target,kind="open"),pic)
    for x in lines(DSET/"standard.jsonl"):
        assert len(x["images"])==1 and x["source_group"] and x["source_revision"] and x["source_url"]
        assert hashlib.sha256(x["target"].encode()).hexdigest()==x["target_sha256"]
        question=x["input_prompt"].split("Question: ",1)[1].split("\nOptions:",1)[0]
        base=dict(id=x["id"],dataset="D/"+x["source_slug"],group=x["source_group"],source_id=x["id"],
                  source_revision=x["source_revision"],source_url=x["source_url"],
                  question=question,prompt=x["input_prompt"],target=x["target"],kind="mc")
        add(base,DSET/x["images"][0])
    # Connect groups sharing a pixel-identical image; an image or its variants may occur in only one split.
    parent={}
    def find(k):
        parent.setdefault(k,k)
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    for x in rows:
        a,b=find("g/"+x["group"]),find("p/"+x["pixel_sha256"])
        parent[max(a,b)]=min(a,b)
    train=[];dev=[]
    for x in rows:
        group=find("g/"+x["group"])
        x["source_partition_group"]=group
        bucket=int(hashlib.sha256(("campaign-1819/"+group).encode()).hexdigest()[:8],16)%6
        (dev if bucket==0 else train).append(x)
    assert len(rows)==len({x["id"] for x in rows}) and train and dev
    assert not {x["pixel_sha256"] for x in train}&{x["pixel_sha256"] for x in dev}
    assert not {x["source_partition_group"] for x in train}&{x["source_partition_group"] for x in dev}
    rng=random.Random(3407)
    summary={}
    for name,items in (("train",train),("dev",dev)):
        rng.shuffle(items)
        path=DATA/(name+".jsonl")
        write(path,"".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
        summary[name]=dict(n=len(items),sha256=sha(path),sources=dict(Counter(x["dataset"] for x in items)),
                           kinds=dict(Counter(x["kind"] for x in items)),
                           independent_groups=len({x["source_partition_group"] for x in items}))
    receipt=dict(splits=summary,rejected=rejected,mmmu_reference_questions=900,mmmu_reference_images=982,
                 fingerprint_source_sha256=sha(REF),reference_sha256=sha(OFFICIAL),
                 source_files={str(x.relative_to(REPO)):sha(x) for x in (ORIGINAL/"source_records.jsonl",DSET/"standard.jsonl",PILOT/"exports/train_era.jsonl")},
                 recipe_sha256=sha(Path(__file__)),overlap_rule="Exclude exact normalized question or pixel and dHash<=4 or pHash<=6 matches. Source-level provenance checked; semantic overlap/pretraining exposure is not ruled out.")
    write(DATA/"receipt.json",json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(splits=summary,rejections=len(rejected)),ensure_ascii=False))


if __name__=="__main__":prepare()
