"""Independent, read-only package checks. Standard library; no model/API calls."""
import argparse
import hashlib
import json
import re
import string
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rows(p):return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def objsha(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def check(root,workspace=None):
    root=root.resolve()
    paired=rows(root/"paired_train.draft.jsonl")
    plain=rows(root/"train_plain.draft.jsonl");era=rows(root/"train_era.draft.jsonl")
    source={r["sample_id"]:r for r in rows(root/"source_records.jsonl")}
    original={r["sample_id"]:r for owner in "ABCD" for r in rows(root/f"provenance/submissions/{owner}.jsonl")}
    edits=read(root/"provenance/content_revisions.json")
    checklist=rows(root/"cross_review_checklist.jsonl")
    registry=read(root/"visual_group_registry.json")
    images={r["sample_id"]:r for r in registry["images"]}
    config=read(root/"provenance/baseline_config.json")
    assert len(paired)==len(plain)==len(era)==len(source)==len(checklist)==len(images)==200
    assert Counter(r["owner"] for r in paired)==dict.fromkeys("ABCD",50)
    assert len({(r["source_dataset"],r["source_id"]) for r in paired})==200
    assert len({r["original_image_path"] for r in paired})==200
    assert {r["sample_id"] for r in paired}==(set(original)-{"C_1003","C_10413"})|{"C_10598","C_10611"}
    changed=[];unaltered=0
    for p,a,b,c in zip(paired,plain,era,checklist):
        sid=p["sample_id"];s=source[sid]
        assert sid==a["sample_id"]==b["sample_id"]==c["sample_id"]
        assert c["record_sha256"]==objsha(p)
        assert c["status"]=="not_started" and all(c[k]=="" for k in ["verdict","reviewer_name","reviewed_at"])
        assert p["quality_status"]=="draft" and p["reviewer_status"]=="not_started"
        assert p["review_checks"]["final_acceptance"] is False
        assert p["reviewer"]=={"A":"B","B":"A","C":"D","D":"C"}[p["owner"]]
        assert p["question"]==s["question"]==p["original_question"]
        assert p["hint"]==s["hint"]
        assert p["original_answer"]==s["original_answer"]
        assert p["image_group_no"]==s["image_group_no"]
        assert p["source_revision"]==s["source_revision"]
        assert p["source_fingerprint"]==s["source_fingerprint"]
        assert p["image_group_no"]%2==(1 if p["owner"] in "AC" else 0)
        if p["source_dataset"]=="ScienceQA":
            assert p["options"]==s["original_options"]==p["original_options"]
            assert p["answer_letter"]==string.ascii_uppercase[s["original_answer"]]
            assert p["answer_value"]==s["original_options"][s["original_answer"]]
        else:
            old=original[sid]
            assert p["options"]==(list(old["options"].values()) if isinstance(old["options"],dict) else old["options"])
            assert p["answer_letter"]==old["answer_letter"] and p["answer_value"]==old["answer_value"]
        path=(root/p["image_path"]).resolve()
        assert path.is_relative_to(root) and path.is_file()
        assert sha(path)==p["image_sha256"]==images[sid]["file_sha256"]
        assert p["visual_group_id"]==images[sid]["visual_group_id"]
        assert p["visual_group_registry_id"]==registry["registry_id"]
        hint=config["prompts"]["optional_hint"].format(hint=p["hint"]) if p["hint"] else ""
        letters="".join(f"{k}. {v}\n" for k,v in zip(string.ascii_uppercase,p["options"]))
        expected=config["prompts"]["multiple-choice"].format(optional_hint=hint,question=p["question"],lettered_options=letters).rstrip()
        expected+="\n"+config["prompts"]["suffixes"]["multiple-choice"]
        assert p["input_prompt"]==expected
        expected_user={"role":"user","content":[{"type":"image","path":p["image_path"]},{"type":"text","text":expected}]}
        assert a["messages"][0]==b["messages"][0]==expected_user
        assert len(a["messages"])==len(b["messages"])==2
        for out,field in [(a,"response_general"),(b,"response_era")]:
            assert out["messages"][1]=={"role":"assistant","content":p[field]}
            assert re.findall(r"(?m)^Final answer: ([A-Z])$",p[field])==[p["answer_letter"]]
        assert re.findall(r"(?m)^(Evidence|Rule|Application):",p["response_era"])==["Evidence","Rule","Application"]
        assert not re.search(r"(?m)^(Evidence|Rule|Application):",p["response_general"])
        if sid in edits:
            for unit in p["semantic_units"]:
                assert unit["general_excerpt"] in p["response_general"]
                assert unit["era_excerpt"] in p["response_era"]
            if sid in original:
                assert p["response_general"]!=original[sid]["response_general"]
                assert p["response_era"]!=original[sid]["response_era"]
                changed.append(sid)
        else:
            assert p["response_general"]==original[sid]["response_general"]
            assert p["response_era"]==original[sid]["response_era"]
            unaltered+=1
    assert len(changed)==9 and unaltered==189
    assert len({p["image_sha256"] for p in paired})==200
    assert len({p["image_pixel_sha256"] for p in paired})==200
    group_members=[s for g in registry["groups"] for s in g["member_ids"]]
    assert set(group_members)==set(images) and len(group_members)==200
    assert len(registry["groups"])==189 and registry["unresolved_pairs"]==0
    for d in registry["decisions"]:
        assert d["left_sha256"]==images[d["left"]]["file_sha256"]
        assert d["right_sha256"]==images[d["right"]]["file_sha256"]
        assert d["verdict"] in {"same","different"}
        if d["verdict"]=="same":assert images[d["left"]]["visual_group_id"]==images[d["right"]]["visual_group_id"]
        else:assert images[d["left"]]["visual_group_id"]!=images[d["right"]]["visual_group_id"]
    near=[]
    image_values=list(images.values())
    for idx,x in enumerate(image_values):
        for y in image_values[idx+1:]:
            dh=(int(x["dhash"],16)^int(y["dhash"],16)).bit_count()
            ph=(int(x["phash"],16)^int(y["phash"],16)).bit_count()
            if dh<=4 or ph<=6 or x["pixel_sha256"]==y["pixel_sha256"]:
                near.append(tuple(sorted((x["sample_id"],y["sample_id"]))))
    assert set(near)=={tuple(sorted((p["left"],p["right"]))) for p in registry["candidates"]}
    assert len(near)==470 and all(p["status"] in {"same","different"} for p in registry["candidates"])
    # Every non-singleton family is explicitly visible to the next reviewers.
    repeated=read(root/"repeated_visual_families.json")
    assert repeated["final_retention"]=="pending_human_cross_review"
    assert len(repeated["groups"])==9 and sum(len(g["member_ids"]) for g in repeated["groups"])==20
    tokens=read(root/"token_report.json")
    assert len(tokens["records"])==200 and {r["sample_id"] for r in tokens["exceptions"]}=={"B-chartqa-006"}
    for k in ["general","era","general_body","era_body"]:assert tokens["totals"][k]==sum(r[k] for r in tokens["records"])
    assert max(max(r["general"],r["era"]) for r in tokens["records"])<=512
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[];self.images=0;self.articles=0
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=="img":self.images+=1
            if tag=="article":self.articles+=1
            for k in ["src","href"]:
                if k in a:self.links.append(a[k])
    parser=Links();parser.feed((root/"review.html").read_text(encoding="utf-8"))
    assert parser.images==parser.articles==200
    for link in parser.links:
        if not link.startswith(("http:","https:","#")):assert (root/link.split("#")[0]).is_file(),link
    workspace_checks="not_run_no_workspace_argument"
    if workspace:
        official=workspace/"reports/era_dataset_examples_20261014"
        lock=read(root/"provenance/source_lock.json")
        for name,entry in lock.items():assert sha(official/name)==entry["sha256"]
        science=read(official/"scienceqa_problems.json");splits=read(official/"scienceqa_splits.json")
        chart=read(official/"chartqa_train_human.json");train={str(x) for x in splits["train"]}
        for p in paired:
            if p["source_dataset"]=="ScienceQA":
                s=science[p["source_id"]]
                assert p["source_id"] in train
                assert (p["question"],p["hint"],p["options"],p["original_answer"])==(s["question"],s["hint"],s["choices"],s["answer"])
            else:
                s=chart[int(p["source_id"].split("_")[-1])]
                assert (p["question"],p["original_answer"],p["original_image_path"])==(s["query"],s["label"],"train/png/"+s["imgname"])
        for entry in read(root/"provenance/archive_identities.json").values():
            assert sha(Path(entry["path"]))==entry["sha256"]
            assert sha(Path(entry["manifest"]))==entry["manifest_sha256"]
        workspace_checks="passed_locked_official_sources_and_original_archives_unchanged"
    package_hashes=root/"PACKAGE_SHA256.json"
    integrity="manifest_not_created_yet"
    if package_hashes.exists():
        files=read(package_hashes)
        for name,d in files.items():assert sha(root/name)==d,name
        actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name!="PACKAGE_SHA256.json"}
        assert set(files)==actual
        integrity="passed_all_files"
    return dict(status="passed",records=200,images=200,input_pairs_identical=200,existing_pairs_revised=9,
        new_pairs=2,existing_response_pairs_unchanged=189,candidate_pairs_checked=470,
        human_verdicts_prefilled=0,html_images_and_articles=200,broken_html_links=0,
        workspace_source_checks=workspace_checks,package_integrity=integrity,
        semantic_correctness="not_claimed_by_this_structural_validator")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("package",type=Path)
    parser.add_argument("--workspace",type=Path)
    args=parser.parse_args()
    print(json.dumps(check(args.package,args.workspace),ensure_ascii=False,indent=2))
