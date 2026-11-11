from score import *
import zipfile
def main():
    receipt=read(R/"server_receipt.json")
    for name,h in receipt.items():assert sha(R/name)==h,name
    ms=[read(R/"models"/a/"manifest.json") for a in ARMS[1:]]
    assert ms[0]["initial_adapter"]==ms[1]["initial_adapter"] and ms[0]["sequence_sha256"]==ms[1]["sequence_sha256"]
    inputs={sp:read(R/sp/"gold.json") for sp in ("train","holdout")}
    assert not ({i["image_sha256"] for i in inputs["train"]}&{i["image_sha256"] for i in inputs["holdout"]})
    for sp,items in inputs.items():
        for it in items:
            assert sha(R/sp/it["image"])==it["image_sha256"]
            if it["mode"]=="direct":v=D(it["rows"][it["direct_name"]])
            else:v=sum(sorted(map(D,it["rows"].values()))[:3])
            assert v==D(it["truth"])==D(it["options"]["ABCD".index(it["letter"])])
    trainmap={i["id"]:i for i in inputs["train"]}
    n=0
    for line in (R/"paired_train.jsonl").read_text(encoding="utf8").splitlines():
        row=json.loads(line); n+=1
        assert row["statement"].split("\n")[1:]==row["question"].split("\n")[1:]
        for arm in ARMS[1:]:
            it=trainmap[row["sample_id"]];s=assess(it,dict(raw=row[arm],finish_reason="stop",output_tokens=0))
            assert all(s[k] for k in ("answer","E","R","A"))
            if row["action"]=="replace":assert s["repair"]
        expected=input_prompt(trainmap[row["sample_id"]])
        assert row["input_prompt"]==expected
    assert n==read(R/"target_receipt.json")["included"]
    rows=read(R/"scores.json");x={(r["arm"],r["id"]):r for r in rows};assert len(rows)==len(x)==216
    judge=read(R/"judge.json");res=[]
    for j in judge:
        rec=x[j["arm"],j["id"]]; pj=j["parsed"]
        if not isinstance(pj,dict) or pj.get("joint") not in (0,1) or pj.get("repair") not in (0,1):j["valid"]=False;res.append(j);continue
        refrep=(rec["draft_supported"] if j["id"].endswith("dreplace") else False if j["id"].endswith("dkeep") else rec["repair"])
        j.update(valid=True,ref_joint=rec["joint"],ref_repair=refrep,agree_joint=bool(pj["joint"])==rec["joint"],agree_repair=bool(pj["repair"])==refrep)
        res.append(j)
    assert len(res)==24
    save(R/"judge_comparison.json",res)
    print("Judge: valid",sum(r["valid"] for r in res), "joint agreement",sum(r.get("agree_joint",False) for r in res),"repair agreement",sum(r.get("agree_repair",False) for r in res))
    files=["protocol.md","report.md","review.html","scores.csv","scores.json","metrics.json","paired_changes.json",
           "target_receipt.json","paired_train.jsonl","statement.jsonl","question.jsonl","draft_audit.json","freeze.json",
           "adjudication.json","judge.json","judge_comparison.json","server_receipt.json","build.py","targets.py","score.py","score_transfer.py","train.py","campaign.sh","judge.py","eval_config.json","eval.py",
           "build_next_pairs.py","collect.py","review.py","verify.py","transfer.py",
           "models/statement/manifest.json","models/question/manifest.json","models/matched_init.json"]
    files.extend(f for f in ("transfer_metrics.json","transfer_scores.json","transfer_adjudication.json") if (R/f).exists())
    arc=R.parent/"era_link_repair_pilot_review.zip"
    with zipfile.ZipFile(arc,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(R/f,f)
        for folder in ("train","holdout","eval","transfer","next_train"):
            for f in sorted((R/folder).rglob("*")):
                if f.is_file():z.write(f,f.relative_to(R))
    with zipfile.ZipFile(arc) as z:assert z.testzip() is None
def input_prompt(it):
    return "Question: "+it["question"]+"\nOptions:\n"+"".join(f"{chr(i+65)}. {s}\n" for i,s in enumerate(it["options"]))+'Please select the correct answer from the options above.\nOn the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'
if __name__=="__main__":main()
