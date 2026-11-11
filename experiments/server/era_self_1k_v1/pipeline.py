"""Run the 1k matched pilot in order. A finished stage can be validated and resumed."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
TRAIN=Path.home()/"era_decoder_lora_20261030/.venv/bin/python"
EVAL=Path.home()/"mmdl_test/.venv/bin/python"
MAX_SHARDS=40


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def save(p,obj):
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(p)


def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def stage(name,interp,script,*args):
    print(f"{now()} {name}",flush=True)
    save(ROOT/"stage.json",dict(stage=name,start=now()))
    with (ROOT/"logs"/(name+".log")).open("a") as log:
        subprocess.run([str(interp),"-u",script,*map(str,args)],check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)


def inspect(index):
    raw=subprocess.check_output([str(EVAL),"-u","dataprep.py","inspect"],cwd=ROOT,text=True)
    report=json.loads(raw.strip().splitlines()[-1])
    assert type(report["ready"]) is bool
    save(ROOT/"logs"/f"screen-{index:03}.json",report)
    print(f"{now()} measured {json.dumps(report,ensure_ascii=False)[:1400]}",flush=True)
    return report


def main():
    subprocess.run(["sha256sum","--check","--quiet","release.ready"],cwd=ROOT,check=True)
    first=inspect(0)
    stage("recovery",TRAIN,"recovery.py")
    assert read(ROOT/"recovery/verified.json")["complete"]
    if not (ROOT/"anchors/receipt.json").exists() or not (ROOT/"holdout/receipt.json").exists():
        stage("shared-data",EVAL,"dataprep.py","fixed")
    for idx in range(MAX_SHARDS):
        shard=ROOT/"probe"/f"{idx:03}"
        if not (shard/"done.json").exists():
            if not (shard/"receipt.json").exists():
                stage(f"render-{idx:03}",EVAL,"dataprep.py","probe","--shard",idx)
            stage(f"probe-{idx:03}",EVAL,"infer.py","--probe",idx)
        assert read(shard/"done.json")["n"]==500
        info=first if idx==0 else inspect(idx)
        if info["ready"]:break
    else:
        raise RuntimeError("Reached the 20,000 genuine rollout budget without the required classes. No synthetic substitution.")
    stage("assemble",EVAL,"dataprep.py","assemble")
    freeze=read(ROOT/"freeze.json")
    assert freeze["train.jsonl"]==sha(ROOT/"train.jsonl")
    for arm in ("A","B"):
        stage("train-"+arm,TRAIN,"train.py",arm)
        stage("merge-"+arm,TRAIN,"train.py",arm,"--merge")
    mans=[read(ROOT/"models"/arm/"manifest.json") for arm in ("A","B")]
    assert all(m["complete"] and m["epoch"]==3 and m["actual_steps"]==189 for m in mans)
    for k in ("sequence_sha256","initial_adapter"):assert mans[0][k]==mans[1][k]
    save(ROOT/"train.ready",dict(completed=now(),steps=189,train_sha256=freeze["train.jsonl"]))
    for arm in ("start","priorcontrol","priorpaired","A","B"):
        stage("eval-"+arm,EVAL,"infer.py","--arm",arm)
        for mode in (("natural","common","self") if arm in ("start","A","B") else ("natural","common")):
            path=ROOT/"eval"/arm/mode
            assert read(path/"done.json")["output_sha256"]==sha(path/"predictions.jsonl")
    save(ROOT/"eval.ready",dict(completed=now(),arms=["start","priorcontrol","priorpaired","A","B"]))
    save(ROOT/"stage.json",dict(stage="evaluation_complete",time=now()))
    print(f"{now()} dependent analysis released",flush=True)


if __name__=="__main__":main()
