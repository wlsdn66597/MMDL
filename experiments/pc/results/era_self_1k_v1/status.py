"""Read-only live KST estimates; main training, evaluation, and analysis are distinct."""
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import json
import math
import statistics

ROOT=Path(__file__).resolve().parent


def read(path,default=None):
    if not path.is_file():return default
    return json.loads(path.read_text(encoding="utf8"))


def jsonl(path):
    if not path.is_file():return []
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines() if l.strip()]


def calculate():
    now=datetime.now(ZoneInfo("Asia/Seoul"));folders=sorted((ROOT/"probe").glob("[0-9][0-9][0-9]"))
    n=0;elapsed=0
    for f in folders:
        records=jsonl(f/"predictions.jsonl");n+=len(records);elapsed+=sum(r["elapsed_s"] for r in records)
    screens=sorted((ROOT/"logs").glob("screen-*.json"))
    info=read(screens[-1],{}) if screens else {}
    completed=sum((f/"done.json").is_file() for f in folders)
    rate=elapsed/n if n else None
    stage=read(ROOT/"stage.json",{}).get("stage","probe-000")
    out=dict(time=now.isoformat(timespec="seconds"),stage=stage,rollouts=n,completed_probe_shards=completed,
             measured_batch_seconds_per_item=round(rate,3) if rate else None,screen=info,
             recovery_verified=(ROOT/"recovery/verified.json").is_file(),failure=bool((ROOT/"failed").exists()),
             trained=(ROOT/"train.ready").is_file(),eval_complete=(ROOT/"eval.ready").is_file(),
             analysis_complete=(ROOT/"analysis-job.done").is_file())
    if (ROOT/"failed").exists() or (ROOT/"analysis.failed").exists():
        out["eta_status"]="Paused on error; see failure markers and logs."
        return out
    if info.get("ready") or (ROOT/"dataset_receipt.json").is_file():target=n
    else:
        counts=info.get("counts",{})
        keep=counts.get("keep",0);wrong=counts.get("swap",counts.get("replace",0))
        if not isinstance(keep,int) or not isinstance(wrong,int) or not min(keep,wrong):target=None
        else:
            screened=info.get("complete_shards",completed)*500
            target=500*math.ceil(max(250*screened/keep,250*screened/wrong)/500)
    out["projected_rollouts"]=target
    if target is None or not rate:
        out["eta_status"]="Awaiting complete first-shard class counts."
        return out
    mine=max(0,target-n)*(rate+0.1)
    remaining_train=0
    for arm in ("A","B"):
        path=ROOT/"models"/arm
        if not (path/"merge.json").is_file():
            progress=read(path/"progress.json",{})
            if (path/"manifest.json").is_file():remaining_train+=25
            else:remaining_train+=float(progress.get("eta_seconds",progress.get("remaining_seconds",1050)))+60
    if not out["recovery_verified"]:remaining_train+=120
    measured=[]
    for file in ROOT.glob("eval/*/*/predictions.jsonl"):
        records=jsonl(file)
        if records:measured.extend(r["elapsed_s"] for r in records)
    throughput=statistics.mean(measured) if measured else rate*1.25
    left=0;reloads=0
    for arm in ("start","priorcontrol","priorpaired","A","B"):
        pending=False
        for mode,expected in [("natural",240),("common",120)]+([("self",180)] if arm in ("start","A","B") else []):
            path=ROOT/"eval"/arm/mode
            if (path/"done.json").exists():continue
            pending=True
            if mode=="self" and (path/"coverage.json").exists():expected=sum(r["draft"] is not None for r in read(path/"coverage.json"))
            left+=max(0,expected-len(jsonl(path/"predictions.jsonl")))
        reloads+=pending
    work=mine+remaining_train
    def window(seconds):
        return [(now+timedelta(seconds=seconds*scale)).isoformat(timespec="minutes") for scale in (.80,1.30)]
    out.update(estimated_training_finish_kst=window(work) if not out["trained"] else read(ROOT/"train.ready")["completed"],
               estimated_autoanalysis_finish_kst=window(work+left*throughput+reloads*25+120) if not out["analysis_complete"] else "completed",
               pending_eval_calls_upper=left,estimate_method="Observed per-item rollout wall time; previous 3-epoch 1k train estimate until step telemetry. Range is an engineering margin, not a statistical interval.")
    return out


if __name__=="__main__":print(json.dumps(calculate(),ensure_ascii=False,indent=2))
