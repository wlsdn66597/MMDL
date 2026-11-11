"""Publish verifiable progress and automatically package the final result on the server."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from zoneinfo import ZoneInfo

from report_results import ROOT, digest, load, store

KST = ZoneInfo("Asia/Seoul")
RUN = ROOT/"mmmu/ordinary50"


def complete():
    subprocess.run([sys.executable, str(ROOT/"report_results.py")],cwd=ROOT,check=True)
    names = ["report.md","verification.json","transitions.json","started.json","status.json",
             "mmmu/roster.json","mmmu/nominee.json","mmmu/ordinary50/manifest.json",
             "mmmu/ordinary50/predictions.jsonl","scale_receipts/ordinary.json","steps"]
    for f in ["judge_attempts.jsonl","judge_identity.json","screen","full"]:
        if (RUN/f).exists():
            names.append("mmmu/ordinary50/"+f)
    tmp=ROOT/"final.building"
    with tarfile.open(tmp,"w:gz",compresslevel=2) as archive:
        for name in names:
            archive.add(ROOT/name,arcname=name)
    dst=ROOT/"final.tgz"
    tmp.replace(dst)
    store(ROOT/"final_ready.json",{"sha256":digest(dst),"bytes":dst.stat().st_size,
                                    "time_kst":datetime.now(KST).isoformat(timespec="seconds")})


def snapshot(state):
    preds=RUN/"predictions.jsonl"
    if not preds.exists(): return
    from report_results import lines
    records=lines(preds)
    if not records: return
    res=subprocess.run([sys.executable,str(ROOT/"report_results.py"),"--partial"],
                       cwd=ROOT,capture_output=True)
    if res.returncode:
        raise RuntimeError(res.stderr.decode("utf8",errors="replace")[-500:])
    partial=load(ROOT/"partial_status.json")
    now=datetime.now(KST)
    data=dict(time_kst=now.isoformat(timespec="seconds"),state=state["state"],stage=state["stage"],
              generated=partial["generated"],resolved=partial["parser_or_judge_resolved"],
              pending_judge=len(partial["pending_judge_ids"]),ungenerated=partial["ungenerated"],
              resolved_correct=partial["subset_counts"]["correct"],
              resolved_baseline=partial["subset_counts"]["baseline_correct"],last_id=records[-1]["id"])
    if (RUN/"screen/complete.json").exists():
        s=load(RUN/"screen/summary.json")
        data["screen"]=dict(n=s["n"],correct=s["correct"],baseline=s["baseline_correct"],
                            recovered=s["paired_gained"],lost=s["paired_lost"])
    store(ROOT/"progress.json",data)
    begun=datetime.fromisoformat(load(ROOT/"started.json")["started"])
    if (now-begun).total_seconds()>=43200 and not (ROOT/"interim_12h.json").exists():
        store(ROOT/"interim_12h.json",data)


def main():
    while True:
        state=load(ROOT/"status.json")
        if state["state"]=="completed":
            complete()
            snapshot(state)
            return
        if state["state"]=="failed":
            store(ROOT/"observer_failure.json",state)
            return
        try:
            snapshot(state)
        except (OSError,ValueError,RuntimeError) as exc:
            print(datetime.now(KST).isoformat(timespec="seconds"),str(exc),flush=True)
        time.sleep(45)


if __name__=="__main__":
    main()
