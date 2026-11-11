"""Sequential single-GPU round; stages wait for frozen data or review on file events."""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
TRAIN=Path.home()/"era_decoder_lora_20261030/.venv/bin/python"
EVAL=Path.home()/"mmdl_test/.venv/bin/python"
OFFICIAL=Path.home()/"era_single_call_campaign/mmmu_eval.py"


def sha(p):
    return hashlib.file_digest(p.open("rb"),"sha256").hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def write(p, data):
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(p)


def stamp():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def waitfile(path,deadline):
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),"inotify_init")
    try:
        if libc.inotify_add_watch(fd,os.fsencode(path.parent),0x0080|0x0100|0x0008)<0:raise OSError(ctypes.get_errno(),"watch")
        while not path.exists():
            if not select.select([fd],[],[],max(0,deadline-time.time()))[0]:raise TimeoutError(path.name)
            os.read(fd,65536)
    finally:os.close(fd)


def status(state,stage,**more):
    write(ROOT/"status.json",dict(state=state,stage=stage,time_kst=stamp(),**more))


def waiting(name,deadline):
    status("waiting",name)
    path=ROOT/"signals"/(name+".json")
    waitfile(path,deadline)
    result=read(path)
    for rel,h in result["hashes"].items():
        assert sha(ROOT/rel)==h,rel
    return result


def step(name,env,script,args,deadline):
    path=ROOT/"steps"/(name+".json")
    cmd=[str(env),"-u",str(script),*map(str,args)]
    identity=dict(command=cmd,script_sha256=sha(script))
    if path.exists():
        assert read(path)["identity"]==identity
        return
    status("running",name)
    began=time.time()
    with (ROOT/"logs"/(name+".log")).open("a") as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
        try:rc=proc.wait(timeout=max(0,deadline-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise TimeoutError(name)
    if rc:raise RuntimeError(f"{name} exited {rc}")
    write(path,dict(identity=identity,ended=stamp(),elapsed_s=time.time()-began))


def main():
    for d in ("logs","steps","signals","mmmu"):(ROOT/d).mkdir(exist_ok=True)
    with (ROOT/".lock").open("a") as exclusive:
        fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=ROOT/"started.json"
        if not start.exists():write(start,dict(started=stamp(),deadline=time.time()+43200,protocol=sha(ROOT/"protocol.md")))
        deadline=read(start)["deadline"]
        assert read(start)["protocol"]==sha(ROOT/"protocol.md")
        try:
            waiting("train_release",deadline)
            trainer=ROOT/"train/train_a.py"
            infer=ROOT/"eval/infer.py"
            step("preflight",TRAIN,trainer,["--preflight"],deadline)
            for arm in ("ordinary","enhanced"):
                step(arm+"-train",TRAIN,trainer,["--arm",arm],deadline)
                step(arm+"-merge",TRAIN,trainer,["--arm",arm,"--merge"],deadline)
            step("training-verification",TRAIN,ROOT/"train/verify_training.py",[],deadline)
            waiting("evaluation_release",deadline)
            step("independence",TRAIN,ROOT/"check_splits.py",[],deadline)
            step("freeze-solver",EVAL,ROOT/"eval/prepare.py",[],deadline)
            for arm in ("base","ordinary","enhanced"):
                step(arm+"-infer",EVAL,infer,[arm],deadline)
            step("blind-packet",EVAL,ROOT/"eval/grade.py",["packet"],deadline)
            step("blind-grade",EVAL,ROOT/"eval/grade.py",["judge"],deadline)
            step("audit-packet",EVAL,ROOT/"eval/score.py",["audit"],deadline)
            waiting("manual_review",deadline)
            step("score",EVAL,ROOT/"eval/score.py",["score"],deadline)
            step("validate",EVAL,ROOT/"eval/validate.py",[],deadline)
            decision=read(ROOT/"decision.json")
            assert decision["scored_sha256"]==sha(ROOT/"eval/review/scored.jsonl")
            assert decision["summary_sha256"]==sha(ROOT/"eval/review/summary.json")
            if decision["promote"]:
                assert sha(OFFICIAL)=="ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
                for scope in ("screen","full"):
                    step("mmmu-"+scope,EVAL,OFFICIAL,["--scope",scope,"--name","enhanced","--output-root",ROOT/"mmmu","--checkpoint",ROOT/"train/models/r9/enhanced/merged"],deadline)
            status("completed","final",promoted=decision["promote"])
        except Exception as exc:
            status("time_limit" if isinstance(exc,TimeoutError) else "failed",read(ROOT/"status.json")["stage"],error=str(exc))
            raise


if __name__=="__main__":main()
