"""Run the frozen R6 stages in tmux until completion or the declared branch deadline."""
import argparse
import ctypes
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
EVAL=Path.home()/"mmdl_test/.venv/bin/python"
TRAIN=Path.home()/"era_decoder_lora_20261030/.venv/bin/python"
OFFICIAL=Path.home()/"era_single_call_campaign/mmmu_eval.py"
DEV=datetime.fromisoformat("2026-11-08T19:30:00+09:00").timestamp()
FULL=datetime.fromisoformat("2026-11-09T09:30:00+09:00").timestamp()
KST=ZoneInfo("Asia/Seoul")


def timestamp(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t,KST).isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, data):
    tmp=path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def status(state, stage, deadline, **attrs):
    save(ROOT/"status.json",dict(state=state,stage=stage,time_kst=timestamp(),deadline_kst=timestamp(deadline),**attrs))


def wait_event(path, deadline, terminal_ok=False):
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),"inotify_init")
    try:
        for d in {path.parent,ROOT}:
            if libc.inotify_add_watch(fd,os.fsencode(d),0x0008|0x0080|0x0100)<0:
                raise OSError(ctypes.get_errno(),str(d))
        while not path.exists():
            if terminal_ok and (ROOT/"status.json").exists() and read(ROOT/"status.json")["state"] in {"failed","completed","time_limit"}:
                return False
            if not select.select([fd],[],[],max(0,deadline-time.time()))[0]:
                raise TimeoutError(str(path))
            os.read(fd,65536)
        return True
    finally:
        os.close(fd)


def release(name, deadline):
    path=ROOT/"signals"/(name+".json")
    status("waiting",name,deadline)
    wait_event(path,deadline)
    blob=read(path)
    for rel,expected in blob["hashes"].items():
        assert sha(ROOT/rel)==expected,rel
    return blob


def step(name, deadline, env, script, *args, cpu=False):
    dst=ROOT/"steps"/(name+".json")
    cmd=[str(env),"-u",str(script),*map(str,args)]
    identity=dict(command=cmd,script_sha256=sha(script))
    if dst.exists():
        assert read(dst)["identity"]==identity
        return
    if time.time()>=deadline:raise TimeoutError(name)
    status("running",name,deadline)
    began=time.time()
    print(timestamp(),name,flush=True)
    procenv=os.environ.copy()
    if cpu:procenv["CUDA_VISIBLE_DEVICES"]=""
    with (ROOT/"logs"/(name+".log")).open("a") as out:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=procenv,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=proc.wait(timeout=max(.1,deadline-time.time()))
        except BaseException as exc:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
            if isinstance(exc,subprocess.TimeoutExpired):raise TimeoutError(name) from exc
            raise
    if rc:raise RuntimeError(f"{name} exited with {rc}, see its stage log")
    save(dst,dict(identity=identity,started_kst=timestamp(began),ended_kst=timestamp(),elapsed_seconds=time.time()-began))


def run():
    release("train",DEV)
    trainer=ROOT/"train/train_direction.py"
    step("preflight",DEV,TRAIN,trainer,"--preflight",cpu=True)
    release("eval",DEV)
    infer=ROOT/"eval/run.py"
    step("verify-baseline",DEV,EVAL,infer,"prepare",cpu=True)
    for arm,label in (("control","paraphrase"),("reverse","reversed")):
        model=ROOT/"train/models/r6"/arm/"merged"
        step(arm+"-train",DEV,TRAIN,trainer,"--arm",arm)
        step(arm+"-merge",DEV,TRAIN,trainer,"--arm",arm,"--merge")
        for dataset in ("charts","science"):
            step(label+"-"+dataset,DEV,EVAL,infer,"infer","--dataset",dataset,"--arm",label,"--checkpoint",model)
    step("blind-packet",DEV,EVAL,ROOT/"eval/review.py","packet",cpu=True)
    gate=release("decision",DEV)
    if not gate["promote"]:
        status("completed","final",DEV,promoted=False,reason=gate["reason"])
        return
    assert sha(OFFICIAL)=="ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
    for scope in ("screen","full"):
        step("mmmu-"+scope,FULL,EVAL,OFFICIAL,"--scope",scope,"--name","reverse","--output-root",ROOT/"mmmu",
             "--checkpoint",ROOT/"train/models/r6/reverse/merged")
    status("completed","final",FULL,promoted=True)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--wait")
    a=cli.parse_args()
    if a.wait:
        path=(ROOT/a.wait).resolve()
        assert path.is_relative_to(ROOT)
        print("event" if wait_event(path,FULL,True) else "terminal",path,flush=True)
        return
    for name in ("logs","signals","steps"):(ROOT/name).mkdir(exist_ok=True)
    with (ROOT/".lock").open("a") as exclusive:
        fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
        began=ROOT/"started.json"
        if not began.exists():
            save(began,dict(started_kst=timestamp(),protocol=sha(ROOT/"protocol.md"),dev_stop=timestamp(DEV),mmmu_stop=timestamp(FULL)))
        assert read(began)["protocol"]==sha(ROOT/"protocol.md")
        try:run()
        except Exception as exc:
            last=read(ROOT/"status.json")
            status("time_limit" if isinstance(exc,TimeoutError) else "failed",last["stage"],
                   datetime.fromisoformat(last["deadline_kst"]).timestamp(),error=str(exc))
            raise


if __name__=="__main__":
    main()
