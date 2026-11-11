"""Finite event-driven pipeline. Evaluation and teacher requests retain their own no-retry ledgers."""
import argparse
import ctypes
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PRIOR = Path.home() / "era_single_call_campaign"
EVAL = Path.home() / "mmdl_test/.venv/bin/python"
TRAIN = Path.home() / "era_decoder_lora_20261030/.venv/bin/python"
BASE = Path.home() / "mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"


def stamp():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def save(path, value):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def status(state, stage, **details):
    save(ROOT/"status.json",dict(state=state,stage=stage,time_kst=stamp(),**details))


def wait_file(path, until=None):
    """Atomic rename/create wakeups. No interpreter/GPU polling while waiting on review."""
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:
        raise OSError(ctypes.get_errno(),"inotify_init1")
    try:
        for folder in {path.parent,ROOT}:
            if libc.inotify_add_watch(fd,os.fsencode(folder),0x0080|0x0100|0x0008)<0:
                raise OSError(ctypes.get_errno(),str(folder))
        while not path.exists():
            if until and until():
                return False
            select.select([fd],[],[])
            os.read(fd,65536)
        return True
    finally:
        os.close(fd)


def barrier(name):
    path=ROOT/"signals"/(name+".json")
    if not path.exists():
        status("awaiting_release",name)
        wait_file(path)
    obj=read(path)
    for rel,digest in obj["hashes"].items():
        assert sha(ROOT/rel)==digest,rel
    return obj


def step(name, env, script, *args, cpu=False):
    dest=ROOT/"steps"/(name+".json")
    cmd=[str(env),"-u",str(script),*map(str,args)]
    identity=dict(command=cmd,source=sha(script))
    if dest.exists():
        assert read(dest)["identity"]==identity,name
        return
    status("running",name)
    print(stamp(),name,flush=True)
    start=time.monotonic()
    started=stamp()
    environment=os.environ.copy()
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"]=""
    with (ROOT/"logs"/(name+".log")).open("a") as log:
        subprocess.run(cmd,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(dest,dict(identity=identity,started_kst=started,ended_kst=stamp(),seconds=time.monotonic()-start))


def run():
    infer=ROOT/"eval/run_inference.py"
    barrier("teacher-ready")
    step("teacher-192",EVAL,infer,"--dataset","teacher","--arm","base","--checkpoint",BASE)
    barrier("eval-ready")
    for ds in ("charts","science"):
        step("base-"+ds,EVAL,infer,"--dataset",ds,"--arm","base","--checkpoint",BASE)
    barrier("train-ready")
    trainer=ROOT/"train/train_midstep.py"
    step("train-preflight",TRAIN,trainer,"--preflight",cpu=True)
    for arm,label in (("standard","ordinary"),("preserved","preserved")):
        step(arm+"-train",TRAIN,trainer,"--arm",arm)
        step(arm+"-merge",TRAIN,trainer,"--arm",arm,"--merge")
        ckpt=ROOT/"train/models/r5"/arm/"merged"
        for ds in ("charts","science"):
            step(label+"-"+ds,EVAL,infer,"--dataset",ds,"--arm",label,"--checkpoint",ckpt)
    gate=barrier("blind-gate")
    if gate["promote"]:
        official=PRIOR/"mmmu_eval.py"
        assert sha(official)=="ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
        for scope in ("screen","full"):
            step("mmmu-"+scope,EVAL,official,"--scope",scope,"--name","preserved","--output-root",
                 ROOT/"mmmu","--checkpoint",ROOT/"train/models/r5/preserved/merged")
    status("completed","final",promoted=gate["promote"],reason=gate["reason"])


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--wait",help="Existing run's relative event path")
    args=cli.parse_args()
    if args.wait:
        dest=(ROOT/args.wait).resolve()
        assert dest.is_relative_to(ROOT)
        state=lambda:(ROOT/"status.json").exists() and read(ROOT/"status.json")["state"] in ("failed","completed")
        print("event" if wait_file(dest,state) else "terminal",dest,flush=True)
        if (ROOT/"status.json").exists():
            print(read(ROOT/"status.json"),flush=True)
        return
    for d in ("signals","steps","logs"):
        (ROOT/d).mkdir(exist_ok=True)
    with (ROOT/".pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (ROOT/"started.json").exists():
            save(ROOT/"started.json",dict(time_kst=stamp(),protocol_sha256=sha(ROOT/"protocol.md")))
        assert read(ROOT/"started.json")["protocol_sha256"]==sha(ROOT/"protocol.md")
        try:
            run()
        except Exception as e:
            prior=read(ROOT/"status.json") if (ROOT/"status.json").exists() else {}
            status("failed",prior.get("stage","setup"),error=str(e))
            raise


if __name__=="__main__":
    main()
