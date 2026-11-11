"""Wait on a Linux filesystem completion event without polling or allocating a GPU."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import time

ROOT=Path(__file__).resolve().parent


def wait(relative):
    requested=(ROOT/relative).resolve()
    if not requested.is_relative_to(ROOT):raise ValueError("Event must be inside the campaign")
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),"inotify_init1")
    try:
        for p in {ROOT,requested.parent}:
            if libc.inotify_add_watch(fd,os.fsencode(p),0x0008|0x0080|0x0100)<0:
                raise OSError(ctypes.get_errno(),str(p))
        while True:
            if requested.is_file():
                print("EVENT",relative,flush=True);return
            status=ROOT/"status.json"
            if status.exists():
                state=json.loads(status.read_text())
                if state["state"] in {"completed","failed","budget_limit"}:
                    print(json.dumps(state,ensure_ascii=False),flush=True);return
            start=json.loads((ROOT/"started.json").read_text())
            left=max(0,start["deadline"]+10-time.time())
            if not select.select([fd],[],[],left)[0]:
                print("DEADLINE",relative,flush=True);return
            os.read(fd,65536)
    finally:os.close(fd)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("event");wait(ap.parse_args().event)
