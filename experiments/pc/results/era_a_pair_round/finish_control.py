"""Complete only never-submitted control items and score saved answers up to the authorized end time."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "control_recovery"
sys.path.insert(0,str(Path.home()/"era_single_call_campaign"))
from campaign import read, save, timestamp


def stage(name, script, args, deadline):
    now = time.time()
    if now+2 >= deadline:
        return "deadline"
    save(OUT/"status.json",dict(state="running",stage=name,time_kst=timestamp(),hard_deadline_kst=timestamp(deadline)))
    with (OUT/"logs"/(name+".log")).open("a") as log:
        proc = subprocess.Popen([sys.executable,"-u",str(ROOT/script),*args],cwd=ROOT,
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc = proc.wait(timeout=max(.1,deadline-time.time()-2))
            return "ok" if rc == 0 else f"exit {rc}"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return "deadline"


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/"logs").mkdir(exist_ok=True)
    lock = (OUT/"controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    hard = read(ROOT/"started.json")["deadline"] + 4*3600
    assert timestamp(hard) == read(ROOT/"extension.json")["hard_deadline_kst"]
    assert read(ROOT/"mmmu/control/control/solver_inflight.json") == {"id":"validation_Mechanical_Engineering_11"}
    attempts = read(OUT/"stage_log.json",[])
    for name, script, args in (("solver","recover_solver.py",[]),("judge","recover_scores.py",["--judge"])):
        if any(a["stage"]==name and a["rc"]=="ok" for a in attempts):
            continue
        t = time.time()
        rc = stage(name,script,args,hard)
        attempts.append(dict(stage=name,rc=rc,began_kst=timestamp(t),ended_kst=timestamp()))
        save(OUT/"stage_log.json",attempts)
        if rc != "ok":
            break
    summary = subprocess.run([sys.executable,"-u",str(ROOT/"recover_scores.py"),"--partial"],cwd=ROOT)
    state = "completed_with_missing_response" if summary.returncode == 0 and all(
        any(a["stage"]==s and a["rc"]=="ok" for a in attempts) for s in ("solver","judge")) else "partial"
    save(OUT/"status.json",dict(state=state,time_kst=timestamp(),hard_deadline_kst=timestamp(hard),
                                partial_returncode=summary.returncode))
    print(state, flush=True)


if __name__ == "__main__":
    main()
