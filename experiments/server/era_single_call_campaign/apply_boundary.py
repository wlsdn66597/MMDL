"""One-time local-controller update; never interrupt or repeat an in-flight solver request."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"pending_v2"


def main():
    os.chdir(ROOT)
    started=json.loads((ROOT/"started.json").read_text())
    deadline=started["deadline"]
    if time.time()>deadline:raise RuntimeError("Budget expired")
    assert not (ROOT/"rounds.json").exists() and not (ROOT/"nominee.json").exists()
    # Stage all replacement bytes up front; the revised files never affect existing solver code.
    subprocess.run(["sha256sum","--quiet","-c","incoming.sha256"],cwd=BASE,check=True)
    pid=int(subprocess.check_output(["tmux","display-message","-p","-t","era-single","#{pane_pid}"]))
    proc=Path(f"/proc/{pid}")
    assert (proc/"cwd").resolve()==ROOT and (proc/"cmdline").read_bytes().split(b"\0")[-2]==b"campaign.py"
    parentfd=os.pidfd_open(pid)
    stopped=False
    try:
        os.kill(pid,signal.SIGSTOP);stopped=True
        for _ in range(150):
            if (proc/"stat").read_text().split(") ")[1][0]=="T":break
            time.sleep(.02)
        else:raise RuntimeError("Old controller could not pause")
        stage=json.loads((ROOT/"status.json").read_text())["stage"]
        if not stage.startswith("r1-candidate-") or (ROOT/"rounds.json").exists():raise RuntimeError("Too late for a pre-result amendment")
        children=[int(c) for c in (proc/"task"/str(pid)/"children").read_text().split()]
        assert len(children)==1
        child=children[0]
        assert os.getpgid(child)==child
        childfd=os.pidfd_open(child)
        try:
            (ROOT/"handoff-state.json").write_text(json.dumps(dict(stage=stage,child=child,state="waiting for already-submitted stage",deadline=deadline)))
            if not select.select([childfd],[],[],max(0,deadline-time.time()))[0]:
                os.killpg(child,signal.SIGTERM)
                if not select.select([childfd],[],[],2)[0]:os.killpg(child,signal.SIGKILL)
                os.kill(pid,signal.SIGCONT);stopped=False
                raise RuntimeError("Budget expired during in-flight stage")
        finally:os.close(childfd)
        # This registered phase finished. A restart validates its immutable output via its normal entrypoint.
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False
        if not select.select([parentfd],[],[],10)[0]:raise RuntimeError("Old controller did not exit")
        for rel in ("campaign.py","protocol.md","train_format.py","release.sha256"):
            src=BASE/rel
            temp=ROOT/(rel+".new")
            temp.write_bytes(src.read_bytes());temp.replace(ROOT/rel)
        subprocess.run(["sha256sum","--quiet","-c","release.sha256"],check=True)
        (ROOT/"handoff-state.json").write_text(json.dumps(dict(stage=stage,state="installed; prior solver ledger and deadline retained",deadline=deadline)))
    finally:
        if stopped:os.kill(pid,signal.SIGCONT)
        os.close(parentfd)
    # Keep the original connection command convenient if the retiring tmux session is already gone.
    subprocess.run(["tmux","rename-session","-t","era-single-update","era-single"],check=False)
    os.execvp("bash",["bash",str(ROOT/"start.sh")])


if __name__=="__main__":main()
