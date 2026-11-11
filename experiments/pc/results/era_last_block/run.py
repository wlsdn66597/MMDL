"""Serial tmux controller. Each completed stage is journaled before proceeding."""
import fcntl
import hashlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
TRAIN = Path.home() / "era_decoder_lora_20261030/.venv/bin/python"
EVAL = Path.home() / "mmdl_test/.venv/bin/python"
CODE = ROOT / "evaluate_stage.py"


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2) + "\n")
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def step(name, cmd):
    receipt = ROOT / "steps" / (name+".json")
    if receipt.exists():
        return
    write(ROOT / "status.json",dict(state="running",stage=name,started=now()))
    with (ROOT / (name+".log")).open("a") as log:
        subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(receipt,dict(stage=name,finished=now(),command=list(map(str,cmd))))


def execute():
    while not (ROOT / "training_verified").is_file():
        if subprocess.run(["tmux","has-session","-t","era_last35_train"],stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError("Training exited before verification")
        time.sleep(3)
    for arm in ("base","all50","last50"):
        if arm != "base":
            step("export-"+arm,[TRAIN,"-u",CODE,"export",arm])
        step("infer-"+arm,[EVAL,"-u",CODE,"infer",arm])
        if arm == "all50":
            copied = ROOT / "exports/all50"
            if copied.exists():
                # Remove only this run's disposable verified copy after its inference ends.
                receipt = json.loads((ROOT / "all50-weights.json").read_text())
                assert copied.resolve() == ROOT/"exports/all50" and not copied.is_symlink()
                assert receipt["weights"] == {p.name:digest(p) for p in copied.glob("*.safetensors")}
                shutil.rmtree(copied)
    for action,env in (("grade",EVAL),("score",TRAIN)):
        step(action,[env,"-u",CODE,action])
    gate = json.loads((ROOT / "evaluation.json").read_text())["decision"]
    if gate == "pass":
        official = Path.home() / "era_single_call_campaign/mmmu_eval.py"
        assert digest(official) == "ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
        for scope in ("screen","full"):
            step("mmmu-"+scope,[EVAL,"-u",official,"--scope",scope,"--output-root",ROOT/"mmmu",
                 "--checkpoint",ROOT/"exports/last50","--name","last50"])
    write(ROOT/"status.json",dict(state="needs_review" if gate == "review" else "completed",gate=gate,time=now()))


if __name__ == "__main__":
    (ROOT/"steps").mkdir(exist_ok=True)
    with (ROOT/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            execute()
        except Exception as error:
            status = ROOT / "status.json"
            info = json.loads(status.read_text()) if status.exists() else {}
            write(status,dict(info,state="failed",error=str(error),time=now()))
            raise
