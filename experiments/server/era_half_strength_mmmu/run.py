"""Evaluate the previously measured ordinary half-strength checkpoint on official MMMU."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
R9 = ROOT.parent/"era_a_only_round"
sys.path.insert(0,str(R9))
import campaign as jobs

HALF = ROOT.parent/"era_half_strength"
SCALE_HASH = "9ee1ee1ae0755ff43c9591fa7ce451633fe70b5bbeb05e98fbb1ffb0bcefe021"
PROOF_HASH = "1fdf6aeaca39c491fe51a5dce15e5ca3a4d2a02a9f6161d13bbf6a5f8de9ca8e"
OFFICIAL_HASH = "ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
ARM = "ordinary"
NAME = "ordinary50"
MODEL = ROOT/"models"/ARM/"merged"
OUT = ROOT/"mmmu"
RUN = OUT/NAME


def scale_module():
    script = HALF/"scale.py"
    assert jobs.sha(script)==SCALE_HASH
    spec=importlib.util.spec_from_file_location("verified_r9_scaler",script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT=ROOT
    return module


def model_stage(retire=False):
    module = scale_module()
    expected = HALF/"scale_receipts"/"ordinary.json"
    assert jobs.sha(expected)==PROOF_HASH
    receipt=ROOT/"scale_receipts/ordinary.json"
    receipt.parent.mkdir(exist_ok=True)
    if not receipt.exists():
        receipt.write_bytes(expected.read_bytes())
    assert jobs.sha(receipt)==PROOF_HASH
    assert module.read(receipt)["weights"]==jobs.read(HALF/"eval/fresh/runs/half-ordinary/manifest.json")["weights"]
    if not retire:
        module.main(ARM)
        assert module.weights(MODEL)==module.read(receipt)["weights"]
        return
    final=jobs.read(RUN/"full/complete.json")
    summary=jobs.read(RUN/"full/summary.json")
    assert final["n"]==summary["n"]==900 and summary["baseline_correct"]==575
    assert all(jobs.sha(RUN/"full"/f)==h for f,h in final["outputs"].items())
    assert final["predictions_sha256"]==jobs.sha(RUN/"predictions.jsonl")
    if MODEL.exists():
        module.remove_artifacts(MODEL,ARM,True)


def do(stage, cmd):
    receipt=ROOT/"steps"/(stage+".json")
    if receipt.exists():
        return
    jobs.write(ROOT/"status.json",dict(state="running",stage=stage,time_kst=jobs.stamp()))
    with (ROOT/"logs"/(stage+".log")).open("a") as log:
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs.write(receipt,dict(time_kst=jobs.stamp(),command=list(map(str,cmd))))


def main():
    for d in ("logs","steps","mmmu"):
        (ROOT/d).mkdir(exist_ok=True)
    with (ROOT/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(protocol_sha256=jobs.sha(ROOT/"protocol.md"),runner_sha256=jobs.sha(Path(__file__)),
                      checkpoint_receipt_sha256=PROOF_HASH,official_sha256=OFFICIAL_HASH,model=NAME)
        started=ROOT/"started.json"
        if started.exists():
            assert jobs.read(started)["identity"]==identity
        else:
            jobs.write(started,dict(started=jobs.stamp(),identity=identity))
        assert jobs.sha(jobs.OFFICIAL)==OFFICIAL_HASH
        try:
            do("checkpoint",[jobs.TRAIN,"-u",__file__,"checkpoint"])
            for scope in ("screen","full"):
                do(scope,[jobs.EVAL,"-u",jobs.OFFICIAL,"--name",NAME,"--scope",scope,"--output-root",OUT,"--checkpoint",MODEL])
            do("retire",[jobs.TRAIN,"-u",__file__,"retire"])
            w=jobs.read(RUN/"full/summary.json")
            jobs.write(ROOT/"status.json",dict(state="completed",stage="final",time_kst=jobs.stamp(),summary=w))
        except Exception as exc:
            stage=jobs.read(ROOT/"status.json")["stage"] if (ROOT/"status.json").is_file() else "setup"
            jobs.write(ROOT/"status.json",dict(state="failed",stage=stage,time_kst=jobs.stamp(),error=str(exc)))
            raise


if __name__=="__main__":
    if len(sys.argv)==1:
        main()
    else:
        assert sys.argv[1:] in (["checkpoint"],["retire"])
        model_stage(retire=sys.argv[1]=="retire")
