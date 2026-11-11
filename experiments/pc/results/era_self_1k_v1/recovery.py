"""Require exact replay of a real process exit and restart before production SFT."""
import json
import os
from pathlib import Path
import subprocess
import sys

from train import ROOT, file_lock, read, sha256_file, write_json


def launch(name, interrupt=False):
    command=[sys.executable,"-u",str(ROOT/"train.py"),"B","--smoke",name]
    if interrupt:
        command.append("--interrupt")
    output=ROOT/"recovery"/(name+".log")
    with output.open("a",encoding="utf8") as logfile:
        code=subprocess.run(command,cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT,check=False).returncode
    expected=86 if interrupt else 0
    if code!=expected:
        raise RuntimeError(f"{name}: expected exit {expected}, got {code}. See {output}")


def require_equal(left,right,where):
    import numpy as np
    import torch
    if isinstance(left,dict):
        assert left.keys()==right.keys(),where
        for k in left:
            require_equal(left[k],right[k],f"{where}.{k}")
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right),where
        for i,(a,b) in enumerate(zip(left,right)):
            require_equal(a,b,f"{where}[{i}]")
    elif isinstance(left,torch.Tensor):
        assert left.dtype==right.dtype and torch.equal(left,right),where
    elif isinstance(left,np.ndarray):
        assert np.array_equal(left,right),where
    else:
        assert left==right,where


def main():
    import torch
    out=ROOT/"recovery"
    out.mkdir(exist_ok=True)
    with file_lock(out/".test.lock"):
        reference=out/"reference"
        interrupted=out/"interrupted"
        if not (reference/"manifest.json").exists():
            launch("reference")
        if not (interrupted/"smoke_exit.json").exists():
            launch("interrupted",interrupt=True)
        origin=read(interrupted/"smoke_exit.json")
        assert origin["step"]==2 and origin["expected_exit"]==86
        if not (interrupted/"manifest.json").exists():
            launch("interrupted")
        a,b=[read(p/"manifest.json") for p in (reference,interrupted)]
        assert a["complete"] and b["complete"] and a["actual_steps"]==b["actual_steps"]==6
        assert a["identity"]==b["identity"]==origin["identity"]
        assert a["initial_adapter"]==b["initial_adapter"] and a["final_adapter"]==b["final_adapter"]
        assert read(reference/"seen_ids.json")==read(interrupted/"seen_ids.json") and a["sequence_sha256"]==b["sequence_sha256"]
        assert len(read(interrupted/"seen_ids.json"))==18 and a["mask_summary"]==b["mask_summary"]
        for name in ("optimizer.pt","scheduler.pt","rng.pt"):
            left,right=[torch.load(p/"checkpoints/step-6"/name,map_location="cpu",weights_only=name!="rng.pt")
                        for p in (reference,interrupted)]
            require_equal(left,right,name)
        receipt=dict(complete=True,train_sha256=sha256_file(ROOT/"train.py"),parent=a["identity"]["start_weights"],
            exact_adapter_sha256=a["final_adapter"],sequence_sha256=a["sequence_sha256"],actual_steps=6,seen=18,
            stop_after_step=2,exact_optimizer=True,exact_scheduler=True,exact_rng=True,mask_summary=a["mask_summary"])
        write_json(out/"verified.json",receipt)
        print(json.dumps(receipt),flush=True)


if __name__=="__main__":main()
