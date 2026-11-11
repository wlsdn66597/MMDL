"""Run each new input once using the previous round's journaled vLLM runner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=Path.home()/"era_midstep_round/eval/run_inference.py"
BASE=Path.home()/"mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"


def sha(path):
    return hashlib.file_digest(path.open("rb"),"sha256").hexdigest()


def lines(p):
    return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))


def main(arm):
    assert arm in ("base","ordinary","enhanced")
    spec=HERE/"spec";receipt=json.loads((spec/"receipt.json").read_text())
    assert all(sha(spec/(n+".jsonl"))==h for n,h in receipt.items())
    assert sha(OLD)=="a59d89cee13548fba4c2bf416a05581e3364129a11c36cdf0c0795a2a6b0c594"
    module=importlib.util.spec_from_file_location("sealed_solver",OLD)
    runner=importlib.util.module_from_spec(module)
    module.loader.exec_module(runner)
    runner.HERE=HERE/"fresh"
    class RoundEngine(runner.Engine):
        def one(self,row,root):
            return super().one(row,ROOT)
    runner.Engine=RoundEngine
    full=lines(spec/"inputs.jsonl")
    source=spec/"inputs.jsonl"
    def inputs(ds):
        rows=[]
        for r in full:
            assert r["dataset"] in ("paired","general")
            f=(ROOT/r["image"]).resolve()
            assert f.is_relative_to(ROOT) and sha(f)==r["image_sha256"]
            rows.append(dict(id=r["id"],image=str(f),image_sha256=r["image_sha256"],prompt=r["prompt"]))
        return source,rows
    runner.inputs=inputs
    model=BASE if arm=="base" else ROOT/"train/models/r9"/arm/"merged"
    if arm!="base":
        merge=json.loads((model.parent/"merge.json").read_text())
        assert merge["weights"]=={p.name:sha(p) for p in model.glob("*.safetensors")}
    runner.run("r9",arm,model)
    out=runner.HERE/"runs"/("r9-"+arm)/"predictions.jsonl"
    preds=lines(out)
    assert len(preds)==len(full)==192
    for r,d in zip(preds,full,strict=True):
        assert r["id"]==d["id"] and r["prompt"]==d["prompt"] and r["image_sha256"]==d["image_sha256"]
        assert r["seed"]==3407 and r["max_output"]==2048
    target=HERE/"runs"/arm
    target.mkdir(exist_ok=True,parents=True)
    buf="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in preds).encode()
    p=target/"predictions.jsonl"
    if p.exists():assert p.read_bytes()==buf
    else:p.write_bytes(buf)
    (target/"receipt.json").write_text(json.dumps(dict(arm=arm,n=len(preds),fresh=len(lines(out)),sha256=sha(p),
        solver=sha(OLD),wrapper=sha(Path(__file__)),inputs_sha256=sha(source)))+"\n")


if __name__=="__main__":main(sys.argv[1])
