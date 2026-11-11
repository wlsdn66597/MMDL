"""Run the sealed R9 single-call solver on the fixed diagnostic or fresh confirmation roster."""
import argparse
import importlib.util
import json
from pathlib import Path

from scale import ARMS, BASE, BASE_HASHES, R9, ROOT, freeze, read, sha, source, weights

OLD = Path.home()/"era_midstep_round/eval/run_inference.py"
RUNNER_HASH = "a59d89cee13548fba4c2bf416a05581e3364129a11c36cdf0c0795a2a6b0c594"
R9_INPUTS_HASH = "ff9f3846e0820e3939dd07488dcdb5c3bc2d800b8d96a81fbea82afe3f8934a5"


def lines(p):
    return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines()]


def checkpoint(arm):
    if arm == "base":
        assert weights(BASE) == BASE_HASHES
        return BASE, None
    if arm == "enhanced100":
        folder = source("enhanced")
        assert weights(folder/"merged") == read(folder/"merge.json")["weights"]
        return folder/"merged", None
    receipt = ROOT/"scale_receipts"/(arm+".json")
    model = ROOT/"models"/arm/"merged"
    proof = read(receipt)
    assert proof == read(model/"half-export.json") and proof["arm"] == arm and proof["factor"] == 0.5
    assert weights(model) == proof["weights"]
    return model, sha(receipt)


def main(arm, stage):
    assert stage=="confirm" or arm in ARMS, "Reuse the sealed R9 endpoints on the diagnostic roster."
    here = (ROOT/"confirm" if stage=="confirm" else ROOT)/"eval"
    original = here.parent if stage=="confirm" else R9
    spec = original/"eval/spec"
    dataset = spec/"inputs.jsonl"
    receipt = read(spec/"receipt.json")
    assert sha(dataset)==receipt["inputs"] and sha(spec/"gold.jsonl")==receipt["gold"]
    assert stage=="confirm" or receipt["inputs"]==R9_INPUTS_HASH
    records = lines(dataset)
    assert len(records)==len({r["id"] for r in records})==192
    assert sum(r["dataset"]=="paired" for r in records)==sum(r["dataset"]=="general" for r in records)==96
    images = []
    for row in records:
        assert set(row)=={"id","dataset","image","image_sha256","prompt"}
        path=(original/row["image"]).resolve()
        assert path.is_relative_to(original.resolve()) and sha(path)==row["image_sha256"]
        images.append(dict(id=row["id"],image=str(path),image_sha256=row["image_sha256"],prompt=row["prompt"]))
    assert sha(OLD)==RUNNER_HASH
    module=importlib.util.spec_from_file_location("sealed_r9_solver",OLD)
    runner=importlib.util.module_from_spec(module)
    module.loader.exec_module(runner)
    runner.HERE=here/"fresh"
    class RoundEngine(runner.Engine):
        def one(self,row,root):
            return super().one(row,original)
    runner.Engine=RoundEngine
    runner.inputs=lambda dataset_name: (dataset,images)
    model, scale_hash=checkpoint(arm)
    runner.run("half",arm,model)
    journal=runner.HERE/"runs"/("half-"+arm)
    output=journal/"predictions.jsonl"
    assert read(journal/"done.json")["output_sha256"]==sha(output)
    preds=lines(output)
    assert len(preds)==len(records)
    for prediction, row in zip(preds,records,strict=True):
        for field in ("id","prompt","image_sha256"):
            assert prediction[field]==row[field]
        assert prediction["seed"]==3407 and prediction["max_output"]==2048
        assert prediction["finish_reason"] in ("stop","length")
    baseline=read(R9/"eval/fresh/runs/r9-base/manifest.json")
    current=read(journal/"manifest.json")
    assert all(current[k]==baseline[k] for k in ("config_sha256","engine_sha256","runner_sha256","batch","request_seed","max_output","eval_budget"))
    dst=here/"runs"/arm
    dst.mkdir(exist_ok=True,parents=True)
    copy=dst/"predictions.jsonl"
    if copy.exists():
        assert copy.read_bytes()==output.read_bytes()
    else:
        tmp=copy.with_suffix(".tmp")
        tmp.write_bytes(output.read_bytes())
        tmp.replace(copy)
    freeze(dst/"receipt.json",dict(arm=arm,n=len(preds),fresh=len(preds),sha256=sha(copy),solver=sha(OLD),
           wrapper=sha(Path(__file__)),inputs_sha256=sha(dataset),scale_receipt_sha256=scale_hash,
           stage=stage,journal_manifest_sha256=sha(journal/"manifest.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=[*ARMS,"base","enhanced100"])
    parser.add_argument("--stage",choices=["diagnostic","confirm"],default="diagnostic")
    args=parser.parse_args()
    main(args.arm,args.stage)
