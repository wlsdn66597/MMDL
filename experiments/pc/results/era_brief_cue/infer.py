"""Submit exactly one response per cued item, using the archived journaled solver."""
import argparse
from importlib import util
from pathlib import Path

from checkpoint import HALF, checkpoint, exporter
from prepare import ROOT, R9, SPEC, R9_INPUTS, R9_GOLD, prompted, read, rows, seal_bytes, sha

SOLVER = Path.home()/"era_midstep_round/eval/run_inference.py"
SOLVER_SHA = "a59d89cee13548fba4c2bf416a05581e3364129a11c36cdf0c0795a2a6b0c594"
DECODING = ("config_sha256","engine_sha256","runner_sha256","batch","request_seed","max_output","eval_budget")


def main(arm):
    ds = SPEC/"inputs.jsonl"
    proof = read(SPEC/"receipt.json")
    assert sha(ds)==proof["inputs"] and sha(SPEC/"gold.jsonl")==proof["gold"]==R9_GOLD
    original = R9/"eval/spec/inputs.jsonl"
    assert sha(original)==proof["original_inputs"]==R9_INPUTS
    records = rows(ds)
    assert records==[prompted(r) for r in rows(original)]
    assert len(records)==len({r["id"] for r in records})==192
    images=[]
    for row in records:
        image=(R9/row["image"]).resolve()
        assert image.is_relative_to(R9.resolve()) and sha(image)==row["image_sha256"]
        images.append(dict(id=row["id"],prompt=row["prompt"],image=str(image),image_sha256=row["image_sha256"]))
    assert sha(SOLVER)==SOLVER_SHA
    spec=util.spec_from_file_location("cued_single_call_solver",SOLVER)
    runner=util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.HERE=ROOT/"eval/fresh"
    class CueEngine(runner.Engine):
        def one(self,row,root):
            return super().one(row,R9)
    runner.Engine=CueEngine
    runner.inputs=lambda dataset: (ds,images)
    module=exporter()
    model, scale_hash=checkpoint(arm,module)
    runner.run("cue",arm,model)
    journal=runner.HERE/"runs"/("cue-"+arm)
    manifest=read(journal/"manifest.json")
    original_journal=(R9/"eval/fresh/runs/r9-base" if arm=="base"
                      else HALF/"eval/fresh/runs"/("half-"+arm))
    control=read(original_journal/"manifest.json")
    assert all(manifest[k]==control[k] for k in (*DECODING,"weights"))
    assert manifest["inputs_sha256"]==proof["inputs"]!=control["inputs_sha256"]
    out=journal/"predictions.jsonl"
    done=read(journal/"done.json")
    assert done["output_sha256"]==sha(out) and done["manifest_sha256"]==sha(journal/"manifest.json")
    assert not (journal/"submitted.json").exists()
    preds=rows(out)
    assert done["n"]==len(preds)==len(records)
    for pred,row in zip(preds,records,strict=True):
        assert all(pred[k]==row[k] for k in ("id","prompt","image_sha256"))
        assert pred["seed"]==3407 and pred["max_output"]==2048 and pred["finish_reason"] in ("stop","length")
    dst=ROOT/"eval/runs"/arm
    seal_bytes(dst/"predictions.jsonl",out.read_bytes())
    module.freeze(dst/"receipt.json",dict(arm=arm,n=len(preds),fresh=len(preds),sha256=sha(out),
                  solver=SOLVER_SHA,wrapper=sha(Path(__file__)),inputs_sha256=proof["inputs"],
                  original_inputs_sha256=R9_INPUTS,scale_receipt_sha256=scale_hash,
                  journal_manifest_sha256=sha(journal/"manifest.json"),control_manifest_sha256=sha(original_journal/"manifest.json")))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=("base","ordinary","enhanced"))
    main(parser.parse_args().arm)
