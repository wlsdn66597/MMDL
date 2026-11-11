"""Reuse R5 frozen inputs, original responses and single-submission runner."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / "era_midstep_round" / "eval"
# R5's two internal journal names are retained for the original blind scorer.
ARMS = {"paraphrase": "ordinary", "reversed": "preserved"}
RUNNER_HASH = "a59d89cee13548fba4c2bf416a05581e3364129a11c36cdf0c0795a2a6b0c594"
SCORER_HASH = "43ed42de0b9e5097838d0d93043afa053e8df3aa7a4f39d4dcfa2fde2ce2ecc2"
SUMMARY_HASH = "55750db714ead039b337fa22ea30d48fa37ffe5bdc7a30b6f6c63940d804872a"
PACKET_HASH = "a69627729086490894615dc5a30fc535191f9a059d73151fd7bb0f3e1dc36ee9"
INPUTS = {"charts":"f1be9e728a836bfecf784451c5b4dbcd5092d4b9519498ece1b9425d7ed16233",
          "science":"4538cf9358debc6eb46d01c3ce63a060784ecb4ce1377748fdd615309394c216"}
GOLD = {"charts":"0171459c7599cfadb788f8c3950db89b84ff80f7c16058e9126dbd3135fe3cda",
        "science":"cf24d70f750bbde044fc6eff074a4135832175488da2c656bf32db6093755edb"}
ORIGINAL = {
    "charts": ("9f236b80e13641f7fa5b4a80eab98241a950591b84f790f36e4254f7bd3e91a6","fc4115c7a4403b62d95d2d32dd428b7cfe431e764cc2e85f3acf5fd0ffd43103",96),
    "science": ("664eeb6ba49e0ad753767bb566c2babee954849c456cfa71721d0013490fdb22","4d0693f717e0039b6bcd8bd650a916599d86dba4708b561081503cb5650f7e8d",19)}

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def freeze(path, data):
    raw = (json.dumps(data,indent=2,ensure_ascii=False)+"\n").encode()
    if path.exists():
        assert path.read_bytes()==raw,str(path)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(".preparing")
        tmp.write_bytes(raw)
        tmp.replace(path)

def load_tool(name, expected):
    path=PREVIOUS/(name+".py")
    assert sha(path)==expected,"Previous implementation checksum differs"
    spec=importlib.util.spec_from_file_location("previous_"+name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE=HERE
    return module

def copy_sealed(source,dest,expected):
    assert sha(source)==expected,str(source)
    if dest.exists():
        assert sha(dest)==expected,str(dest)
        return
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(".preparing")
    shutil.copyfile(source,tmp)
    assert sha(tmp)==expected
    tmp.replace(dest)

def prepare():
    assert sha(PREVIOUS/"run_inference.py")==RUNNER_HASH
    assert sha(PREVIOUS/"score.py")==SCORER_HASH
    assert sha(PREVIOUS/"review/summary.json")==SUMMARY_HASH
    assert sha(PREVIOUS/"review/packet.json")==PACKET_HASH
    for d,(pred,manifest,n) in ORIGINAL.items():
        src=PREVIOUS/"spec"/d
        spec=HERE/"spec"/d
        copy_sealed(src/"inputs.json",spec/"inputs.json",INPUTS[d])
        copy_sealed(src/"gold.json",spec/"gold.json",GOLD[d])
        if d=="science":
            copy_sealed(src/"visual_review.md",spec/"visual_review.md","7b4a93b3a76b4e0b957e6065320f45ee58b7edc80038790eae2e7ddfe1b98032")
        inputs=read(spec/"inputs.json")
        assert n==len({r["id"] for r in inputs})==len(inputs)
        ids=[]
        for row in inputs:
            img=Path(row["image"])
            assert not img.is_absolute() and ".." not in img.parts
            copy_sealed(src/img,spec/img,row["image_sha256"])
            ids.append(row["id"])
        old=PREVIOUS/"runs"/(d+"-base")
        run=HERE/"runs"/(d+"-base")
        assert read(old/"done.json")==dict(n=n,output_sha256=pred,manifest_sha256=manifest)
        for file,expected in (("predictions.jsonl",pred),("manifest.json",manifest),("done.json",sha(old/"done.json"))):
            copy_sealed(old/file,run/file,expected)
        assert [json.loads(x)["id"] for x in (run/"predictions.jsonl").read_text(encoding="utf8").splitlines()]==ids
    freeze(HERE/"prepared.json",dict(original=ORIGINAL,inputs=INPUTS,gold=GOLD,runner_sha256=RUNNER_HASH,
        scorer_sha256=SCORER_HASH,historical_packet_sha256=PACKET_HASH,historical_summary_sha256=SUMMARY_HASH,
        arm_journal_names=ARMS,base_solver_calls_this_round=0))

def main():
    cli=argparse.ArgumentParser()
    sub=cli.add_subparsers(dest="mode",required=True)
    sub.add_parser("prepare")
    cmd=sub.add_parser("infer")
    cmd.add_argument("--dataset",choices=INPUTS,required=True)
    cmd.add_argument("--arm",choices=ARMS,required=True)
    cmd.add_argument("--checkpoint",type=Path,required=True)
    cmd.add_argument("--check-only",action="store_true")
    opt=cli.parse_args()
    if opt.mode=="prepare":
        prepare()
        print("Sealed 96/19 inputs and baseline submissions; 0 new solver calls.")
    else:
        receipt=read(HERE/"prepared.json")
        assert receipt["arm_journal_names"]==ARMS and receipt["runner_sha256"]==RUNNER_HASH
        folder_arm={"paraphrase":"control","reversed":"reverse"}[opt.arm]
        export=(HERE.parent/"train/models/r6"/folder_arm/"merged").resolve()
        assert opt.checkpoint.resolve()==export,(opt.checkpoint,"checkpoint for another arm")
        baseline=read(HERE/"runs"/(opt.dataset+"-base")/"manifest.json")
        runner=load_tool("run_inference",RUNNER_HASH)
        assert sha(runner.PILOT/"eval_config.json")==baseline["config_sha256"]
        assert sha(runner.PILOT/"infer.py")==baseline["engine_sha256"]
        merged=read(export.parent/"merge.json")
        assert merged["weights"]=={p.name:sha(p) for p in export.glob("*.safetensors")}
        assert merged["weights"]!=baseline["weights"]
        runner.run(opt.dataset,ARMS[opt.arm],opt.checkpoint,opt.check_only)

if __name__=="__main__":
    main()
