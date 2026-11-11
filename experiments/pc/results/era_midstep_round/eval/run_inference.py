"""Submit each frozen image/question once to one checkpoint; preserve in-flight IDs on interruption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PILOT = Path.home()/"era_self_1k_v1"
sys.path.insert(0,str(PILOT))
from infer import Runner, atomic, digest, dump, read

INPUT_HASH = {"charts":"f1be9e728a836bfecf784451c5b4dbcd5092d4b9519498ece1b9425d7ed16233",
              "science":"4538cf9358debc6eb46d01c3ce63a060784ecb4ce1377748fdd615309394c216"}
SIZES = {"teacher":192,"charts":96,"science":19}
OPTIONS = {"base","ordinary","preserved"}
BATCH = 6


def sync_directory(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inputs(dataset):
    path = (HERE.parent/"data/teacher_inputs.json" if dataset=="teacher" else HERE/"spec"/dataset/"inputs.json")
    expected = ((path.parent/"teacher_input_sha256.txt").read_text().strip().split()[0]
                if dataset=="teacher" else INPUT_HASH[dataset])
    if digest(path) != expected:
        raise RuntimeError(f"Frozen {dataset} inputs changed")
    rows=read(path)
    assert len(rows)==len({r["id"] for r in rows})==SIZES[dataset]
    keys = {"id","image","image_sha256"} | ({"question","options"} if dataset=="charts" else {"prompt"})
    seen=set()
    for r in rows:
        assert set(r)==keys and (path.parent/r["image"]).resolve().is_relative_to(path.parent.resolve())
        assert digest(path.parent/r["image"])==r["image_sha256"]
        key = (r["image_sha256"], r.get("prompt") or (r["question"],tuple(r["options"])))
        assert key not in seen
        seen.add(key)
    return path, rows


class Engine(Runner):
    def __init__(self,checkpoint):
        self.model = checkpoint.resolve()
        self.cfg = read(PILOT/"eval_config.json")
        self.batch = BATCH
        self.llm = None

    def one(self,row,root):
        if "prompt" not in row:
            return self.encode(row,root)
        import evaluate
        # Reuse the same processor, original chat template and image token checks for a pre-rendered raw prompt.
        with patch.object(evaluate, "prompt_for", return_value=row["prompt"]):
            return self.encode(dict(row, question="",options=[]),root)


def run(dataset,arm,checkpoint,check_only=False):
    import evaluate
    from vllm import SamplingParams
    assert dataset!="teacher" or arm=="base"
    root=HERE/"runs"
    folder=root/(dataset+"-"+arm)
    folder.mkdir(exist_ok=True,parents=True)
    src,rows=inputs(dataset)
    engine=Engine(checkpoint)
    weights={p.name:digest(p) for p in sorted(engine.model.glob("*.safetensors"))}
    assert weights and engine.cfg["max_new_tokens"]==2048
    max_new=4096 if dataset=="teacher" else 2048
    manifest=dict(dataset=dataset,arm=arm,checkpoint=str(engine.model),weights=weights,inputs_sha256=digest(src),
                  config_sha256=digest(PILOT/"eval_config.json"),engine_sha256=digest(PILOT/"infer.py"),
                  runner_sha256=digest(Path(__file__)),batch=BATCH,request_seed=3407,max_output=max_new,
                  teacher_budget="min(4096,8192-input tokens)", eval_budget="2048")
    with evaluate.run_lock(root/"single-consumer.lock"):
        for work in root.glob("*/manifest.json"):
            existing=read(work)
            if existing["dataset"]==dataset and existing["weights"]==weights and existing["arm"]!=arm:
                raise RuntimeError("Checkpoint already registered for this dataset under a different ID")
        file=folder/"manifest.json"
        if file.exists():
            assert read(file)==manifest
        elif not check_only:
            dump(file,manifest)
        out=folder/"predictions.jsonl"
        data=[json.loads(x) for x in out.read_text(encoding="utf8").splitlines()] if out.exists() else []
        ids=[r["id"] for r in rows]
        assert [r["id"] for r in data]==ids[:len(data)]
        intent=folder/"submitted.json"
        if intent.exists():
            pending=set(read(intent))-set(ids[:len(data)])
            if pending:raise RuntimeError("Submitted without a committed result; no reissue: "+str(sorted(pending)))
            if not check_only:intent.unlink()
        done=folder/"done.json"
        if done.exists():
            assert read(done)["output_sha256"]==digest(out) and len(data)==len(rows)
            return
        if check_only:
            print(dataset,len(rows),"inputs, identity and journals validated; zero submissions")
            return
        started=time.monotonic()
        for i in range(len(data),len(rows),BATCH):
            if engine.llm is None:
                engine.load()
            batch=rows[i:i+BATCH]
            requests=[engine.one(row,src.parent) for row in batch]
            caps=[min(max_new,8192-r[2]) for r in requests]
            if min(caps)<min(2048,max_new):raise RuntimeError("Input token budget exceeded")
            params=[SamplingParams(**dict(engine.cfg["sampling"],seed=3407),max_tokens=c,stop_token_ids=[]) for c in caps]
            dump(intent,[r["id"] for r in batch])
            sync_directory(folder)
            start=time.monotonic()
            outputs=engine.llm.generate([r[0] for r in requests],params,use_tqdm=False)
            wall=time.monotonic()-start
            assert len(outputs)==len(batch)
            for row,req,pred,cap in zip(batch,requests,outputs,caps):
                assert len(pred.outputs)==1 and len(pred.prompt_token_ids)==req[2]
                completion=pred.outputs[0]
                assert completion.finish_reason in ("stop","length")
                data.append(dict(id=row["id"],image_sha256=row["image_sha256"],prompt=req[1],seed=3407,
                                 input_tokens=req[2],max_output=cap,raw=completion.text,
                                 output_tokens=len(completion.token_ids),finish_reason=completion.finish_reason,batch_elapsed_s=wall))
            atomic(out,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in data))
            sync_directory(folder)
            intent.unlink()
            sync_directory(folder)
            if dataset=="teacher" and len(data)==24 and not (folder/"first24.json").exists():
                elapsed=time.monotonic()-started
                dump(folder/"first24.json",dict(n=24,wall_seconds=elapsed,remaining_wall_estimate_seconds=elapsed*7,
                     generated_tokens=sum(r["output_tokens"] for r in data),length=sum(r["finish_reason"]=="length" for r in data),
                     prefix_sha256=digest(out)))
            print(f"{dataset}/{arm} {len(data)}/{len(rows)}",flush=True)
        dump(done,dict(n=len(data),output_sha256=digest(out),manifest_sha256=digest(file)))


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--dataset",choices=SIZES,required=True)
    args.add_argument("--arm",choices=sorted(OPTIONS),required=True)
    args.add_argument("--checkpoint",type=Path,required=True)
    args.add_argument("--check-only",action="store_true")
    opt=args.parse_args()
    run(opt.dataset,opt.arm,opt.checkpoint,opt.check_only)
