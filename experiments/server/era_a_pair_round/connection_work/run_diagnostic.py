"""Run one completion per (checkpoint, new diagnostic item). Gold never enters this process."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = HERE / "spec"
PILOT = Path.home() / "era_self_1k_v1"
sys.path.insert(0, str(PILOT))
from infer import Runner, atomic, digest, dump, read


def flush_dir(folder):
    fd = os.open(folder, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Engine(Runner):
    def __init__(self, checkpoint):
        self.model = checkpoint.resolve()
        self.cfg = read(PILOT / "eval_config.json")
        self.batch = 6
        self.llm = None
        assert self.cfg["max_new_tokens"] == 2048


def run(checkpoint, name):
    import evaluate
    from vllm import SamplingParams
    root = HERE / "runs"
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    rt = Engine(checkpoint)
    spec = SPEC / "inputs.json"
    expect = read(SPEC / "manifest.json")
    assert digest(spec) == expect["inputs_sha256"]
    items = read(spec)
    ids = [r["id"] for r in items]
    assert len(ids) == len(set(ids)) == 48
    for item in items:
        assert set(item) == {"id", "image", "image_sha256","question","options"}
        assert digest(SPEC/item["image"]) == item["image_sha256"]
    with evaluate.run_lock(root / "one-consumer.lock"):
        weights = {p.name:digest(p) for p in sorted(rt.model.glob("*.safetensors"))}
        assert weights
        manifest = dict(weights=weights, checkpoint=str(rt.model), inputs=digest(spec),
                        config=digest(PILOT/"eval_config.json"), engine=digest(PILOT/"infer.py"),
                        source=digest(Path(__file__)), batch=rt.batch, max_new_tokens=2048)
        saved = folder / "manifest.json"
        for other in root.glob("*/manifest.json"):
            if other != saved and read(other)["weights"] == weights:
                raise RuntimeError(f"Checkpoint already registered under {other}; reuse its output")
        if saved.exists():
            assert read(saved)==manifest, "Run inputs changed"
        else:
            dump(saved,manifest)
        output = folder / "predictions.jsonl"
        data = [json.loads(l) for l in output.read_text(encoding="utf8").splitlines()] if output.exists() else []
        assert [r["id"] for r in data] == ids[:len(data)]
        intent = folder / "submitted.json"
        if intent.exists():
            missing = set(read(intent)) - {r["id"] for r in data}
            if missing:
                raise RuntimeError(f"Refusing to regenerate submitted items with no saved output: {sorted(missing)}")
            intent.unlink()
        done = folder / "done.json"
        if done.exists():
            assert len(data)==len(items) and read(done)["output_sha256"]==digest(output)
            return
        for i in range(len(data),len(items),rt.batch):
            if rt.llm is None:
                rt.load()
            rows = items[i:i+rt.batch]
            encoded = [rt.encode(r,SPEC) for r in rows]
            def seed(rid):
                return int.from_bytes(hashlib.sha256(("3407/"+rid).encode()).digest()[:4],"big") & 0x7fffffff
            seeds = [seed(r["id"]) for r in rows]
            params = [SamplingParams(**dict(rt.cfg["sampling"],seed=s),max_tokens=2048,stop_token_ids=[]) for s in seeds]
            dump(intent,[r["id"] for r in rows])
            flush_dir(folder)
            start = time.monotonic()
            outputs = rt.llm.generate([enc[0] for enc in encoded],params,use_tqdm=False)
            assert len(outputs)==len(rows)
            wall = time.monotonic()-start
            for r,enc,s,out in zip(rows,encoded,seeds,outputs):
                assert len(out.outputs)==1 and len(out.prompt_token_ids)==enc[2]
                result = out.outputs[0]
                assert result.finish_reason in ("stop","length")
                data.append(dict(id=r["id"],image_sha256=r["image_sha256"],prompt=enc[1],seed=s,input_tokens=enc[2],
                                 raw=result.text,output_tokens=len(result.token_ids),finish_reason=result.finish_reason,
                                 batch_elapsed_s=wall))
            atomic(output,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in data))
            flush_dir(folder)
            intent.unlink()
            print(f"{name}: {len(data)}/{len(items)}",flush=True)
        dump(done,dict(n=len(items), output_sha256=digest(output)))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--name",choices=["base","control","paired"],required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    args=ap.parse_args()
    run(args.checkpoint,args.name)
