"""Single independent completion per chart; diagnostic predictions never re-enter a solver."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PILOT=Path.home()/"era_self_1k_v1"
sys.path.insert(0,str(PILOT))
from infer import Runner, atomic, digest, dump, read


class ChartEngine(Runner):
    def __init__(self,checkpoint,batch):
        self.model=checkpoint.resolve()
        self.batch=batch
        self.llm=None
        self.cfg=read(PILOT/"eval_config.json")
        assert self.cfg["max_new_tokens"]==2048


def main():
    import evaluate as official
    from vllm import SamplingParams
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--name",required=True)
    args=ap.parse_args()
    folder=ROOT/"charts"/args.name
    folder.mkdir(exist_ok=True,parents=True)
    spec=ROOT/"chart_spec"/"inputs.json"
    rows=read(spec)
    ids=[r["id"] for r in rows]
    assert len(ids)==len(set(ids))==96
    rt=ChartEngine(args.checkpoint,6)
    with official.run_lock(folder/"lock"):
        weights={p.name:digest(p) for p in rt.model.glob("*.safetensors")}
        assert weights
        signature=dict(weights=weights,checkpoint=str(rt.model),spec=digest(spec),script=digest(Path(__file__)),
            engine=digest(PILOT/"infer.py"),config=digest(PILOT/"eval_config.json"),batch=6,cap=2048)
        manifest=folder/"manifest.json"
        if manifest.exists():assert read(manifest)==signature
        else:dump(manifest,signature)
        output=folder/"predictions.jsonl";intent=folder/"submitted.json"
        predictions=[json.loads(s) for s in output.read_text().splitlines()] if output.is_file() else []
        assert [r["id"] for r in predictions]==ids[:len(predictions)]
        if intent.exists():
            completed={r["id"] for r in predictions}
            if not set(read(intent)).issubset(completed):
                raise RuntimeError("Prior submitted chart has no durable completion; refusing duplicate inference")
            intent.unlink()
        for i in range(len(predictions),len(rows),rt.batch):
            if rt.llm is None:rt.load()
            batch=rows[i:i+rt.batch]
            enc=[rt.encode(x,PILOT/"holdout") for x in batch]
            seeds=[int.from_bytes(hashlib.sha256(("3407/"+x["id"]).encode()).digest()[:4],"big")&0x7fffffff for x in batch]
            params=[SamplingParams(**dict(rt.cfg["sampling"],seed=s),max_tokens=2048,stop_token_ids=[]) for s in seeds]
            dump(intent,[x["id"] for x in batch]);tic=time.monotonic()
            outputs=rt.llm.generate([e[0] for e in enc],params,use_tqdm=False)
            assert len(outputs)==len(enc)
            elapsed=time.monotonic()-tic
            for x,e,k,g in zip(batch,enc,seeds,outputs):
                y=g.outputs[0]
                assert len(g.prompt_token_ids)==e[2] and y.finish_reason in ("stop","length")
                predictions.append(dict(id=x["id"],image_sha256=x["image_sha256"],raw=y.text,finish_reason=y.finish_reason,
                    prompt=e[1],seed=k,input_tokens=e[2],output_tokens=len(y.token_ids),elapsed_s=elapsed/len(batch)))
            atomic(output,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in predictions))
            intent.unlink()
            print(f"{len(predictions)}/{len(ids)}",flush=True)
        dump(folder/"done.json",dict(n=len(ids),output_sha256=digest(output)))


if __name__=="__main__":main()
