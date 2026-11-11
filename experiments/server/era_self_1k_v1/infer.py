"""Run the frozen VLM on blind chart inputs, or replay its own first stated candidates."""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parent
PROJECT=Path.home()/"era_decoder_lora_20261030"
sys.path.insert(0,str(PROJECT/"baseline/src"))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def atomic(path,text):
    temp=path.with_name(path.name+".tmp")
    with temp.open("w",encoding="utf8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def dump(path,obj):
    atomic(path,json.dumps(obj,ensure_ascii=False,indent=2)+"\n")


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def model_for(arm):
    if arm=="start":return Path.home()/"era_link_repair_pilot/models/question/merged"
    if arm.startswith("prior"):return Path.home()/"era_second_pass_v1/models"/arm[5:]/"merged"
    return ROOT/"models"/arm/"merged"


def early_gate(raw):
    e=re.match(r"\s*(?:\*\*)?Evidence(?:\*\*)?\s*:\s*(.*)",raw,re.I)
    if not e:return None
    m=re.match(r"(?:(?:my initial candidates are|initial candidates:)\s*[-\d., and]+?\.\s+)?(Check|Recheck|Direct(?: read)?)\s*[.:]",e[1],re.I)
    return not m[1].lower().startswith("direct") if m else None


def own_values(raw,rows):
    from dataprep import extract_initial
    # Refuse chart-derived numbers which the model did not actually write in its E.
    e=re.search(r"(?ims)^\s*(?:\*\*)?Evidence(?:\*\*)?\s*:\s*(.*?)(?=^\s*(?:Rule|Application|Final answer)\s*:|\Z)",raw)
    if not e:return None
    vals=extract_initial(raw,rows)
    if vals is None or len(vals)!=3:return None
    first=re.split(r"(?i)\b(?:recalculat\w*|replac\w*|corrected|revised|instead)\b",e[1],maxsplit=1)[0]
    try:
        numerals=Counter(map(Decimal,re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?",first.replace(", "," "))))
        if Counter(map(Decimal,vals))-numerals:return None
    except InvalidOperation:return None
    return [str(v) for v in vals]


def replay_inputs(blind,gold,arm):
    src=ROOT/"eval"/arm/"natural"
    assert (src/"done.json").is_file()
    first={p["id"]:p for p in map(json.loads,(src/"predictions.jsonl").read_text(encoding="utf8").splitlines())}
    rows=[];coverage=[]
    for r in blind:
        it=gold[r["id"]]
        if it["mode"]!="rank":continue
        prev=first[r["id"]]
        vals=own_values(prev["raw"],it["rows"])
        gate=early_gate(prev["raw"])
        coverage.append(dict(id=r["id"],draft=vals,policy_check=gate,reason="copied" if vals else "no extractable numeric first selection"))
        if vals is None:continue
        q=r["question"]+" A provisional selection is "+", ".join(vals)+"."
        rows.append(dict(r,question=q,draft=vals,first_pass_id=r["id"],original_first_raw=prev["raw"],policy_check=gate))
    return rows,coverage


class Runner:
    def __init__(self,arm,batch):
        self.arm=arm;self.batch=batch;self.llm=None;self.model=model_for(arm)
        self.cfg=read(ROOT/"eval_config.json")
        self.weights={p.name:digest(p) for p in sorted(self.model.glob("*.safetensors"))}
        if not self.weights:raise RuntimeError("Missing model weights "+str(self.model))

    def load(self):
        from transformers import AutoProcessor
        from vllm import LLM
        self.processor=AutoProcessor.from_pretrained(self.model,local_files_only=True)
        c=self.cfg;rt=c["runtime"];v=c["vision"]
        self.llm=LLM(model=str(self.model),dtype=rt["dtype"],quantization=None,tensor_parallel_size=1,
            max_model_len=8192,gpu_memory_utilization=rt["gpu_memory_utilization"],max_num_seqs=self.batch,
            max_num_batched_tokens=4096,enforce_eager=True,seed=c["sampling"]["seed"],
            limit_mm_per_prompt={"image":1},mm_processor_kwargs={"size":{"shortest_edge":v["min_pixels"],"longest_edge":v["max_pixels"]},"do_resize":False})

    def encode(self,row,base):
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        import evaluate as frozen
        img=(base/row["image"]).resolve()
        assert img.is_relative_to(base.resolve()) and digest(img)==row["image_sha256"]
        p=frozen.prompt_for(dict(question=row["question"],options=row["options"],question_type="multiple-choice"),self.cfg)
        with Image.open(img) as source:rgb=source.convert("RGB")
        msg=[{"role":"user","content":[dict(type="image",image=rgb,**self.cfg["vision"]),dict(type="text",text=p)]}]
        rendered=self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        imgs,vid,opts=process_vision_info(msg,image_patch_size=self.processor.image_processor.patch_size,return_video_kwargs=True,return_video_metadata=True)
        assert vid is None
        opts["do_resize"]=False
        n=self.processor(text=[rendered],images=imgs,return_tensors="pt",**opts)["input_ids"].shape[-1]
        assert n+self.cfg["max_new_tokens"]<=8192
        return dict(prompt=rendered,multi_modal_data={"image":imgs},mm_processor_kwargs=opts),p,n

    def run(self,base,folder,rows,mode,coverage=None):
        import evaluate as frozen
        import prepare
        from vllm import SamplingParams
        folder.mkdir(parents=True,exist_ok=True)
        with frozen.run_lock(folder/"inference.lock"):
            ids=[r["id"] for r in rows];assert len(ids)==len(set(ids))
            identity=dict(model=str(self.model),weights=self.weights,model_checkpoint=str(self.model),mode=mode,
                seed_base=self.cfg["sampling"]["seed"],config_sha256=digest(ROOT/"eval_config.json"),
                script_sha256=digest(Path(__file__)),batch=self.batch,max_model_len=8192,
                inputs_sha256=digest(base/"inputs.json"),requests_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
            if mode=="self":identity["extractor_sha256"]=digest(ROOT/"dataprep.py")
            manifest=folder/"manifest.json"
            if manifest.exists():assert read(manifest)==identity,"Run identity differs"
            else:dump(manifest,identity)
            if coverage is not None:dump(folder/"coverage.json",coverage)
            path=folder/"predictions.jsonl"
            recs=prepare.read_jsonl(path)
            assert [r["id"] for r in recs]==ids[:len(recs)]
            if (folder/"done.json").is_file():
                assert len(recs)==len(ids) and read(folder/"done.json")["output_sha256"]==digest(path)
                return
            for i in range(len(recs),len(rows),self.batch):
                if self.llm is None:self.load()
                chunk=rows[i:i+self.batch];tic=time.monotonic()
                encoded=[self.encode(r,base) for r in chunk]
                seeds=[int.from_bytes(hashlib.sha256(f'{self.cfg["sampling"]["seed"]}/{r["id"]}'.encode()).digest()[:4],"big") & 0x7fffffff for r in chunk]
                params=[SamplingParams(**dict(self.cfg["sampling"],seed=s),max_tokens=self.cfg["max_new_tokens"],stop_token_ids=[]) for s in seeds]
                predictions=self.llm.generate([x[0] for x in encoded],params,use_tqdm=False)
                assert len(predictions)==len(chunk)
                elapsed=time.monotonic()-tic
                for r,enc,k,out in zip(chunk,encoded,seeds,predictions):
                    result=out.outputs[0];assert len(out.prompt_token_ids)==enc[2] and result.finish_reason in ("stop","length")
                    extra={key:r[key] for key in ("draft","first_pass_id","original_first_raw","policy_check") if key in r}
                    recs.append(dict(id=r["id"],image_sha256=r["image_sha256"],prompt=enc[1],raw=result.text,
                        input_tokens=enc[2],output_tokens=len(result.token_ids),finish_reason=result.finish_reason,seed=k,
                        elapsed_s=elapsed/len(chunk),batch_wall_s=elapsed,**extra))
                atomic(path,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in recs))
                print(f"{mode} {len(recs)}/{len(ids)} {elapsed:.2f}s batch",flush=True)
            if not path.exists():atomic(path,"")
            dump(folder/"done.json",dict(n=len(recs),output_sha256=digest(path)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=("start","priorcontrol","priorpaired","A","B"),default="start")
    ap.add_argument("--probe",type=int);ap.add_argument("--batch",type=int,default=6)
    args=ap.parse_args()
    import prepare
    prepare.runtime_check(read(ROOT/"eval_config.json"))
    run=Runner(args.arm,args.batch)
    if args.probe is not None:
        assert args.arm=="start"
        base=ROOT/"probe"/f"{args.probe:03d}"
        run.run(base,base,read(base/"inputs.json"),"probe")
        return
    base=ROOT/"holdout";blind=read(base/"inputs.json");gold={r["id"]:r for r in read(base/"gold.json")}
    out=ROOT/"eval"/args.arm
    for mode in ("natural","common"):
        rows=[r for r in blind if (gold[r["id"]]["mode"]=="draft")== (mode=="common")]
        run.run(base,out/mode,rows,mode)
    if args.arm in ("start","A","B"):
        rows,coverage=replay_inputs(blind,gold,args.arm)
        run.run(base,out/"self",rows,"self",coverage)


if __name__=="__main__":main()
