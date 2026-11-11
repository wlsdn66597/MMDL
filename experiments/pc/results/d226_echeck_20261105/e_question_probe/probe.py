"""Blind image-pair inference with the frozen project's prompt and generation settings."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = Path.home() / "era_decoder_lora_20261030"
MODEL = BASE / "runs/era_4epoch_20261030/epoch_checkpoints/epoch_3/merged"
sys.path.insert(0,str(BASE/"baseline/src"))

def digest(path):
    return hashlib.file_digest(path.open("rb"),"sha256").hexdigest()

def main():
    global MODEL
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("arm",choices=["baseline","standard","checked"])
    arm=ap.parse_args().arm
    if arm!="baseline": MODEL=HERE.parent/"train"/arm/"merged"
    import evaluate as frozen
    import prepare
    from PIL import Image
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    c=prepare.read(HERE.parent/"eval_config.json")
    prior_config=prepare.read(BASE/"baseline/protocol/config.seed3407.json")
    assert c["max_new_tokens"]==2048 and dict(c,max_new_tokens=prior_config["max_new_tokens"])==prior_config
    prepare.runtime_check(c)
    run=HERE/"eval"/arm; run.mkdir(exist_ok=True,parents=True)
    lock=frozen.run_lock(run/"writer.lock")
    rows=prepare.read(HERE/"inputs.json")
    assert len(rows)==len({r["id"] for r in rows})==64
    identity={"model":str(MODEL),"weights":{p.name:digest(p) for p in sorted(MODEL.glob("*.safetensors"))},
              "inputs_sha256":digest(HERE/"inputs.json"),"config_sha256":digest(HERE.parent/"eval_config.json"),
              "eval_script_sha256":digest(Path(__file__)),"runner_frozen_prompt_sha256":digest(BASE/"baseline/src/evaluate.py")}
    man=run/"manifest.json"
    if man.exists() and prepare.read(man)!=identity: raise RuntimeError("Run identity mismatch")
    man.write_text(json.dumps(identity,indent=2)+"\n")
    existing=prepare.read_jsonl(run/"predictions.jsonl")
    done={r["id"] for r in existing}
    assert len(existing)==len(done) and done <= {r["id"] for r in rows}
    assert all(digest(HERE.parent/"data"/r["image"])==r["image_sha256"] for r in rows)
    if len(done)==len(rows): return
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    rt=c["runtime"]; mx=256
    engine=LLM(model=str(MODEL),dtype=rt["dtype"],quantization=None,tensor_parallel_size=rt["tensor_parallel_size"],
               max_model_len=rt["max_model_len"],gpu_memory_utilization=rt["gpu_memory_utilization"],max_num_seqs=rt["max_num_seqs"],
               max_num_batched_tokens=rt["max_num_batched_tokens"],enforce_eager=rt["enforce_eager"],
               limit_mm_per_prompt={"image":rt["max_images_per_prompt"]},seed=c["sampling"]["seed"],
               mm_processor_kwargs={"size":{"shortest_edge":c["vision"]["min_pixels"],"longest_edge":c["vision"]["max_pixels"]},"do_resize":False})
    sample=SamplingParams(**c["sampling"],max_tokens=mx,stop_token_ids=[])
    for r in rows:
        if r["id"] in done: continue
        start=time.monotonic()
        prompt=r["input_prompt"]
        with Image.open(HERE.parent/"data"/r["image"]) as source: rgb=source.convert("RGB")
        messages=[{"role":"user","content":[dict(type="image",image=rgb,**c["vision"]),dict(type="text",text=prompt)]}]
        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        imgs,vid,args=process_vision_info(messages,image_patch_size=processor.image_processor.patch_size,return_video_kwargs=True,return_video_metadata=True)
        assert vid is None
        args["do_resize"]=False
        n=processor(text=[rendered],images=imgs,return_tensors="pt",**args)["input_ids"].shape[-1]
        assert n+mx <= rt["max_model_len"]
        pred=engine.generate([dict(prompt=rendered,multi_modal_data={"image":imgs},mm_processor_kwargs=args)],sample,use_tqdm=False)[0]
        assert len(pred.prompt_token_ids)==n
        out=pred.outputs[0]; assert out.finish_reason in ("stop","length")
        prepare.append(run/"predictions.jsonl",dict(id=r["id"],image_sha256=r["image_sha256"],prompt=prompt,raw=out.text,
                       finish_reason=out.finish_reason,input_tokens=n,output_tokens=len(out.token_ids),elapsed_s=time.monotonic()-start))
        done.add(r["id"]); print(f"{len(done)}/{len(rows)} {r['id']} {len(out.token_ids)} tokens",flush=True)
    assert len(done)==64
    lock.close()

if __name__=="__main__": main()
