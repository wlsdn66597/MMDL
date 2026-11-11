"""For original selected D prompts, compare the two annotated first tokens after Evidence:."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.home()/"era_decoder_lora_20261030"/"baseline/src"))
def read(path):return json.loads(path.read_text())
def rows(path):return {r["id"]:r for r in map(json.loads,path.read_text().splitlines())}
def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    ap=argparse.ArgumentParser();ap.add_argument("arm",choices=["standard","checked"]);arm=ap.parse_args().arm
    source=ROOT.parent
    weights=source/"train"/arm/"merged";cfg=read(source/"eval_config.json")
    items=read(ROOT/"inputs.json");standard,checked=[rows(source/"data"/(a+".jsonl")) for a in ("standard","checked")]
    proc=AutoProcessor.from_pretrained(weights,local_files_only=True)
    mdl=Qwen3VLForConditionalGeneration.from_pretrained(weights,dtype=torch.bfloat16,attn_implementation="sdpa",device_map={"":"cuda"},local_files_only=True).eval()
    output=[]
    for i in items:
        id_=i["id"];a,b=standard[id_],checked[id_]
        if a["target"]==b["target"]:continue
        with Image.open(source/"data"/i["image"]) as f: im=f.convert("RGB")
        messages=[{"role":"user","content":[dict(type="image",image=im,**cfg["vision"]),dict(type="text",text=i["input_prompt"])]}]
        prefix=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        tok=proc.tokenizer;head=tok.encode(prefix+"Evidence:",add_special_tokens=False)
        candidates=[]
        for row in (a,b):
            ids=tok.encode(prefix+row["target"],add_special_tokens=False)
            assert head==ids[:len(head)]
            candidates.append(ids[len(head)])
        ct,qt=candidates
        if ct==qt:
            output.append(dict(id=id_,same_initial_token=qt));continue
        imgs,vid,extra=process_vision_info(messages,image_patch_size=proc.image_processor.patch_size,return_video_kwargs=True,return_video_metadata=True)
        assert vid is None
        x=proc(text=[prefix+"Evidence:"],images=imgs,return_tensors="pt",**dict(extra,do_resize=False)).to("cuda")
        with torch.inference_mode(): scores=mdl(**x,logits_to_keep=1,use_cache=False).logits[0,-1].float()
        probs=torch.softmax(scores,dim=-1);best=int(probs.argmax())
        output.append(dict(id=id_,standard_token=tok.decode([ct]),question_token=tok.decode([qt]),top=tok.decode([best]),
            prob_standard=float(probs[ct]),prob_question=float(probs[qt]),raw_log_odds=float(scores[qt]-scores[ct]),
            rank_question=int((scores>scores[qt]).sum())+1))
    assert len(output)==47
    (ROOT/f"{arm}_next_token.json").write_text(json.dumps(output,indent=2)+"\n")
if __name__=="__main__":main()
