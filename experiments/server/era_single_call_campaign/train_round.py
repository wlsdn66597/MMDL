"""Train a single-pass ERA adapter against its same-data, same-budget ERA control."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent
PROJECT=Path.home()/"era_decoder_lora_20261030"
CFG=PROJECT/"configs/era_4epoch_20261030.json"
HASHES={
    "model-00001-of-00002.safetensors":"30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    "model-00002-of-00002.safetensors":"046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
}
# No adaptive tuning on individual MMMU responses. The external campaign chooses branches by aggregate gates.
ROUNDS={
    1:dict(name="broad_light",epochs=1,lr=5e-5,anchor=1.0,final=.5),
    2:dict(name="base_retention",epochs=1,lr=2e-5,anchor=1.0,final=.5),
    3:dict(name="link_strength",epochs=2,lr=3.5e-5,anchor=2.0,final=.5),
}
STOP=set("these those that this with from after before given therefore because hence thus option answer question specified shown displayed evidence rule application result values value selected chart image figure final matching select which largest smallest smallest measured than into using requested the are was were their have has two three each there other both".split())
FACT=re.compile(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?%?|[A-Za-z][A-Za-z-]{2,}")
SECTIONS=re.compile(r"^Evidence: (.+?)\nRule: (.+?)\nApplication: (.+?)\nFinal answer: (.+?)$",re.S)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def focus_spans(text):
    m=SECTIONS.fullmatch(text);assert m,text
    atoms=[]
    for n in (1,2,3):
        tokens=[]
        for w in FACT.finditer(m[n]):
            k=w[0].lower()
            if k not in STOP:
                tokens.append((k,(m.start(n)+w.start(),m.start(n)+w.end())))
        atoms.append(tokens)
    upstream={k for group in atoms[:2] for k,_ in group}
    downstream={k for k,_ in atoms[2]}
    links=[span for group in atoms for k,span in group if k in upstream & downstream]
    return sorted(links),m.span(4)


def baseline():
    c=read(CFG)
    snapshot=Path(c["model"]["local_path"])
    assert snapshot.name=="ebb281ec70b05090aa6165b016eac8ec08e71b17"
    actual={p.name:sha(p) for p in snapshot.glob("*.safetensors")}
    assert actual==HASHES and not (snapshot/"adapter_config.json").exists()
    return c,snapshot


def trainer_module():
    # Use the optimizer/cursor implementation for which bit-identical crash recovery was established.
    p=Path.home()/"era_self_1k_v1/train.py"
    verified=read(p.parent/"recovery/verified.json")
    assert sha(p)==verified["train_sha256"]=="cbfa08063e08c85188f5fe369fa98a9b9f3347ac199e573f250cbb3d59ab2c4b"
    assert all(verified[k] for k in ("exact_optimizer","exact_scheduler","exact_rng"))
    spec=importlib.util.spec_from_file_location("recovery_helpers",p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def dataset():
    d=ROOT/"data"
    rec=read(d/"receipt.json")["splits"]["train"]
    assert sha(d/"train.jsonl")==rec["sha256"]
    rows=[]
    for line in (d/"train.jsonl").read_text(encoding="utf8").splitlines():
        x=json.loads(line);img=(d/x["image"]).resolve()
        assert img.is_relative_to((d/"images").resolve()) and sha(img)==x["image_sha256"]
        rows.append(dict(sample_id=x["id"],input_prompt=x["prompt"],image_path=str(img),
                         image_sha256=x["image_sha256"],response=x["target"]))
    assert len(rows)==len({x["sample_id"] for x in rows})==rec["n"]
    return rows


def focus_weights(feature,report,batch,processor,hp,arm):
    import torch
    text=feature["response"]
    ids=processor.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    first=report["input_tokens"]
    assert batch["input_ids"][0,first:first+len(ids.input_ids)].tolist()==ids.input_ids
    positions=batch["labels"][0].ne(-100).nonzero().flatten()
    assert positions.tolist()==list(range(first,first+len(ids.input_ids)+1))
    assert batch["labels"][0,positions[-1]].item()==report["assistant_end_token_id"]
    spans,final=focus_spans(text)
    weights=[1.0]*len(positions)
    if arm=="candidate":
        for i,(a,b) in enumerate(ids.offset_mapping):
            if any(a<d and b>c for c,d in spans):weights[i]+=hp["anchor"]
            if a<final[1] and b>final[0]:weights[i]+=hp["final"]
    return positions,torch.tensor(weights,dtype=torch.float32)


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from peft import LoraConfig,TaskType,get_peft_model,set_peft_model_state_dict,PeftModel
    from safetensors.torch import load_file
    from transformers import Qwen3VLForConditionalGeneration,get_scheduler,set_seed
    sys.path.insert(0,str(PROJECT))
    from era_lora.common import write_json,canonical_json_hash
    from era_lora.train import parameter_digest,assert_trainables,discover_decoder_targets,runtime_check,file_lock,validate_gradient_ownership
    from era_lora.data import AssistantOnlyCollator,load_processor
    from era_lora.merge import copy_processor_files

    old=trainer_module();cfg,start=baseline();hp=ROUNDS[args.round]
    root=ROOT/"models"/f"r{args.round}";out=root/args.arm;out.mkdir(exist_ok=True,parents=True)
    cfg["_config_dir"]=str(PROJECT)
    train=cfg["training"];lora=cfg["lora"];seed=train["seed"];acc_steps=train["gradient_accumulation_steps"]
    kwargs=dict(dtype=torch.bfloat16,attn_implementation=train["attn_implementation"],local_files_only=True,
                low_cpu_mem_usage=True,use_safetensors=True)
    with file_lock(out/".writer.lock"):
        if args.merge:
            man=read(out/"manifest.json")
            assert man["complete"] and sha(out/"adapter/adapter_model.safetensors")==man["adapter_file_sha256"]
            merged=out/"merged"
            if not merged.exists():
                base=Qwen3VLForConditionalGeneration.from_pretrained(str(start),**kwargs)
                model=PeftModel.from_pretrained(base,str(out/"adapter"),is_trainable=False)
                model=model.eval().merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
                assert not any("lora_" in n for n,_ in model.named_parameters())
                model.config.use_cache=True
                temp=out/(".merged."+uuid.uuid4().hex+".partial")
                model.save_pretrained(temp,safe_serialization=True,max_shard_size="4GB")
                copy_processor_files(start,temp)
                weights={p.name:sha(p) for p in temp.glob("*.safetensors")}
                assert weights;write_json(temp/"artifact_manifest.json",dict(weights=weights,parent=HASHES,adapter=man["adapter_file_sha256"]))
                temp.rename(merged)
            meta=read(merged/"artifact_manifest.json")
            assert meta["parent"]==HASHES and meta["adapter"]==man["adapter_file_sha256"] and meta["weights"]=={p.name:sha(p) for p in merged.glob("*.safetensors")}
            write_json(out/"merge.json",meta);return
        env=runtime_check();rows=dataset();epochs=hp["epochs"]
        schedule=old.sample_schedule(len(rows),epochs,seed)
        sequence=[rows[i]["sample_id"] for epoch in schedule for i in epoch]
        per=math.ceil(len(rows)/acc_steps);steps=epochs*per
        identity=dict(arm=args.arm,round=args.round,hyperparameters=hp,parent=HASHES,
            receipt=sha(ROOT/"data/receipt.json"),code=sha(Path(__file__)),train_modules={x:sha(PROJECT/"era_lora"/x) for x in ("train.py","data.py","common.py")},
            train_config=sha(CFG),helper=sha(Path(old.__file__)),versions=env["versions"],seed=seed,accumulation=acc_steps,steps=steps,sequence=canonical_json_hash(sequence))
        if (out/"identity.json").exists():assert read(out/"identity.json")==identity
        else:write_json(out/"identity.json",identity)
        if (out/"manifest.json").exists():
            man=read(out/"manifest.json")
            assert man["complete"] and man["identity"]==identity and sha(out/"adapter/adapter_model.safetensors")==man["adapter_file_sha256"]
            return
        processor=load_processor(cfg);collator=AssistantOnlyCollator(processor,cfg)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True,warn_only=False)
        set_seed(seed);accel=Accelerator(mixed_precision="bf16");assert accel.num_processes==1 and accel.scaler is None
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(start),**kwargs)
        model.requires_grad_(False);targets=discover_decoder_targets(model)
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,target_modules=targets,r=lora["r"],
            lora_alpha=lora["alpha"],lora_dropout=lora["dropout"],bias="none",init_lora_weights=True))
        trainables=assert_trainables(model,targets)
        frozen=parameter_digest(model,False);initial=parameter_digest(model,True)
        matched=dict(frozen=frozen,initial_adapter=initial,sequence=identity["sequence"],base=HASHES)
        with file_lock(root/".initial.lock"):
            if (root/"matched_init.json").exists():assert read(root/"matched_init.json")==matched
            else:write_json(root/"matched_init.json",matched)
        model.config.use_cache=False;model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=hp["lr"],weight_decay=train["weight_decay"])
        scheduler=get_scheduler(train["lr_scheduler_type"],optimizer,num_warmup_steps=math.ceil(steps*train["warmup_ratio"]),num_training_steps=steps)
        model,optimizer=accel.prepare(model,optimizer);raw=accel.unwrap_model(model)
        folder=out/"checkpoints";ck=old.select_checkpoint(folder,identity);step=0;seen=[];prior=0.;counts=dict(supervised=0,upweighted=0)
        if ck:
            cur=read(ck/"cursor.json");step=cur["step"];seen=cur["seen_ids"];counts=cur["counts"];prior=cur["training_seconds"]
            assert seen==sequence[:len(seen)]
            restored=set_peft_model_state_dict(raw,load_file(str(ck/"adapter_model.safetensors")))
            assert not restored.unexpected_keys and parameter_digest(raw,True)==cur["adapter_digest"]
            optimizer.load_state_dict(torch.load(ck/"optimizer.pt",map_location="cpu",weights_only=True))
            scheduler.load_state_dict(torch.load(ck/"scheduler.pt",weights_only=True))
            assert scheduler.last_epoch==step
            old.restore_rng(torch.load(ck/"rng.pt",map_location="cpu",weights_only=False))
        tick=time.monotonic();elapsed=prior
        while step<steps:
            epoch,group=divmod(step,per);ixes=schedule[epoch][group*acc_steps:(group+1)*acc_steps];optimizer.zero_grad(set_to_none=True);losses=[]
            for ix in ixes:
                feat=rows[ix];batch,rep=collator.encode(feat)
                pos,ws=focus_weights(feat,rep,batch,processor,hp,args.arm)
                labels=batch.pop("labels")[0,pos].to(accel.device)
                batch={k:v.to(accel.device) for k,v in batch.items()}
                ws=ws.to(accel.device)
                with accel.autocast():
                    # Derive all gradients from the one supervised training forward pass.
                    logits=model(**batch,logits_to_keep=(pos-1).to(accel.device)).logits[0]
                    assert logits.shape[0]==len(labels)
                    loss=(F.cross_entropy(logits.float(),labels,reduction="none")*ws).sum()/ws.sum()
                if not bool(torch.isfinite(loss)):raise ValueError(f"nonfinite loss for {feat['sample_id']}")
                accel.backward(loss/len(ixes));losses.append(loss.detach().item());seen.append(feat["sample_id"])
                counts["supervised"]+=len(labels);counts["upweighted"]+=int((ws>1).sum())
            if step==0:write_json(out/"gradient_audit.json",validate_gradient_ownership(raw))
            accel.clip_grad_norm_(model.parameters(),train["max_grad_norm"])
            optimizer.step();scheduler.step();step+=1
            elapsed=prior+time.monotonic()-tick
            progress=dict(step=step,steps=steps,seen=len(seen),loss=sum(losses)/len(losses),training_seconds=elapsed,eta_seconds=elapsed/step*(steps-step))
            write_json(out/"progress.json",progress);print(json.dumps(progress),flush=True)
            if step%12==0 or step==steps:
                cur=dict(identity=identity,step=step,seen_ids=seen,training_seconds=elapsed,counts=counts,adapter_digest=parameter_digest(raw,True))
                old.checkpoint(folder,raw,optimizer,scheduler,cur)
        assert seen==sequence and parameter_digest(raw,False)==frozen and parameter_digest(raw,True)!=initial
        assert (counts["upweighted"]>0)==(args.arm=="candidate")
        old.save_final(out,old.select_checkpoint(folder,identity))
        write_json(out/"manifest.json",dict(complete=True,identity=identity,arm=args.arm,actual_steps=step,epoch=epochs,
            seen=len(sequence),sequence_sha256=identity["sequence"],initial_adapter=initial,final_adapter=parameter_digest(raw,True),
            adapter_file_sha256=sha(out/"adapter/adapter_model.safetensors"),training_seconds=elapsed,objective_tokens=counts,
            trainable=sum(r["numel"] for r in trainables),frozen_verified=True,environment=env))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--round",type=int,choices=ROUNDS,required=True)
    parser.add_argument("--arm",choices=("control","candidate"),required=True)
    parser.add_argument("--merge",action="store_true")
    run(parser.parse_args())
