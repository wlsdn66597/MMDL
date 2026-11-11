"""Continue training from the verified merged ERA epoch-3 checkpoint, one matched D arm per process."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
PROJECT=Path.home()/"era_decoder_lora_20261030"
START=Path.home()/"d226_echeck_20261105/train/checked/merged"
sys.path.insert(0,str(PROJECT))
from era_lora.common import canonical_json_hash, sha256_file, write_json
from era_lora.data import AssistantOnlyCollator, PairedSFTDataset, load_processor, validate_encoded_pair
from era_lora.train import (assert_trainables,discover_decoder_targets,file_lock,make_audit_callback,
                            parameter_digest,runtime_check)

ARMS={"statement":"plain","question":"era"}
def read(p):return json.loads(p.read_text(encoding="utf8"))
def verify_start():
    actual={p.name:sha256_file(p) for p in START.glob("*.safetensors")}
    assert len(actual)==3 and actual==read(START.parent/"merge.json")["weights"]
    return actual
def records():
    file=ROOT/"paired_train.jsonl";receipt=read(ROOT/"target_receipt.json")
    assert sha256_file(file)==receipt["paired_sha256"]
    raw=list(map(json.loads,file.read_text().splitlines()));data=[]
    assert len(raw)==receipt["included"]==len({r["sample_id"] for r in raw})
    for r in raw:
        img=(ROOT/r["image"]).resolve();assert img.is_relative_to(ROOT/"train/images") and sha256_file(img)==r["image_sha256"]
        assert r["statement"].partition("\n")[2]==r["question"].partition("\n")[2]
        data.append(dict(sample_id=r["sample_id"],input_prompt=r["input_prompt"],image_path=str(img),image_sha256=r["image_sha256"],response_plain=r["statement"],response_era=r["question"]))
    return data

def run(args):
    import torch
    from peft import LoraConfig,TaskType,get_peft_model,PeftModel
    from transformers import Qwen3VLForConditionalGeneration,Trainer,TrainingArguments,set_seed
    from era_lora.merge import copy_processor_files
    cfg=read(PROJECT/"configs/era_4epoch_20261030.json")
    cfg["model"]["local_path"]=str(START);cfg["training"]["epochs"]=3
    cfg["training"]["output_root"]=str(ROOT/"models")
    cfg["_config_dir"]=str(ROOT); cfg["_config_path"]=str(PROJECT/"configs/era_4epoch_20261030.json")
    settings=cfg["training"];lora=cfg["lora"];seed=settings["seed"]
    root=ROOT/"models";root.mkdir(exist_ok=True)
    save=root/args.arm
    weights=verify_start();rows=records()
    identity=dict(start=str(START),start_weights=weights,training=settings,lora=lora,data_hashes=[sha256_file(ROOT/"paired_train.jsonl")],
                  input_order=canonical_json_hash([r["sample_id"] for r in rows]),wrapper_sha256=sha256_file(Path(__file__)))
    env=runtime_check()
    kwargs=dict(dtype=torch.bfloat16,attn_implementation=settings["attn_implementation"],local_files_only=True,
                low_cpu_mem_usage=True,use_safetensors=True)
    if args.merge:
        man=read(save/"manifest.json"); assert man["complete"] and man["identity"]==identity
        if (save/"merged").exists():
            assert (save/"merge.json").is_file();return
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(START),**kwargs)
        model=PeftModel.from_pretrained(model,str(save/"adapter"),is_trainable=False)
        merged=model.eval().merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
        assert not any("lora_" in n for n,_ in merged.named_parameters())
        merged.config.use_cache=True
        tmp=save/"merged.tmp";tmp.mkdir()
        merged.save_pretrained(tmp,safe_serialization=True,max_shard_size="4GB")
        copy_processor_files(START,tmp)
        shards={p.name:sha256_file(p) for p in sorted(tmp.glob("*.safetensors"))};assert shards
        meta=dict(arm=args.arm,weights=shards,parent=weights);write_json(tmp/"artifact_manifest.json",meta);tmp.rename(save/"merged");write_json(save/"merge.json",meta)
        return
    with file_lock(root/("."+args.arm+".lock")):
        if (save/"manifest.json").exists():
            man=read(save/"manifest.json");assert man["complete"] and man["identity"]==identity;return
        if save.exists() and any(save.iterdir()):raise RuntimeError("Incomplete run; preserve attempt before retrying")
        save.mkdir(exist_ok=True)
        processor=load_processor(cfg)
        collator=AssistantOnlyCollator(processor,cfg)
        if not (root/"paired_preflight.json").exists():
            print("checking all input/token pairs",flush=True)
            paired=[validate_encoded_pair(r,processor,cfg,collator=collator) for r in rows]
            write_json(root/"paired_preflight.json",paired)
        reports=read(root/"paired_preflight.json")
        assert len(reports)==len(rows) and all(r["paired_prefix_and_image_grid_equal"] for r in reports)
        set_seed(seed)
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(START),**kwargs)
        model.requires_grad_(False);targets=discover_decoder_targets(model)
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,target_modules=targets,r=lora["r"],
            lora_alpha=lora["alpha"],lora_dropout=lora["dropout"],bias="none",init_lora_weights=True))
        trainables=assert_trainables(model,targets)
        model.config.use_cache=False;model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        initial=parameter_digest(model,True); frozen=parameter_digest(model,False)
        pair_meta=dict(identity=identity,initial_adapter=initial,frozen_parameters=frozen)
        if (root/"matched_init.json").exists():assert read(root/"matched_init.json")==pair_meta
        else:write_json(root/"matched_init.json",pair_meta)
        sequence=[]
        def encode(features):
            sequence.append(features[0]["sample_id"])
            return collator(features)
        arguments=TrainingArguments(output_dir=str(save/"state"),num_train_epochs=3,per_device_train_batch_size=1,
            gradient_accumulation_steps=16,learning_rate=settings["learning_rate"],weight_decay=settings["weight_decay"],
            warmup_ratio=settings["warmup_ratio"],lr_scheduler_type=settings["lr_scheduler_type"],max_grad_norm=1.,
            adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,optim="adamw_torch",bf16=True,fp16=False,
            gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},seed=seed,data_seed=seed,
            dataloader_num_workers=0,dataloader_drop_last=False,remove_unused_columns=False,save_strategy="no",
            eval_strategy="no",logging_strategy="steps",logging_steps=1,report_to=[],disable_tqdm=True)
        trainer=Trainer(model=model,args=arguments,train_dataset=PairedSFTDataset(rows,ARMS[args.arm]),data_collator=encode,
                        callbacks=[make_audit_callback(save,identity,frozen)])
        trainer.model_accepts_loss_kwargs=False
        result=trainer.train()
        assert len(sequence)==3*len(rows) and set(Counter(sequence).values())=={3} and trainer.state.epoch==3
        assert parameter_digest(model,False)==frozen
        final=parameter_digest(model,True);assert final!=initial
        model.config.use_cache=True;model.save_pretrained(save/"adapter",safe_serialization=True)
        write_json(save/"seen_ids.json",sequence)
        peer=root/("statement" if args.arm=="question" else "question")
        if (peer/"seen_ids.json").exists():assert read(peer/"seen_ids.json")==sequence
        write_json(save/"manifest.json",dict(complete=True,identity=identity,arm=args.arm,environment=env,
            actual_steps=trainer.state.global_step,epoch=trainer.state.epoch,trainable=sum(r["numel"] for r in trainables),
            frozen_verified=True,initial_adapter=initial,final_adapter=final,sequence_sha256=canonical_json_hash(sequence),
            metrics=result.metrics,preflight_sha256=sha256_file(root/"paired_preflight.json")))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("arm",choices=ARMS);parser.add_argument("--merge",action="store_true")
    run(parser.parse_args())
