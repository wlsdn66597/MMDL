"""Single training-only warm start, then exact matched unscaled/contracted LoRA exports."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import train_round as common

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/"models/retention"
PROJECT=common.PROJECT
BASE=common.HASHES
SCALE={"control":1.0,"candidate":0.8}
LR=1e-5
DONOR=PROJECT/"runs/era_4epoch_20261030/train/era/checkpoint-39"
DONOR_SHA="4d0b6454053575bc8e54ead5a166b409bdca459757ce882f6933c3891448b3b3"
DONOR_CFG_SHA="e0be0cb640fe3757bc056e6f7e3277d13d54b145e1229d76b0a0c20a7b0c51a1"
DONOR_STATE_SHA="8c0aa45579c742032801e8d4eb467c1f3243ea390653ea131b8a98c6747cbd81"
DONOR_RUN_SHA="1fbcda193147aad2752a3c7d1f65dd6153c0439ff4ecfc867b0ff710090309a7"
DONOR_EXPORT_SHA="323fd434babdd4693b809072eaa3c5681f4749b5cfbefb204544367cd8b57f35"
read=common.read
sha=common.sha


def verify_donor():
    export=DONOR.parents[2]/"epoch_checkpoints/epoch_3/merged/artifact_manifest.json"
    run=DONOR.parent/"run_manifest.json"
    for file,h in ((DONOR/"adapter_model.safetensors",DONOR_SHA),(DONOR/"adapter_config.json",DONOR_CFG_SHA),
                   (DONOR/"trainer_state.json",DONOR_STATE_SHA),(run,DONOR_RUN_SHA),(export,DONOR_EXPORT_SHA)):
        assert sha(file)==h, file
    e=read(export); p=read(run); state=read(DONOR/"trainer_state.json"); cfg=read(DONOR/"adapter_config.json")
    assert e["training_identity"]==p["identity"] and e["checkpoint"]==str(DONOR)
    assert e["optimizer_step"]==state["global_step"]==39 and e["epoch"]==state["epoch"]==3
    assert e["adapter_weights_sha256"]==DONOR_SHA and p["identity"]["arm"]=="era"
    assert p["identity"]["base_revision"]=="ebb281ec70b05090aa6165b016eac8ec08e71b17"
    assert all(p["base_files_sha256"][name]==h for name,h in BASE.items())
    assert (cfg["r"],cfg["lora_alpha"],cfg["lora_dropout"],cfg["bias"])==(16,32,.05,"none")
    assert not any(cfg.get(k) for k in ("use_rslora","use_dora","rank_pattern","alpha_pattern","modules_to_save","fan_in_fan_out"))
    return dict(checkpoint=str(DONOR),step=39,epoch=3,adapter_sha256=DONOR_SHA,config_sha256=DONOR_CFG_SHA,
                trainer_state_sha256=DONOR_STATE_SHA,run_sha256=DONOR_RUN_SHA,epoch_export_sha256=DONOR_EXPORT_SHA,
                original_train_identity=p["identity"])


def factor_pairs(weights):
    starts=[k[:-len(".lora_A.weight")] for k in weights if k.endswith(".lora_A.weight")]
    assert len(starts)>0
    expected={f"{mod}.lora_{kind}.weight" for mod in starts for kind in ("A","B")}
    assert set(weights)==expected,"Adapter contains something besides paired LoRA factors"
    for mod in sorted(starts):
        a=mod+".lora_A.weight"; b=mod+".lora_B.weight"
        assert len(weights[a].shape)==len(weights[b].shape)==2 and weights[a].shape[0]==weights[b].shape[1]
        yield a,b


def project(weights,scale):
    """W0+scale*(alpha/r)BA: leave every A alone, multiply only B by scale."""
    assert 0 < scale <= 1
    scaled={}
    for a,b in factor_pairs(weights):
        scaled[a]=weights[a]
        scaled[b]=weights[b] if scale==1 else weights[b]*scale
    return scaled


def validate_projection(source,actual,scale):
    assert set(source)==set(actual)
    changed=0
    pairs=0
    for a,b in factor_pairs(source):
        pairs+=1
        for name,target in ((a,source[a]),(b,source[b] if scale==1 else source[b]*scale)):
            assert actual[name].dtype==target.dtype and actual[name].shape==target.shape
            assert bool((target==actual[name]).all()),name
        changed+=int(bool((source[b]!=actual[b]).any()))
    assert changed==0 if scale==1 else changed>0
    return dict(pairs=pairs,all_A_unchanged=True,all_B_scaled_exactly=True,changed_B_tensors=changed,delta_scale=scale)


def train_shared(config,base,donor):
    import torch
    from accelerate import Accelerator
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen3VLForConditionalGeneration, get_scheduler, set_seed
    from era_lora.common import canonical_json_hash, write_json
    from era_lora.data import AssistantOnlyCollator, load_processor
    from era_lora.train import (assert_trainables, discover_decoder_targets, file_lock, parameter_digest,
                                runtime_check, validate_gradient_ownership)

    helper=common.trainer_module()
    out=OUTPUT/"shared"
    out.mkdir(exist_ok=True,parents=True)
    with file_lock(out/".writer.lock"):
        env=runtime_check()
        rows=common.dataset()
        assert len(rows)==len({x["sample_id"] for x in rows})==449
        t=config["training"]; seed=t["seed"]; acc=t["gradient_accumulation_steps"]
        assert acc==16 and seed==3407
        schedule=helper.sample_schedule(len(rows),1,seed)[0]
        sequence=[rows[i]["sample_id"] for i in schedule]
        steps=math.ceil(len(rows)/acc)
        identity=dict(parent=BASE,donor=donor,data_sha256=sha(ROOT/"data/train.jsonl"),recipe_receipt_sha256=sha(ROOT/"data/receipt.json"),
                      training_sources={str(path):sha(path) for path in (Path(__file__),Path(common.__file__),Path(helper.__file__),common.CFG)},
                      seed=seed,epochs=1,steps=steps,accumulation=acc,lr=LR,algorithm="uniform assistant-token SFT from donor",
                      sequence=canonical_json_hash(sequence),software=env["versions"])
        if (out/"identity.json").exists():assert read(out/"identity.json")==identity
        else:write_json(out/"identity.json",identity)
        if (out/"manifest.json").exists():
            man=read(out/"manifest.json")
            assert man["complete"] and man["identity"]==identity and sha(out/"adapter/adapter_model.safetensors")==man["adapter_file_sha256"]
            return man
        config["_config_dir"]=str(PROJECT)
        processor=load_processor(config)
        collator=AssistantOnlyCollator(processor,config)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True,warn_only=False)
        set_seed(seed)
        accel=Accelerator(mixed_precision="bf16")
        assert accel.num_processes==1 and accel.scaler is None
        w0=Qwen3VLForConditionalGeneration.from_pretrained(str(base),dtype=torch.bfloat16,attn_implementation=t["attn_implementation"],
                                                           local_files_only=True,low_cpu_mem_usage=True,use_safetensors=True)
        w0.requires_grad_(False)
        targets=discover_decoder_targets(w0)
        model=PeftModel.from_pretrained(w0,str(DONOR),is_trainable=True)
        allowed=assert_trainables(model,targets)
        saved=load_file(str(DONOR/"adapter_model.safetensors"),device="cpu")
        installed=get_peft_model_state_dict(model)
        assert len(saved)==len(installed)==288 and set(saved)==set(installed)
        assert all(v.dtype==torch.float32 and torch.equal(v,installed[k].detach().cpu()) for k,v in saved.items())
        assert sum(v.numel() for v in saved.values())==11796480
        del saved,installed
        frozen=parameter_digest(model,False); initial=parameter_digest(model,True)
        model.config.use_cache=False;model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=t["weight_decay"])
        scheduler=get_scheduler(t["lr_scheduler_type"],optimizer,num_warmup_steps=math.ceil(steps*t["warmup_ratio"]),num_training_steps=steps)
        model,optimizer=accel.prepare(model,optimizer)
        raw=accel.unwrap_model(model); ck_root=out/"checkpoints"; ck=helper.select_checkpoint(ck_root,identity)
        step=0;seen=[];prior=0.0
        if ck:
            cur=read(ck/"cursor.json");step=cur["step"];seen=cur["seen_ids"];prior=cur["training_seconds"]
            assert seen==sequence[:min(step*acc,len(rows))]
            restored=set_peft_model_state_dict(raw,load_file(str(ck/"adapter_model.safetensors")))
            assert not restored.unexpected_keys and parameter_digest(raw,True)==cur["adapter_digest"]
            optimizer.load_state_dict(torch.load(ck/"optimizer.pt",map_location="cpu",weights_only=True))
            scheduler.load_state_dict(torch.load(ck/"scheduler.pt",weights_only=True));assert scheduler.last_epoch==step
            helper.restore_rng(torch.load(ck/"rng.pt",map_location="cpu",weights_only=False))
        tic=time.monotonic(); elapsed=prior
        while step<steps:
            ix=schedule[step*acc:(step+1)*acc];optimizer.zero_grad(set_to_none=True)
            for i in ix:
                batch,rep=collator.encode(rows[i])
                pos=batch["labels"][0].ne(-100).nonzero().flatten()
                assert len(pos)==rep["supervised_tokens"] and pos[0].item()>0
                labels=batch.pop("labels")[0,pos].to(accel.device)
                batch={k:v.to(accel.device) for k,v in batch.items()}
                with accel.autocast():
                    logits=model(**batch,logits_to_keep=(pos-1).to(accel.device)).logits[0]
                    loss=torch.nn.functional.cross_entropy(logits.float(),labels)
                if not bool(torch.isfinite(loss)):raise ValueError(f"Nonfinite loss: {rows[i]['sample_id']}")
                accel.backward(loss/len(ix));seen.append(rows[i]["sample_id"])
            if step==0:write_json(out/"gradient_audit.json",validate_gradient_ownership(raw))
            accel.clip_grad_norm_(model.parameters(),t["max_grad_norm"])
            optimizer.step();scheduler.step();step+=1;elapsed=prior+time.monotonic()-tic
            progress=dict(step=step,steps=steps,seen=len(seen),training_seconds=elapsed,eta_seconds=elapsed/step*(steps-step))
            write_json(out/"progress.json",progress);print(progress,flush=True)
            if step%12==0 or step==steps:
                cur=dict(identity=identity,step=step,seen_ids=seen,training_seconds=elapsed,adapter_digest=parameter_digest(raw,True))
                helper.checkpoint(ck_root,raw,optimizer,scheduler,cur)
        final_digest=parameter_digest(raw,True)
        assert seen==sequence and parameter_digest(raw,False)==frozen and final_digest!=initial
        helper.save_final(out,helper.select_checkpoint(ck_root,identity))
        write_json(out/"seen_ids.json",seen)
        man=dict(complete=True,identity=identity,epoch=1,actual_steps=steps,seen=len(seen),frozen_verified=True,
                 trainable=sum(r["numel"] for r in allowed),initial_adapter=initial,final_adapter=final_digest,
                 adapter_file_sha256=sha(out/"adapter/adapter_model.safetensors"),sequence_sha256=identity["sequence"],
                 training_seconds=elapsed,environment=env)
        write_json(out/"manifest.json",man)
        return man


def export_arm(arm,shared):
    import torch
    from safetensors.torch import load_file,save_file
    from era_lora.common import write_json
    src=OUTPUT/"shared/adapter"; out=OUTPUT/arm; dest=out/"adapter"
    source_weights=src/"adapter_model.safetensors"
    assert sha(source_weights)==shared["adapter_file_sha256"]
    state=load_file(str(source_weights),device="cpu")
    assert all(v.dtype==torch.float32 for v in state.values())
    scale=SCALE[arm]
    if not dest.exists():
        tmp=out/(".adapter."+uuid.uuid4().hex+".partial");tmp.mkdir()
        shutil.copyfile(src/"adapter_config.json",tmp/"adapter_config.json")
        if scale==1:shutil.copyfile(source_weights,tmp/"adapter_model.safetensors")
        else:save_file(project(state,scale),str(tmp/"adapter_model.safetensors"),metadata={"format":"pt"})
        tmp.rename(dest)
    assert sha(dest/"adapter_config.json")==sha(src/"adapter_config.json")
    file=dest/"adapter_model.safetensors";proof=validate_projection(state,load_file(str(file),device="cpu"),scale)
    assert proof["pairs"]==144 and (scale!=1 or sha(file)==shared["adapter_file_sha256"])
    man=dict(complete=True,arm=arm,identity=dict(shared=sha(OUTPUT/"shared/manifest.json"),parent=BASE,delta_scale=scale),
             source_adapter_sha256=shared["adapter_file_sha256"],adapter_file_sha256=sha(file),projection=proof,
             epoch=shared["epoch"],actual_steps=shared["actual_steps"],sequence_sha256=shared["sequence_sha256"],
             training_performed_once_for_both=True)
    if (out/"manifest.json").exists():assert read(out/"manifest.json")==man
    else:write_json(out/"manifest.json",man)
    return man


def merge_arm(out,man,base,config):
    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    from era_lora.common import write_json
    from era_lora.merge import copy_processor_files
    dest=out/"merged"
    if not dest.is_dir():
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(base),dtype=torch.bfloat16,
            attn_implementation=config["training"]["attn_implementation"],local_files_only=True,low_cpu_mem_usage=True,use_safetensors=True)
        adapted=PeftModel.from_pretrained(model,str(out/"adapter"),is_trainable=False)
        merged=adapted.eval().merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
        assert not any("lora_" in k for k,_ in merged.named_parameters())
        merged.config.use_cache=True
        tmp=out/(".merged."+uuid.uuid4().hex+".partial");tmp.mkdir()
        merged.save_pretrained(tmp,safe_serialization=True,max_shard_size="4GB");copy_processor_files(base,tmp)
        weights={p.name:sha(p) for p in tmp.glob("*.safetensors")};assert weights
        write_json(tmp/"artifact_manifest.json",dict(weights=weights,parent=BASE,adapter=man["adapter_file_sha256"],
                   delta_scale=man["projection"]["delta_scale"],shared_adapter=man["source_adapter_sha256"]))
        tmp.rename(dest)
    rec=read(dest/"artifact_manifest.json")
    assert rec["adapter"]==man["adapter_file_sha256"] and rec["parent"]==BASE and rec["delta_scale"]==man["projection"]["delta_scale"]
    assert rec["shared_adapter"]==man["source_adapter_sha256"] and rec["weights"]=={p.name:sha(p) for p in dest.glob("*.safetensors")}
    write_json(out/"merge.json",rec)


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    sys.path.insert(0,str(PROJECT))
    from era_lora.train import file_lock
    cfg,base=common.baseline(); donor=verify_donor()
    out=OUTPUT/args.arm;out.mkdir(exist_ok=True,parents=True)
    if not args.merge: shared=train_shared(cfg,base,donor)
    else:
        shared=read(OUTPUT/"shared/manifest.json")
        assert shared["complete"] and shared["identity"]["donor"]==donor and shared["identity"]["parent"]==BASE
        assert shared["identity"]["training_sources"][str(Path(__file__))]==sha(Path(__file__))
    with file_lock(out/".writer.lock"):
        man=export_arm(args.arm,shared)
        if args.merge:merge_arm(out,man,base,cfg)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--round",type=int,required=True,choices=[2])
    parser.add_argument("--arm",choices=SCALE,required=True)
    parser.add_argument("--merge",action="store_true")
    run(parser.parse_args())
