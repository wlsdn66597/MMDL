"""Matched visual LoRA SFT with prefix-masked loss and optimizer-boundary recovery."""
import argparse
import math
import os
import random
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
PROJECT = Path.home() / "era_decoder_lora_20261030"
START = Path.home() / "era_link_repair_pilot/models/question/merged"
CFG = PROJECT / "configs/era_4epoch_20261030.json"
sys.path.insert(0, str(PROJECT))
from era_lora.common import canonical_json_hash, sha256_file, write_json
from era_lora.data import AssistantOnlyCollator, load_processor
from era_lora.train import (assert_trainables, discover_decoder_targets, file_lock, parameter_digest,
                            runtime_check, validate_gradient_ownership)

ARM_KEY = {"A": "control", "B": "self"}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def input_records(arm, smoke):
    if smoke:
        base = Path.home() / "era_second_pass_v1"
        prefix = "paired"
    else:
        base = ROOT
        prefix = ARM_KEY[arm]
    source = base / "train.jsonl"
    source_hash = sha256_file(source)
    assert read(base / "freeze.json")["train.jsonl"] == source_hash
    rows = [json.loads(line) for line in source.read_text(encoding="utf8").splitlines()]
    if smoke:
        # Include both old contrast examples (nonempty draft masks) and normal full targets.
        pairs = [x for x in rows if x["sample_id"].endswith(("_keep", "_replace"))][:6]
        anchors = [x for x in rows if x not in pairs and not x["sample_id"].endswith(("_keep", "_replace"))][:3]
        rows = pairs + anchors
    assert len(rows) == len({r["sample_id"] for r in rows}) == (9 if smoke else 1000)
    result = []
    for r in rows:
        image = (base/r["image"]).resolve()
        assert image.is_relative_to(base.resolve()) and sha256_file(image) == r["image_sha256"], r["sample_id"]
        target = r[prefix + "_target"]
        if smoke:
            match = re.search(r"(?= Is any excluded bar)",target)
            mask = target[:match.start()] if match else ""
        else:
            mask = r[prefix + "_mask_prefix"]
        assert isinstance(mask,str) and target.startswith(mask) and len(mask) < len(target), r["sample_id"]
        result.append(dict(sample_id=r["sample_id"], input_prompt=r[prefix+"_prompt"], image_path=str(image),
                           image_sha256=r["image_sha256"], response=target, mask_prefix=mask, arm=arm))
    assert any(r["mask_prefix"] for r in result) and any(not r["mask_prefix"] for r in result)
    return result, source_hash


class PrefixCollator(AssistantOnlyCollator):
    def encode(self, feature):
        batch, report = super().encode(feature)
        prefix = feature["mask_prefix"]
        report.update(masked_prefix_tokens=0, boundary_overlap_chars=0)
        if prefix:
            text = feature["response"]
            assert text.startswith(prefix)
            encoded = self.processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            first = report["input_tokens"]
            full_ids = batch["input_ids"][0, first:first+len(encoded.input_ids)].tolist()
            assert full_ids == encoded.input_ids, (feature["sample_id"], "assistant tokenization differs")
            assert batch["labels"][0, first+len(full_ids)].item() == report["assistant_end_token_id"]
            # If a BPE token spans the boundary, mask that whole token, including its suffix.
            count = sum(a < len(prefix) for a,b in encoded.offset_mapping)
            assert 0 < count < len(full_ids)
            assert encoded.offset_mapping[count-1][1] >= len(prefix)
            assert encoded.offset_mapping[count][0] >= len(prefix)
            batch["labels"][0, first:first+count] = -100
            report.update(masked_prefix_tokens=count, supervised_tokens=report["supervised_tokens"]-count,
                          boundary_overlap_chars=encoded.offset_mapping[count-1][1]-len(prefix))
        return batch, report


def verified_weights():
    weights = {p.name:sha256_file(p) for p in START.glob("*.safetensors")}
    assert len(weights)==3 and weights == read(START.parent/"merge.json")["weights"]
    prior = Path.home()/"era_second_pass_v1/models/paired/manifest.json"
    assert weights == read(prior)["identity"]["start_weights"]
    return weights


def file_hashes(folder):
    return {f.relative_to(folder).as_posix():sha256_file(f) for f in sorted(folder.rglob("*")) if f.is_file()}


def select_checkpoint(folder, identity):
    if not folder.is_dir():
        return None
    choices = [(int(m[1]),f) for f in folder.iterdir() if f.is_dir() and (m:=re.fullmatch(r"step-(\d+)",f.name))]
    if not choices:
        return None
    step, latest = max(choices)
    receipt = read(latest/"integrity.json")
    listed = file_hashes(latest)
    del listed["integrity.json"]
    assert receipt == listed, (latest, "checkpoint failed checksum")
    cursor = read(latest/"cursor.json")
    assert cursor["identity"]==identity and cursor["step"]==step, (latest, "wrong training identity")
    assert {"adapter_model.safetensors","adapter_config.json","optimizer.pt","scheduler.pt","rng.pt","cursor.json"} <= set(listed)
    return latest


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint(folder, model, optimizer, scheduler, cursor):
    import torch
    step = cursor["step"]
    folder.mkdir(parents=True,exist_ok=True)
    tmp = folder/f".step-{step}.{uuid.uuid4().hex}.partial"
    tmp.mkdir()
    # Capture stochastic state at the optimizer boundary before any serialization.
    state = rng_state()
    model.save_pretrained(tmp, safe_serialization=True, save_embedding_layers=False)
    torch.save(optimizer.state_dict(), tmp/"optimizer.pt")
    torch.save(scheduler.state_dict(), tmp/"scheduler.pt")
    torch.save(state, tmp/"rng.pt")
    write_json(tmp/"cursor.json",cursor)
    write_json(tmp/"integrity.json",file_hashes(tmp))
    tmp.rename(folder/f"step-{step}")
    restore_rng(state)


def sample_schedule(n, epochs, seed):
    import torch
    schedule=[]
    for epoch in range(epochs):
        generator=torch.Generator().manual_seed(seed+epoch)
        schedule.append(torch.randperm(n,generator=generator).tolist())
    return schedule


def save_final(save, latest):
    files=("adapter_config.json","adapter_model.safetensors")
    out=save/"adapter"
    if not out.exists():
        tmp=save/(".adapter."+uuid.uuid4().hex+".partial")
        tmp.mkdir()
        for name in files:
            shutil.copyfile(latest/name,tmp/name)
        tmp.rename(out)
    assert all(sha256_file(out/name)==sha256_file(latest/name) for name in files)


def merge_model(save, weights):
    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    from era_lora.merge import copy_processor_files
    man=read(save/"manifest.json")
    assert man["complete"] and man["identity"]["start_weights"]==weights
    assert sha256_file(save/"adapter/adapter_model.safetensors")==man["adapter_file_sha256"]
    dest=save/"merged"
    if not dest.is_dir():
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(START),dtype=torch.bfloat16,
            attn_implementation="sdpa",local_files_only=True,low_cpu_mem_usage=True,use_safetensors=True)
        model=PeftModel.from_pretrained(model,str(save/"adapter"),is_trainable=False)
        merged=model.eval().merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
        assert not any("lora_" in name for name,_ in merged.named_parameters())
        merged.config.use_cache=True
        tmp=save/(".merged."+uuid.uuid4().hex+".partial")
        tmp.mkdir()
        merged.save_pretrained(tmp,safe_serialization=True,max_shard_size="4GB")
        copy_processor_files(START,tmp)
        hashes={p.name:sha256_file(p) for p in sorted(tmp.glob("*.safetensors"))}
        assert hashes
        write_json(tmp/"artifact_manifest.json",dict(weights=hashes,parent=weights,adapter=man["adapter_file_sha256"]))
        tmp.rename(dest)
    meta=read(dest/"artifact_manifest.json")
    assert meta["parent"]==weights and meta["adapter"]==man["adapter_file_sha256"]
    assert meta["weights"]=={p.name:sha256_file(p) for p in dest.glob("*.safetensors")}
    write_json(save/"merge.json",meta)


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen3VLForConditionalGeneration, get_scheduler, set_seed

    smoke=args.smoke is not None
    root=ROOT/("recovery" if smoke else "models")
    save=root/(args.smoke if smoke else args.arm)
    save.mkdir(parents=True,exist_ok=True)
    with file_lock(save/".writer.lock"):
        weights=verified_weights()
        if args.merge:
            assert not smoke
            merge_model(save,weights)
            return
        env=runtime_check()
        if not smoke:
            verified=read(ROOT/"recovery/verified.json")
            assert verified["complete"] and verified["train_sha256"]==sha256_file(Path(__file__)) and verified["parent"]==weights
        rows,source_hash=input_records(args.arm,smoke)
        config=read(CFG)
        config["model"]["local_path"]=str(START)
        config["_config_dir"]=str(PROJECT)
        settings=config["training"]
        lora=config["lora"]
        epochs=2 if smoke else 3
        acc_steps=4 if smoke else settings["gradient_accumulation_steps"]
        interval=2 if smoke else 20
        steps_per_epoch=math.ceil(len(rows)/acc_steps)
        steps=epochs*steps_per_epoch
        seed=settings["seed"]
        order=sample_schedule(len(rows),epochs,seed)
        sequence=[rows[i]["sample_id"] for indices in order for i in indices]
        identity=dict(start=str(START),start_weights=weights,arm=args.arm,source_sha256=source_hash,
            config_sha256=sha256_file(CFG),parent_training_modules={n:sha256_file(PROJECT/"era_lora"/n) for n in ("data.py","train.py","common.py")},
            wrapper_sha256=sha256_file(Path(__file__)),environment=env["versions"],seed=seed,epochs=epochs,
            accumulation=acc_steps,expected_steps=steps,sequence_sha256=canonical_json_hash(sequence),smoke=smoke)
        identity_path=save/"identity.json"
        if identity_path.exists():
            assert read(identity_path)==identity, "Cannot resume under changed dataset, code, or training settings"
        else:
            write_json(identity_path,identity)
        if (save/"manifest.json").is_file():
            old=read(save/"manifest.json")
            assert old["identity"]==identity and old["complete"]
            assert sha256_file(save/"adapter/adapter_model.safetensors")==old["adapter_file_sha256"]
            return

        processor=load_processor(config)
        collator=PrefixCollator(processor,config)
        audit_path=save/"preflight.json"
        if not audit_path.exists():
            print("Verifying every input and loss mask",flush=True)
            reports=[collator.encode(row)[1] for row in rows]
            totals={k:sum(r[k] for r in reports) for k in ("input_tokens","sequence_tokens","supervised_tokens","masked_prefix_tokens")}
            assert totals["masked_prefix_tokens"]>0
            write_json(audit_path,dict(identity=identity,totals=totals,records=reports))
        assert read(audit_path)["identity"]==identity
        # Keep RNG state reproducible across process restarts and use a local epoch permutation.
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        # In torch 2.8 SDPA's flash backward selects deterministic reductions
        # only in strict mode; warn_only=True keeps the nondeterministic kernel.
        torch.use_deterministic_algorithms(True,warn_only=False)
        set_seed(seed)
        accel=Accelerator(mixed_precision="bf16")
        assert accel.num_processes==1 and accel.scaler is None
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(START),dtype=torch.bfloat16,
            attn_implementation=settings["attn_implementation"],local_files_only=True,low_cpu_mem_usage=True,use_safetensors=True)
        model.requires_grad_(False)
        targets=discover_decoder_targets(model)
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,target_modules=targets,
            r=lora["r"],lora_alpha=lora["alpha"],lora_dropout=lora["dropout"],bias="none",init_lora_weights=True))
        trainables=assert_trainables(model,targets)
        frozen,initial=parameter_digest(model,False),parameter_digest(model,True)
        matched=dict(start_weights=weights,initial_adapter=initial,frozen_parameters=frozen,sequence_sha256=identity["sequence_sha256"])
        with file_lock(root/".init.lock"):
            if (root/"matched_init.json").is_file():
                assert read(root/"matched_init.json")==matched
            else:
                write_json(root/"matched_init.json",matched)
        model.config.use_cache=False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings["learning_rate"],
                                   weight_decay=settings["weight_decay"],betas=(.9,.999),eps=1e-8)
        scheduler=get_scheduler(settings["lr_scheduler_type"],optimizer,num_warmup_steps=math.ceil(steps*settings["warmup_ratio"]),num_training_steps=steps)
        model,optimizer=accel.prepare(model,optimizer)
        raw_model=accel.unwrap_model(model)
        folder=save/"checkpoints"
        latest=select_checkpoint(folder,identity)
        step=0
        seen=[]
        prior_seconds=0.0
        if latest:
            cursor=read(latest/"cursor.json")
            step=cursor["step"]
            seen=cursor["seen_ids"]
            assert seen==sequence[:len(seen)]
            expected_exposures=(step//steps_per_epoch)*len(rows)+min((step%steps_per_epoch)*acc_steps,len(rows))
            assert len(seen)==expected_exposures
            state=set_peft_model_state_dict(raw_model,load_file(str(latest/"adapter_model.safetensors")))
            assert not state.unexpected_keys
            assert parameter_digest(raw_model,True)==cursor["adapter_digest"]
            optimizer.load_state_dict(torch.load(latest/"optimizer.pt",map_location="cpu",weights_only=True))
            scheduler.load_state_dict(torch.load(latest/"scheduler.pt",map_location="cpu",weights_only=True))
            assert scheduler.last_epoch==step
            restore_rng(torch.load(latest/"rng.pt",map_location="cpu",weights_only=False))
            prior_seconds=cursor["training_seconds"]
            print(f"Resuming after committed update {step}/{steps}, {len(seen)} examples",flush=True)
        tick=time.monotonic()
        while step<steps:
            epoch,group=divmod(step,steps_per_epoch)
            ids=order[epoch][group*acc_steps:(group+1)*acc_steps]
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for ix in ids:
                batch,_=collator.encode(rows[ix])
                batch={k:v.to(accel.device) for k,v in batch.items()}
                with accel.autocast():
                    loss=model(**batch).loss
                if not bool(torch.isfinite(loss.detach())):
                    raise RuntimeError(f"Nonfinite loss at {rows[ix]['sample_id']}")
                accel.backward(loss/len(ids))
                losses.append(float(loss.detach()))
                seen.append(rows[ix]["sample_id"])
            if step==0:
                write_json(save/"gradient_audit.json",validate_gradient_ownership(raw_model))
            grad=accel.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],settings["max_grad_norm"])
            if not bool(torch.isfinite(grad)):
                raise RuntimeError(f"Nonfinite gradients at step {step}")
            optimizer.step()
            scheduler.step()
            step+=1
            elapsed=prior_seconds+time.monotonic()-tick
            progress=dict(step=step,steps=steps,seen=len(seen),loss=sum(losses)/len(losses),training_seconds=elapsed,
                          eta_seconds=elapsed/step*(steps-step),time_unix=time.time())
            write_json(save/"progress.json",progress)
            print(json.dumps(progress),flush=True)
            if step%interval==0 or step==steps:
                cursor=dict(identity=identity,step=step,seen_ids=seen,training_seconds=elapsed,
                            adapter_digest=parameter_digest(raw_model,True))
                checkpoint(folder,raw_model,optimizer,scheduler,cursor)
                if smoke and args.interrupt and step==interval:
                    write_json(save/"smoke_exit.json",dict(step=step,identity=identity,expected_exit=86))
                    print("Recovery test exits after first durable checkpoint",flush=True)
                    os._exit(86)
        assert seen==sequence and len(sequence)==len(rows)*epochs
        final=parameter_digest(raw_model,True)
        assert final!=initial and parameter_digest(raw_model,False)==frozen
        latest=select_checkpoint(folder,identity)
        assert read(latest/"cursor.json")["step"]==steps
        save_final(save,latest)
        write_json(save/"seen_ids.json",seen)
        write_json(save/"manifest.json",dict(complete=True,arm=args.arm,identity=identity,environment=env,
            actual_steps=steps,epoch=epochs,trainable=sum(r["numel"] for r in trainables),frozen_verified=True,
            initial_adapter=initial,final_adapter=final,adapter_file_sha256=sha256_file(save/"adapter/adapter_model.safetensors"),
            sequence_sha256=canonical_json_hash(seen),training_seconds=read(latest/"cursor.json")["training_seconds"],
            preflight_sha256=sha256_file(audit_path),mask_summary=read(audit_path)["totals"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("arm",choices=ARM_KEY)
    parser.add_argument("--merge",action="store_true")
    parser.add_argument("--smoke",choices=["reference","interrupted"])
    parser.add_argument("--interrupt",action="store_true")
    args=parser.parse_args()
    if args.interrupt and args.smoke!="interrupted":
        parser.error("The intentional stop is available only in the interrupted recovery smoke.")
    run(args)
