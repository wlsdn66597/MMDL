"""Plain-arm epoch exports using the same ERA export arithmetic and provenance checks.
Original ERA helpers and training sources are not modified.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from era_lora.common import load_config,resolve_path,write_json,sha256_file,config_digest
from era_lora.train import output_directory,file_manifest,resolve_model_snapshot,discover_decoder_targets,utc_now
from era_lora import merge
from era_lora.evaluate import validate_source,checkpoint_preflight,training_provenance
from era_lora.data import load_paired_records


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def experiment(cfg):return resolve_path(cfg,cfg['training']['output_root']).parent


def checkpoint_info(cfg,epoch):
    run=output_directory(cfg,'plain')
    trained=read(run/'run_manifest.json')
    assert trained['status']=='complete' and trained['frozen_weights_unchanged']
    assert trained['identity']['config_sha256']==config_digest(cfg)
    expected=math.ceil(cfg['dataset']['expected_items']/cfg['training']['gradient_accumulation_steps'])*epoch
    cp=run/f'checkpoint-{expected}'
    state=read(cp/'trainer_state.json')
    assert state['global_step']==expected and abs(state['epoch']-epoch)<1e-8
    assert read(cp/'run_identity.json')==trained['identity']
    assert file_manifest(cp,exclude=('checkpoint_integrity.json',))==read(cp/'checkpoint_integrity.json')
    return run,cp,trained,expected


def model_path(cfg,epoch):
    return (output_directory(cfg,'plain')/'merged' if epoch==cfg['training']['epochs']
            else experiment(cfg)/'plain_epoch_checkpoints'/f'epoch_{epoch}'/'merged')


def validate_export(cfg,epoch):
    run,cp,trained,step=checkpoint_info(cfg,epoch)
    model=model_path(cfg,epoch)
    hashes=checkpoint_preflight(model)
    if epoch==cfg['training']['epochs']:
        provenance=training_provenance(cfg,'plain',model)
        assert sha256_file(cp/'adapter_model.safetensors')==sha256_file(run/'adapter/adapter_model.safetensors')
    else:
        proof=read(model/'artifact_manifest.json')
        assert proof['training_identity']==trained['identity'] and proof['epoch']==epoch and proof['optimizer_step']==step
        assert proof['checkpoint_integrity_sha256']==sha256_file(cp/'checkpoint_integrity.json')
        assert proof['adapter_weights_sha256']==sha256_file(cp/'adapter_model.safetensors')
        assert proof['structure_validation']['all_non_target_tensors_identical']
        assert proof['serialized_validation']['all_weights_bfloat16']
        assert proof['artifact_files_sha256']==file_manifest(model,exclude=('artifact_manifest.json',))
        provenance=proof
    return {'epoch':epoch,'optimizer_step':step,'checkpoint':str(cp),
        'model':str(model),'hashes':hashes,'provenance':provenance}


def export(cfg,epoch):
    if model_path(cfg,epoch).exists():
        validate_export(cfg,epoch);print(f'Validated existing epoch {epoch} export',flush=True);return
    if epoch==cfg['training']['epochs']:
        merge.execute_merge(cfg,'plain');validate_export(cfg,epoch);return
    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    run,cp,trained,step=checkpoint_info(cfg,epoch)
    snapshot=resolve_model_snapshot(cfg)
    assert {p.name:sha256_file(p) for p in sorted(snapshot.iterdir()) if p.is_file()}==trained['base_files_sha256']
    base=Qwen3VLForConditionalGeneration.from_pretrained(str(snapshot),dtype=torch.bfloat16,
        attn_implementation=cfg['training']['attn_implementation'],local_files_only=True,use_safetensors=True)
    targets=discover_decoder_targets(base)
    ac=read(cp/'adapter_config.json');saved=ac['target_modules']
    assert isinstance(saved,list)
    resolved={n for n,_ in base.named_modules() if any(n==t or n.endswith('.'+t) for t in saved)}
    assert resolved==set(targets)
    assert (ac['r'],ac['lora_alpha'],ac['lora_dropout'],ac['bias'],ac.get('modules_to_save'))==(cfg['lora']['r'],cfg['lora']['alpha'],cfg['lora']['dropout'],'none',None)
    before=merge.tensor_hashes(base)
    adapter=PeftModel.from_pretrained(base,str(cp),is_trainable=False).eval()
    merged=adapter.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
    merged.config.use_cache=True
    assert not any('lora_' in n for n,_ in merged.named_modules())
    validation=merge.validate_merged_state(before,merge.tensor_hashes(merged),targets)
    destination=model_path(cfg,epoch);destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.with_name('merged.incomplete-'+uuid.uuid4().hex[:12]);staging.mkdir()
    merged.save_pretrained(staging,safe_serialization=True,max_shard_size='4GB')
    processor_hashes=merge.copy_processor_files(snapshot,staging)
    serialized=merge.validate_saved_weights(staging,set(before),merge.tied_state_groups(merged))
    proof={'kind':'intermediate_epoch_export','created_at':utc_now(),'epoch':epoch,'optimizer_step':step,
        'training_identity':trained['identity'],'training_manifest_sha256':sha256_file(run/'run_manifest.json'),
        'checkpoint':str(cp),'checkpoint_integrity_sha256':sha256_file(cp/'checkpoint_integrity.json'),
        'adapter_weights_sha256':sha256_file(cp/'adapter_model.safetensors'),
        'structure_validation':validation,'serialized_validation':serialized,
        'processor_files_sha256':processor_hashes,'artifact_files_sha256':file_manifest(staging),
        'script_sha256':sha256_file(Path(__file__))}
    write_json(staging/'artifact_manifest.json',proof);staging.rename(destination)
    validate_export(cfg,epoch);print(f'Exported epoch {epoch}, step {step}',flush=True)



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--epoch',type=int,choices=(2,3,4),required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(); cfg=load_config(a.config)
    assert cfg['training']['epochs']==4
    if a.execute: export(cfg,a.epoch)
    else: print(json.dumps({'action':'export_plain','epoch':a.epoch,'execute':False}))
