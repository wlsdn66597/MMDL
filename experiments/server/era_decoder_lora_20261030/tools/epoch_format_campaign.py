"""Epoch-checkpoint exports and fixed-subset diagnostics; execute is explicit.

Intermediate exports retain their actual checkpoint lineage. They never pretend
to be final adapters, and are separate from the full benchmark comparison gate.
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
from train_format_diagnostic import selection,generate,structure


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def experiment(cfg):return resolve_path(cfg,cfg['training']['output_root']).parent


def checkpoint_info(cfg,epoch):
    run=output_directory(cfg,'era')
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
    return (output_directory(cfg,'era')/'merged' if epoch==cfg['training']['epochs']
            else experiment(cfg)/'epoch_checkpoints'/f'epoch_{epoch}'/'merged')


def validate_export(cfg,epoch):
    run,cp,trained,step=checkpoint_info(cfg,epoch)
    model=model_path(cfg,epoch)
    hashes=checkpoint_preflight(model)
    if epoch==cfg['training']['epochs']:
        provenance=training_provenance(cfg,'era',model)
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
        merge.execute_merge(cfg,'era');validate_export(cfg,epoch);return
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


def evaluate(cfg,epoch,kind):
    proof=validate_export(cfg,epoch)
    source,protocol,configs,hashes=validate_source(cfg);c=configs[3407]
    output=experiment(cfg)/'diagnostics'/f'epoch_{epoch}'/kind
    report_file=output/('generation_summary.json' if kind=='train' else 'report.json')
    if report_file.exists():
        previous=read(output/'manifest.json')
        assert previous['export']==proof
        if kind=='train':
            assert all(read(report_file)[c]['items']==20 for c in ('train_resolution','eval_resolution'))
        else:assert read(report_file)['items']==8
        print(f'Reusing completed {kind} epoch {epoch}',flush=True);return
    if output.exists():raise RuntimeError(f'Incomplete diagnostic retained for inspection: {output}')
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ.pop('VLLM_BATCH_INVARIANT',None)
    records,_=load_paired_records(cfg)
    chosen=selection(records)
    previous_ids=read(ROOT/'runs/era_first_20261030/train_format_diagnostic/generate/manifest.json')['generation_ids']
    assert chosen==previous_ids
    ids=read(ROOT/'configs/era_diagnostic_ids.json')
    assert ids==read(ROOT/'runs/era_first_20261030/diagnostic8/manifest.json')['ids']
    manifest={'export':proof,'training_ids':chosen,'mmmu_ids':ids,'sampling':c['sampling'],
        'max_new_tokens':c['max_new_tokens'],'baseline_source_hashes':hashes,
        'prompt_changes':False,'not_full_benchmark':True,'script_sha256':sha256_file(Path(__file__))}
    output.mkdir(parents=True);write_json(output/'manifest.json',manifest)
    if kind=='train':
        view=copy.deepcopy(cfg);view['evaluation']['checkpoints']['era']=proof['model']
        generate(view,records,chosen,output,c);return
    sys.path.insert(0,str(source/'src'))
    spec=importlib.util.spec_from_file_location('frozen_epoch_evaluator',source/'src/evaluate.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    original_load=evaluator.prepare.load_data
    def subset_load(*a,**kw):
        datasets,rows=original_load(*a,**kw);lookup={r['id']:r for r in rows}
        return datasets,[lookup[i] for i in ids]
    original_check=evaluator.check_records
    def subset_check(rows,config,partial=False):
        original_check(rows,config,partial=True)
        assert all(r['id'] in ids for r in rows)
    evaluator.prepare.load_data=subset_load;evaluator.check_records=subset_check
    paths=evaluator.prepare.cache_paths(c,cfg['evaluation']['model_cache'],cfg['evaluation']['dataset_cache'])
    evaluator.infer(SimpleNamespace(checkpoint=proof['model']),c,output,{'cache_paths':paths})
    generated=evaluator.prepare.read_jsonl(output/'predictions.jsonl')
    assert [r['id'] for r in generated]==ids
    previous={r['id']:r for r in evaluator.prepare.read_jsonl(ROOT/'runs/era_first_20261030/diagnostic8/predictions.jsonl')}
    rows=[]
    for r in generated:
        assert r['input_prompt']==previous[r['id']]['input_prompt'] and r['rendered_chat']==previous[r['id']]['rendered_chat']
        rows.append({'id':r['id'],**structure(r['model_response']['raw']),
            'response':r['model_response']['raw'],'finish_reason':r['model_response']['finish_reason'],
            'output_tokens':r['model_response']['output_tokens'],'grade':evaluator.parsers.grade_response(r)})
    report={'epoch':epoch,'optimizer_step':proof['optimizer_step'],'items':len(rows),
        'strict_era':sum(r['strict_era'] for r in rows),'flexible_era':sum(r['flexible_era'] for r in rows),
        'length_count':sum(r['finish_reason']=='length' for r in rows),
        'parser_correct':sum(r['grade']['hit']==1 for r in rows),
        'parser_unresolved':sum(r['grade']['status']=='pending_judge' for r in rows),'judge_called':False,'rows':rows}
    write_json(report_file,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


def summarize(cfg):
    result=[]
    for epoch in range(1,cfg['training']['epochs']+1):
        root=experiment(cfg)/'diagnostics'/f'epoch_{epoch}'
        t=root/'train/generation_summary.json';m=root/'mmmu/report.json'
        if t.exists() and m.exists():
            result.append({'epoch':epoch,'train':read(t),'mmmu':{k:v for k,v in read(m).items() if k!='rows'}})
    write_json(experiment(cfg)/'epoch_comparison.json',{'completed_epochs':len(result),'results':result})


def campaign(cfg):
    logroot=experiment(cfg)/'campaign_logs';logroot.mkdir(exist_ok=True)
    for epoch in range(1,cfg['training']['epochs']+1):
        for action in ('export','train','mmmu'):
            interpreter=str(ROOT/'.venv/bin/python') if action=='export' else cfg['evaluation']['python']
            command=[interpreter,'-u',str(Path(__file__).resolve()),'--config',cfg['_config_path'],
                '--action',action,'--epoch',str(epoch),'--execute']
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='' if action=='export' else '0')
            print(f'Start epoch={epoch} action={action}',flush=True)
            with (logroot/f'epoch_{epoch}_{action}.log').open('a') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(f'Done epoch={epoch} action={action}',flush=True)
        summarize(cfg)
    print('All four epoch diagnostics completed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--action',required=True,choices=['export','train','mmmu','campaign'])
    p.add_argument('--epoch',type=int,choices=range(1,5));p.add_argument('--execute',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    assert cfg['training']['epochs']==4
    if a.action!='campaign' and a.epoch is None:p.error('--epoch required')
    if not a.execute:
        print(json.dumps({'action':a.action,'epoch':a.epoch,'config':cfg['_config_path'],'execute':False}));return
    if a.action=='export':export(cfg,a.epoch)
    elif a.action=='campaign':campaign(cfg)
    else:evaluate(cfg,a.epoch,a.action)


if __name__=='__main__':main()
