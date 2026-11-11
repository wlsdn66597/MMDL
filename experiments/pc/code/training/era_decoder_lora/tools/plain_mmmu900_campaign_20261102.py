"""Train existing paired plain targets, then compare epochs 2/3/4 at seed3407.

No baseline/ERA retraining, no extra seeds, no automatic retry/resume. The frozen
training snapshot preserves exactly the ERA initialization and code identity.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from era_lora.common import load_config,config_digest,sha256_file,write_json,canonical_json_hash
from era_lora.train import utc_now,output_directory,file_manifest
from era_lora.data import load_paired_records
from era_lora.compare import load_run,compare_repeats,compare_records,validate_pair,repetition_metrics
from era_lora.evaluate import validate_source,VERSIONS
from mmmu900_epoch_campaign import metrics,format_metrics
from plain_epoch_export_20261102 import validate_export

CONFIG=ROOT/'configs/era_4epoch_20261030.json'
OUT=ROOT/'runs/plain_mmmu900_20261102'
OLD=ROOT/'runs/mmmu900_epochs_20261031'
TRAIN_SOURCE=ROOT/'runs/era_4epoch_20261030/training_source'
EPOCHS=(2,3,4)

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(name,data):write_json(OUT/name,data)
def status(stage,**kwargs):save('status.json',{'stage':stage,'updated_at':utc_now(),**kwargs})

def verify_pair_manifest(era,plain):
    assert era['status']==plain['status']=='complete'
    assert plain['identity']==dict(era['identity'],arm='plain'),'Training identity mismatch'
    for key in ('training','lora','initial_adapter_sha256','base_files_sha256','data_manifest','processor_provenance','loss_normalization'):
        assert plain[key]==era[key],f'Training drift: {key}'
    assert plain['actual_optimizer_steps']==era['actual_optimizer_steps']==52
    assert plain['frozen_weights_unchanged'] and era['frozen_weights_unchanged']
    assert plain['final_adapter_sha256']!=era['final_adapter_sha256'],'Identical final adapters'
    for key in ('versions','gpu','cuda'):
        assert plain['environment'][key]==era['environment'][key],f'Environment drift: {key}'

def learning_rate_history(state):
    return [(r['step'],r['learning_rate']) for r in state['log_history'] if 'learning_rate' in r]

def helper_hashes():
    files=[Path(__file__),ROOT/'tools/plain_epoch_export_20261102.py',ROOT/'tools/run_plain_mmmu900_20261102.sh',ROOT/'tools/mmmu900_epoch_campaign.py',ROOT/'tools/epoch_format_campaign.py']
    files+=list((ROOT/'era_lora').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in files}

def references(cfg,configs,ids):
    previous=read(OLD/'plan.json');verification=read(OLD/'pilot/verification.json')
    complete=read(OLD/'pilot_completed.json')
    assert complete['state']=='seed3407_inference_and_scoring_complete'
    assert complete['baseline_replay_and_all_three_repeats_passed']
    assert sha256_file(OLD/'plan.json')==verification['original_plan_sha256']
    assert sha256_file(OLD/'scope_change_20261102.json')==verification['scope_change_sha256']
    runs={'baseline':load_run(cfg['evaluation']['baseline_runs']['3407'],3407,configs[3407],ids,'baseline')}
    for e in EPOCHS:runs[f'era_epoch_{e}']=load_run(OLD/f'epoch_{e}_seed3407',3407,configs[3407],ids,'era')
    for name,run in runs.items():
        key='baseline' if name=='baseline' else name.removeprefix('era_')
        assert run['artifacts_sha256']==verification['artifacts'][key]
        proof=verification['repeatability'][key]
        assert proof['passed'] and proof['records_compared']==900 and not proof['differences']
        repeat=OLD/('baseline_seed3407_replay' if key=='baseline' else f'{key}_seed3407_repeat')
        for fn,h in proof['second_artifacts_sha256'].items():assert sha256_file(repeat/fn)==h
        validate_pair(runs['baseline']['records'],run['records'])
    assert runs['baseline']['artifacts_sha256']==previous['baseline_runs']['3407']['hashes']
    return runs

def preflight():
    cfg=load_config(CONFIG)
    era_dir=output_directory(cfg,'era');era=read(era_dir/'run_manifest.json')
    assert era['status']=='complete' and era['actual_optimizer_steps']==52
    assert config_digest(cfg)==era['identity']['config_sha256']
    files={p.name:sha256_file(p) for p in (TRAIN_SOURCE/'era_lora').glob('*.py')}
    assert files==era['identity']['source_sha256'],'Original training snapshot changed'
    rows,data=load_paired_records(cfg)
    assert len(rows)==200 and data==era['data_manifest']
    assert canonical_json_hash(data)==era['identity']['data_sha256']
    # Audited no-execute command imports the original package from its snapshot.
    cmd=[str(ROOT/'.venv/bin/python'),'-m','era_lora.train','--config',str(CONFIG),'--arm','plain']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(TRAIN_SOURCE))
    plan=json.loads(subprocess.check_output(cmd,cwd=TRAIN_SOURCE,env=env,text=True))
    assert plan['identity']==dict(era['identity'],arm='plain')
    source,protocol,configs,hashes=validate_source(cfg)
    ids=[r['id'] for r in read(protocol/'evaluation_items.json')]
    assert len(ids)==len(set(ids))==900
    ref=references(cfg,configs,ids)
    probe="import sys,json,importlib.metadata as m;print(json.dumps({'python':list(sys.version_info[:2]),'versions':{n:m.version(n).split('+')[0] for n in "+repr(list(VERSIONS))+"}}))"
    versions=json.loads(subprocess.check_output([cfg['evaluation']['python'],'-c',probe],text=True))
    assert versions=={'python':[3,12],'versions':VERSIONS}
    result={'kind':'paired_plain_era_single_seed_epoch_comparison','created_at':utc_now(),
      'config_path':str(CONFIG),'config_sha256':config_digest(cfg),'training_plan':plan,
      'original_training_snapshot':str(TRAIN_SOURCE),'training_snapshot_hashes':files,
      'evaluation_seeds':[3407],'epochs':list(EPOCHS),'items_per_run':900,
      'primary_responses':2700,'repeat_responses':2700,'reference_reinference':False,
      'reference_hashes':{n:r['artifacts_sha256'] for n,r in ref.items()},
      'reference_paths':{n:r['path'] for n,r in ref.items()},
      'reference_replay_receipt_sha256':sha256_file(OLD/'pilot/verification.json'),
      'source_hashes':hashes,'python_probe':versions,'helper_hashes':helper_hashes(),
      'semantic_sample':read(OLD/'plan.json')['semantic_sample'],
      'training_differences':['assistant target text only (existing plain versus ERA)'],
      'note':'Existing config retains the historical five-seed list for identity; this queue explicitly executes only seed3407. Training output is the previously absent plain sibling. All reference artifacts stay unchanged.'}
    if (OUT/'plan.json').exists():
        old=read(OUT/'plan.json');assert {k:v for k,v in old.items() if k!='created_at'}=={k:v for k,v in result.items() if k!='created_at'}
        result=old
    else:
        assert not output_directory(cfg,'plain').exists(),'Fresh plain training required'
        save('plan.json',result)
    return cfg,source,protocol,configs,ids,ref,result

def run_logged(command,log,cwd=ROOT,cpu=False):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='' if cpu else '0')
    env.pop('PYTHONPATH',None);env.pop('VLLM_BATCH_INVARIANT',None)
    if cwd==TRAIN_SOURCE:env['PYTHONPATH']=str(TRAIN_SOURCE)
    env.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    with (OUT/log).open('x') as handle:
        subprocess.run(command,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)

def summarize(ref,plain,plan):
    runs={**ref,**plain}
    summary={'seed':3407,'unique_questions':900,'metrics':{m:metrics(r['records']) for m,r in runs.items()},'paired':{},'repeats_excluded':True,'semantic_review':'pending'}
    for e in EPOCHS:
        for left,right in [('baseline',f'plain_epoch_{e}'),(f'plain_epoch_{e}',f'era_epoch_{e}')]:
            if right in runs and left in runs:summary['paired'][left+'_to_'+right]=compare_records(runs[left]['records'],runs[right]['records'])
    save('summary.json',summary)
    inventory=[]
    for i,r in enumerate(ref['baseline']['records']):
        item={'id':r['id'],'subject':r['subject'],'models':{}}
        for m,run in runs.items():
            record=run['records'][i];assert record['id']==r['id']
            item['models'][m]={'record':record,'format':format_metrics(record),'repetition':repetition_metrics(record['model_response']['raw'])}
        inventory.append(item)
    save('all900_response_inventory.json',inventory)
    selected=set(plan['semantic_sample']['ids'])
    save('prespecified30_responses.json',[r for r in inventory if r['id'] in selected])

def campaign():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'status.json').exists(),'No automatic restart/resume of an attempted campaign'
        cfg,source,protocol,configs,ids,ref,plan=preflight()
        assert helper_hashes()==plan['helper_hashes']
        (OUT/'logs').mkdir()
        status('training_plain_4epoch',started_at=utc_now())
        run_logged([str(ROOT/'.venv/bin/python'),'-u','-m','era_lora.train','--config',str(CONFIG),'--arm','plain','--execute'],'logs/train.log',cwd=TRAIN_SOURCE)
        era_dir=output_directory(cfg,'era');plain_dir=output_directory(cfg,'plain')
        era=read(era_dir/'run_manifest.json');trained=read(plain_dir/'run_manifest.json')
        verify_pair_manifest(era,trained)
        assert read(era_dir/'encoded_pair_preflight.json')==read(plain_dir/'encoded_pair_preflight.json')
        assert read(era_dir/'trainable_parameters.json')==read(plain_dir/'trainable_parameters.json')
        eras=read(era_dir/'trainer_state.json');plains=read(plain_dir/'trainer_state.json')
        assert learning_rate_history(eras)==learning_rate_history(plains) and len(learning_rate_history(plains))==52,'Learning rate schedule mismatch'
        save('training_pair_verification.json',{'passed':True,'identical_initial_adapter':trained['initial_adapter_sha256'],'identity_except_arm_identical':True,'optimizer_steps':52,'learning_rate_history_identical':True,'encoded_pairs_identical':True,'targets_identical':True,'frozen_base_unchanged':True,'plain_manifest_sha256':sha256_file(plain_dir/'run_manifest.json'),'era_manifest_sha256':sha256_file(era_dir/'run_manifest.json')})
        plain={};receipts={}
        for epoch in EPOCHS:
            status(f'export_plain_epoch_{epoch}')
            run_logged([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'tools/plain_epoch_export_20261102.py'),'--config',str(CONFIG),'--epoch',str(epoch),'--execute'],f'logs/export_epoch_{epoch}.log',cpu=True)
            proof=validate_export(cfg,epoch);save(f'export_epoch_{epoch}.json',proof)
            assert proof['optimizer_step']==13*epoch
            for repeat in (False,True):
                name=f'plain_epoch_{epoch}_seed3407'+('_repeat' if repeat else '')
                dest=OUT/name;assert not dest.exists();dest.mkdir()
                command=[cfg['evaluation']['python'],str(source/'src/evaluate.py'),'--config',str(protocol/'config.seed3407.json'),'--run-name',name,'--output-root',str(OUT),'--checkpoint',proof['model'],'--model-cache',cfg['evaluation']['model_cache'],'--dataset-cache',cfg['evaluation']['dataset_cache']]
                write_json(dest/'launcher.json',{'argv':command,'fresh_output_verified':True,'python_probe':plan['python_probe'],'checkpoint_sha256':proof['hashes'],'vllm_environment':{'VLLM_ENABLE_V1_MULTIPROCESSING':'0','VLLM_WORKER_MULTIPROC_METHOD':'spawn'},'source_hashes':plan['source_hashes'],'training_provenance':proof,'campaign_script_sha256':plan['helper_hashes'][str(Path(__file__).relative_to(ROOT))]})
                status(name,stage_started_at=utc_now())
                run_logged(command,f'logs/{name}.log')
                run=load_run(dest,3407,configs[3407],ids,'plain')
                validate_pair(ref['baseline']['records'],run['records'])
                for a,b in zip(ref['baseline']['records'],run['records']):
                    assert a['model_response']['input_tokens']==b['model_response']['input_tokens']
                write_json(dest/'format_metrics.json',metrics(run['records']))
                if not repeat:
                    plain[f'plain_epoch_{epoch}']=run
                    summarize(ref,plain,plan)
                else:
                    check=compare_repeats(OUT/f'plain_epoch_{epoch}_seed3407',dest)
                    save(f'epoch_{epoch}_repeatability.json',check)
                    assert check['passed'],f'Epoch {epoch} reproducibility failed'
                    receipts[str(epoch)]=check
        assert helper_hashes()==plan['helper_hashes'],'Running source changed'
        preserved=references(cfg,configs,ids)
        assert {m:r['artifacts_sha256'] for m,r in preserved.items()}==plan['reference_hashes']
        summarize(ref,plain,plan)
        save('completed.json',{'state':'training_and_seed3407_scoring_complete','completed_at':utc_now(),'primary_runs':3,'primary_responses':2700,'repeat_runs':3,'repeat_responses_excluded':2700,'all_repeats_passed':True,'old_baseline_era_preserved':True,'seeds':[3407],'semantic_review':'pending','artifacts':{m:r['artifacts_sha256'] for m,r in plain.items()}})
        status('complete',semantic_review='pending')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true');p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    try:
        if args.execute:campaign()
        elif args.preflight:
            *_,plan=preflight();print(json.dumps({'preflight_passed':True,'training_identity':plan['training_plan']['identity'],'primary_responses':2700,'repeat_responses':2700,'seeds':[3407]},indent=2))
        else:p.error('Choose --preflight or --execute')
    except Exception:
        if args.execute:save('failed.json',{'failed_at':utc_now(),'traceback':traceback.format_exc()})
        raise
