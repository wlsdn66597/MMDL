"""Explicitly authorized recovery after server suspend; never trains a model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import plain_mmmu900_campaign_20261102 as c
from era_lora.common import write_json,sha256_file
from era_lora.compare import load_run,compare_repeats,validate_pair,scrub_timing
from era_lora.train import output_directory,utc_now
from plain_epoch_export_20261102 import validate_export

OUT=c.OUT
REC=OUT/'recovery_20261104'
PARTIAL=OUT/'plain_epoch_3_seed3407_repeat'
PIDS=(68280,71263,71286)
COMPLETED=('plain_epoch_2_seed3407','plain_epoch_2_seed3407_repeat','plain_epoch_3_seed3407')

def proc(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    stat=(p/'stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'state':stat[0],'ppid':int(stat[1]),'start_ticks':stat[19],
      'argv':(p/'cmdline').read_bytes().decode().strip('\0').split('\0'),'cwd':str((p/'cwd').resolve())}

def signal_exact(saved,sig):
    now=proc(saved['pid'])
    if now is None or now['state']=='Z':return
    assert all(now[k]==saved[k] for k in ('pid','start_ticks','argv','cwd')),'Process identity changed'
    os.kill(saved['pid'],sig)

def preserved_hashes(cfg):
    paths=[]
    for name in COMPLETED:paths += [p for p in (OUT/name).iterdir() if p.is_file() and p.name!='.run.lock']
    for name in ('plan.json','training_pair_verification.json','epoch_2_repeatability.json','export_epoch_2.json','export_epoch_3.json'):paths.append(OUT/name)
    for arm in ('plain','era'):
        t=output_directory(cfg,arm)
        paths += [t/'run_manifest.json',t/'trainer_state.json',t/'trainable_parameters.json',t/'encoded_pair_preflight.json']
        paths += list((t/'adapter').glob('*'))
        paths += list(t.glob('checkpoint-*/checkpoint_integrity.json'))
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in paths if p.is_file()}

def verify_preserved(receipt):
    for name,h in receipt['preserved_files'].items():assert sha256_file(ROOT/name)==h,name

def verify_prefix(primary,repeat):
    assert 0<len(repeat)<900
    assert [r['id'] for r in repeat]==[r['id'] for r in primary[:len(repeat)]]
    for a,b in zip(primary,repeat):
        assert scrub_timing(a)==scrub_timing(b),a['id']

def prepare_stop():
    assert not REC.exists(),'A recovery record already exists; do not repeat stop actions'
    cfg,source,protocol,configs,ids,ref,plan=c.preflight()
    assert c.read(OUT/'status.json')['stage']=='plain_epoch_3_seed3407_repeat'
    assert not (OUT/'completed.json').exists() and not (OUT/'plain_epoch_4_seed3407').exists()
    for name in COMPLETED:load_run(OUT/name,3407,configs[3407],ids,'plain')
    original=load_run(OUT/'plain_epoch_3_seed3407',3407,configs[3407],ids,'plain')
    primary=[json.loads(l) for l in (OUT/'plain_epoch_3_seed3407/predictions.jsonl').read_text().splitlines() if l.strip()]
    prefix=[json.loads(l) for l in (PARTIAL/'predictions.jsonl').read_text().splitlines() if l.strip()]
    verify_prefix(primary,prefix)
    assert len(prefix)==423 and time.time()-(PARTIAL/'predictions.jsonl').stat().st_mtime>600
    export=validate_export(cfg,3)
    assert c.read(PARTIAL/'launcher.json')['checkpoint_sha256']==export['hashes']
    assert c.read(PARTIAL/'manifest.json')['fingerprint']==original['manifest']['fingerprint']
    states=[proc(pid) for pid in PIDS]
    assert all(s and s['cwd']==str(ROOT) and s['state'] not in ('T','t','Z') for s in states)
    assert states[1]['ppid']==PIDS[0] and states[2]['ppid']==PIDS[1]
    assert states[0]['argv'][1:]==['-u','tools/plain_mmmu900_campaign_20261102.py','--execute']
    for s in states[1:]:
        assert s['argv'][1]==str(source/'src/evaluate.py')
        assert s['argv'][s['argv'].index('--run-name')+1]=='plain_epoch_3_seed3407_repeat'
        assert s['argv'][s['argv'].index('--checkpoint')+1]==export['model']
    assert states[2]['argv'][-2:]==['--_stage','infer']
    receipt={'prepared_at':utc_now(),'reason':'Explicit user recovery request after system suspend 2026-11-03 16:42:43 to 2026-11-04 11:58:20 KST; no output after resume',
      'processes':states,'prefix_records':len(prefix),'preserved_files':preserved_hashes(cfg),
      'recovery_script_sha256':sha256_file(Path(__file__)),'original_helper_hashes':plan['helper_hashes'],
      'training_restarted':False,'completed_evaluations_restarted':False,'resume_strategy':'Frozen evaluator skips saved IDs, same per-request sampling seed3407, full900 exact repeat gate retained',
      'sleep_inhibitor':'systemd-inhibit denied by server account policy; no system power settings changed'}
    REC.mkdir()
    write_json(REC/'prepare_receipt.json',receipt)
    # Freeze the exact old chain before the snapshot; retire it without its queue advancing.
    for s in states:signal_exact(s,signal.SIGSTOP)
    for p in PARTIAL.iterdir():
        if p.is_file() and p.name!='.run.lock':
            target=REC/'interrupted_repeat'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    for n in ('status.json','campaign.log','summary.json','all900_response_inventory.json','prespecified30_responses.json'):
        p=OUT/n
        if p.exists():shutil.copy2(p,REC/('before_'+n))
    shutil.copy2(OUT/'logs/plain_epoch_3_seed3407_repeat.log',REC/'interrupted_repeat.log')
    receipt['snapshot_files']={str(p.relative_to(REC)):sha256_file(p) for p in REC.rglob('*') if p.is_file() and p.name!='prepare_receipt.json'}
    write_json(REC/'prepare_receipt.json',receipt)
    for s in reversed(states):
        signal_exact(s,signal.SIGTERM);signal_exact(s,signal.SIGCONT)
        until=time.monotonic()+5
        while time.monotonic()<until:
            now=proc(s['pid'])
            if now is None or now['state']=='Z':break
            time.sleep(.2)
        else:signal_exact(s,signal.SIGKILL)
    until=time.monotonic()+3
    while time.monotonic()<until:
        if all(proc(s['pid']) is None or proc(s['pid'])['state']=='Z' for s in states):break
        time.sleep(.1)
    assert all(proc(s['pid']) is None or proc(s['pid'])['state']=='Z' for s in states)
    verify_preserved(receipt)
    write_json(REC/'stopped.json',{'stopped_at':utc_now(),'old_processes_retired':True,'prefix_records_preserved':423,'original_completed_files_unchanged':True})
    print('Stopped exact stalled chain; preserved completed results and 423 response prefix',flush=True)

def state(stage,**extra):
    write_json(OUT/'recovery_status.json',{'stage':stage,'updated_at':utc_now(),**extra})

def execute():
    receipt=c.read(REC/'prepare_receipt.json');assert (REC/'stopped.json').exists()
    assert sha256_file(Path(__file__))==receipt['recovery_script_sha256']
    assert not (OUT/'recovery_status.json').exists(),'Explicit continuation only; no duplicate recovery'
    with (OUT/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_preserved(receipt)
        cfg,source,protocol,configs,ids,ref,plan=c.preflight()
        assert c.helper_hashes()==receipt['original_helper_hashes']
        for name,h in receipt['snapshot_files'].items():assert sha256_file(REC/name)==h
        state('cuda_recovery_check')
        gpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        check=subprocess.run([str(ROOT/'.venv/bin/python'),'-c','import torch; x=torch.ones(1,device="cuda"); y=x+1; torch.cuda.synchronize(); assert y.item()==2; print("CUDA recovered")'],env=gpu_env,timeout=60,capture_output=True,text=True,check=True)
        write_json(REC/'cuda_check.json',{'passed':True,'stdout':check.stdout,'checked_at':utc_now()})
        plain={f'plain_epoch_{e}':load_run(OUT/f'plain_epoch_{e}_seed3407',3407,configs[3407],ids,'plain') for e in (2,3)}
        state('plain_epoch_3_seed3407_repeat',stage_started_at=utc_now(),resumed_saved_records=423)
        launch=c.read(PARTIAL/'launcher.json')
        c.run_logged(launch['argv'],'recovery_20261104/resumed_epoch3_repeat.log')
        saved=(REC/'interrupted_repeat/predictions.jsonl').read_bytes()
        with (PARTIAL/'predictions.jsonl').open('rb') as f:assert f.read(len(saved))==saved,'Saved prefix was modified'
        check=compare_repeats(OUT/'plain_epoch_3_seed3407',PARTIAL)
        c.save('epoch_3_repeatability.json',check)
        assert check['passed'],'Resumed repeat failed; epoch4 blocked'
        write_json(REC/'resume_verification.json',{'passed':True,'preserved_prefix_records':423,'new_responses':477,'full900_repeat_passed':True,'saved_prefix_sha256':sha256_file(REC/'interrupted_repeat/predictions.jsonl')})
        state('export_plain_epoch_4')
        c.run_logged([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'tools/plain_epoch_export_20261102.py'),'--config',str(c.CONFIG),'--epoch','4','--execute'],'recovery_20261104/export_epoch4.log',cpu=True)
        proof=validate_export(cfg,4);c.save('export_epoch_4.json',proof)
        for repeat in (False,True):
            name='plain_epoch_4_seed3407'+('_repeat' if repeat else '')
            dest=OUT/name;assert not dest.exists();dest.mkdir()
            command=[cfg['evaluation']['python'],str(source/'src/evaluate.py'),'--config',str(protocol/'config.seed3407.json'),'--run-name',name,'--output-root',str(OUT),'--checkpoint',proof['model'],'--model-cache',cfg['evaluation']['model_cache'],'--dataset-cache',cfg['evaluation']['dataset_cache']]
            write_json(dest/'launcher.json',{'argv':command,'fresh_output_verified':True,'python_probe':plan['python_probe'],'checkpoint_sha256':proof['hashes'],'vllm_environment':{'VLLM_ENABLE_V1_MULTIPROCESSING':'0','VLLM_WORKER_MULTIPROC_METHOD':'spawn'},'source_hashes':plan['source_hashes'],'training_provenance':proof,'campaign_script_sha256':plan['helper_hashes']['tools/plain_mmmu900_campaign_20261102.py'],'recovery_script_sha256':receipt['recovery_script_sha256']})
            state(name,stage_started_at=utc_now())
            c.run_logged(command,'recovery_20261104/'+name+'.log')
            run=load_run(dest,3407,configs[3407],ids,'plain')
            validate_pair(ref['baseline']['records'],run['records'])
            for a,b in zip(ref['baseline']['records'],run['records']):assert a['model_response']['input_tokens']==b['model_response']['input_tokens']
            write_json(dest/'format_metrics.json',c.metrics(run['records']))
            if not repeat:
                plain['plain_epoch_4']=run;c.summarize(ref,plain,plan)
            else:
                check=compare_repeats(OUT/'plain_epoch_4_seed3407',dest);c.save('epoch_4_repeatability.json',check);assert check['passed']
        verify_preserved(receipt)
        assert c.helper_hashes()==receipt['original_helper_hashes']
        c.references(cfg,configs,ids)
        assert all(c.read(OUT/f'epoch_{e}_repeatability.json')['passed'] for e in (2,3,4))
        c.summarize(ref,plain,plan)
        c.save('completed.json',{'state':'training_and_seed3407_scoring_complete','completed_at':utc_now(),'primary_runs':3,'primary_responses':2700,'repeat_runs':3,'repeat_responses_excluded':2700,'all_repeats_passed':True,'old_baseline_era_preserved':True,'previous_completed_plain_preserved':True,'seeds':[3407],'semantic_review':'pending','recovery':'recovery_20261104','artifacts':{m:r['artifacts_sha256'] for m,r in plain.items()}})
        state('complete',semantic_review='pending')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-stop',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.prepare_stop:prepare_stop()
    elif a.execute:
        try:execute()
        except Exception:
            write_json(OUT/'recovery_failed.json',{'failed_at':utc_now(),'traceback':traceback.format_exc()});raise
    else:p.error('Choose --prepare-stop or --execute')
