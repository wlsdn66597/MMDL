"""CPU-only durable records, immutable protocol, and exact comparisons."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PILOT = Path('/home/rlatjswo/mmmu_batch_invariant_pilot_20261014')
PYTHON = PILOT/'.venv/bin/python'
OLD = Path('/home/rlatjswo/mmdl_test')


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def objsha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def write(p, data):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
    if os.name=='posix':
        fd=os.open(p.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def event(kind, **kwargs):
    p=ROOT/'events.jsonl'
    data=dict(time=now(),event=kind);data.update(kwargs)
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(data,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def make_plan():
    bundle=read(ROOT/'inputs/frozen.json'); ids=[r['id'] for r in bundle['items']]; subset=bundle['seed_subset_ids']
    assert len(ids)==60 and len(set(ids))==60 and len(subset)==12
    stages=[]
    def add(name,seed,chosen,batch,reference=None,cap=32768,**extra):
        stages.append(dict(name=name,seed=seed,ids=chosen,batch=batch,max_tokens=cap,reference=reference,**extra))
    add('api_smoke',3407,subset,1,'parent_pilot',cap=256)
    add('seed3407_reference',3407,ids,1)
    # Only executed when no natural baseline output actually reaches 32,768 tokens.
    force_ids=['validation_Math_13','validation_Psychology_17']
    add('forced32768_reference',3407,force_ids,1,force_full=True,conditional='no_natural_32768')
    add('forced32768_pair',3407,list(reversed(force_ids)),2,'forced32768_reference',force_full=True,conditional='no_natural_32768')
    resume_ids=['validation_Music_21','validation_Agriculture_13','validation_Math_13','validation_Psychology_17','validation_Clinical_Medicine_13','validation_Music_16']
    add('actual_resume_seed3407',3407,resume_ids,1,'seed3407_reference',inject_interrupt=True)
    add('seed3407_batch4',3407,list(reversed(ids)),4,'seed3407_reference')
    for seed in (42,123,2026,9876):
        add(f'seed{seed}_reference',seed,subset,1)
        add(f'seed{seed}_batch4',seed,subset[::2]+subset[1::2],4,f'seed{seed}_reference')
    plan=dict(schema=1,scope='60 items, 30 subjects x2; all five inference seeds on a 12-item subset; actual in-flight process interruption and recovery',
              frozen_inputs_sha256=sha(ROOT/'inputs/frozen.json'),environment_freeze_sha256=sha(ROOT/'provenance/pip_freeze.txt'),
              checkpoint_verification_sha256=sha(ROOT/'provenance/checkpoint_verification.json'),stages=stages,
              checkpoint_policy='One atomic durable JSON per completed item. Unfinished request is regenerated; already committed items are never regenerated or overwritten.',
              fault_policy='Only own actual_resume_seed3407 worker is SIGKILLed after >=2 committed items and >=128 observed tokens of an unfinished request. Then a new worker resumes pending IDs.',
              cache_policy='prefix cache off; MM processor cache zero; eager true; original resolution and prompts unchanged')
    dest=ROOT/'plan.json'
    if dest.exists(): assert read(dest)==plan, 'Refuse to alter an existing protocol.'
    else: write(dest,plan)
    return plan


def stage(name): return next(s for s in read(ROOT/'plan.json')['stages'] if s['name']==name)
def fingerprint(spec):
    return objsha(dict(spec=spec,plan_sha256=sha(ROOT/'plan.json'),runner_sha256=sha(ROOT/'runner.py'),helpers_sha256=sha(ROOT/'input_helpers.py')))


def records(name):
    p=ROOT/'outputs'/name
    return {f.stem:read(f) for f in sorted(p.glob('*.json'))} if p.exists() else {}


def check_record(row,spec,inputs):
    assert row['id'] in spec['ids'] and row['stage']==spec['name']
    assert row['seed']==spec['seed'] and row['max_tokens']==spec['max_tokens']
    assert row['fingerprint']==fingerprint(spec), 'Protocol/source fingerprint mismatch'
    assert len(row['token_ids'])==row['output_tokens']<=spec['max_tokens']
    assert row['input_tokens']==len(inputs[row['id']]['prompt_token_ids'])
    assert row['finish_reason'] in ('stop','length')
    if row['finish_reason']=='length': assert row['output_tokens']==spec['max_tokens']
    if spec.get('force_full'): assert row['output_tokens']==32768 and row['finish_reason']=='length'


def preserve_check():
    old=read(ROOT/'provenance/original_environment_before.json')
    after=dict(files={p:sha(OLD/p) for p in old['files']},pip_freeze=subprocess.check_output([str(OLD/'.venv/bin/python'),'-m','pip','freeze'],text=True))
    new_freeze=subprocess.check_output([str(PYTHON),'-m','pip','freeze'],text=True)
    result=dict(checked_at=now(),original_code_equal=after['files']==old['files'],original_packages_equal=after['pip_freeze']==old['pip_freeze'],
                new_environment_packages_equal=new_freeze==(ROOT/'provenance/pip_freeze.txt').read_text(encoding='utf-8'))
    result['passed']=all(result[k] for k in ('original_code_equal','original_packages_equal','new_environment_packages_equal'))
    write(ROOT/'provenance/environment_preservation.json',result)
    assert result['passed'], 'Environment drift detected'
    return result


def summarize():
    plan=read(ROOT/'plan.json'); bundle=read(ROOT/'inputs/frozen.json'); inputs={r['id']:r for r in bundle['items']}
    comparisons=[]; counts={}; complete=True; natural=[]; forced=[]
    for s in plan['stages']:
        name=s['name']; skip=ROOT/'skipped'/f'{name}.json'; data=records(name)
        if skip.exists():
            assert s.get('conditional')=='no_natural_32768'
            assert any(r['output_tokens']==32768 and r['finish_reason']=='length' for r in records('seed3407_reference').values())
            assert not data
            counts[name]=dict(state='skipped',completed=0,expected=0,reason=read(skip)['reason']);continue
        counts[name]=dict(state='complete' if set(data)==set(s['ids']) else 'pending',completed=len(data),expected=len(s['ids']))
        if set(data)!=set(s['ids']): complete=False
        for row in data.values():
            check_record(row,s,inputs)
            if row['output_tokens']==32768:
                (forced if s.get('force_full') else natural).append(dict(stage=name,id=row['id']))
        if s['reference']=='parent_pilot':
            ref={r['id']:r for r in (json.loads(x) for x in (PILOT/'outputs/vl_reference.jsonl').read_text().splitlines())}
        else: ref=records(s['reference']) if s['reference'] else {}
        if s['reference']:
            for k,row in data.items():
                assert k in ref, (name,k,'missing reference')
                diff=[f for f in ('token_ids','text','finish_reason','stop_reason','input_tokens') if row[f]!=ref[k][f]]
                comparisons.append(dict(stage=name,id=k,passed=not diff,differences=diff))
    recovery=read(ROOT/'recovery_verification.json') if (ROOT/'recovery_verification.json').exists() else None
    result=dict(updated_at=now(),complete=complete,passed=complete and bool(comparisons) and all(c['passed'] for c in comparisons) and bool(natural or forced) and bool(recovery and recovery['passed']),
                stage_counts=counts,comparisons=comparisons,comparison_count=len(comparisons),differences=sum(not c['passed'] for c in comparisons),
                natural_32768_records=natural,forced_32768_records=forced,recovery=recovery)
    write(ROOT/'summary.json',result)
    return result
