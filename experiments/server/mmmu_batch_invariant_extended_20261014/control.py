"""Sequential campaign controller; fault injection is limited to its own child PID."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
from common import ROOT,PYTHON,read,write,event,now,records,sha,make_plan,summarize,preserve_check


def state(value,**kwargs):write(ROOT/'status.json',dict(state=value,updated_at=now(),pid=os.getpid(),**kwargs))


def gpu_idle():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return [s for s in p.stdout.splitlines() if s.strip()]


def worker(spec):
    name=spec['name'];log=ROOT/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\nSTART '+now()+'\n');f.flush()
        child=subprocess.Popen([str(PYTHON),'-B',str(ROOT/'runner.py'),'--stage',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        event('worker_spawned',stage=name,pid=child.pid)
        injected=False
        while child.poll() is None:
            progress=ROOT/'progress'/f'{name}.json'
            if spec.get('inject_interrupt') and not (ROOT/'fault_injected.json').exists() and progress.exists() and not (ROOT/'PAUSE').exists():
                p=read(progress)
                inflight=[x for x in p['active'] if not x['finished'] and x['generated_tokens']>=128]
                committed=records(name)
                if p['pid']==child.pid and len(committed)>=2 and inflight:
                    cmdline=Path(f'/proc/{child.pid}/cmdline').read_bytes().split(b'\0')
                    assert str(ROOT/'runner.py').encode() in cmdline and name.encode() in cmdline
                    before={k:sha(ROOT/'outputs'/name/f'{k}.json') for k in committed}
                    assert not any(x['id'] in before for x in inflight)
                    evidence=dict(time=now(),pid=child.pid,stage=name,attempt=p['attempt'],signal='SIGKILL',
                                  observed_inflight=inflight,committed_sha256=before,process_cmdline=[x.decode() for x in cmdline if x])
                    write(ROOT/'fault_injected.json',evidence)
                    event('planned_inflight_sigkill',**evidence)
                    os.kill(child.pid,signal.SIGKILL)
                    injected=True
            time.sleep(.25)
        rc=child.wait()
        event('worker_exited',stage=name,pid=child.pid,returncode=rc,planned_injection=injected)
        return rc,injected


def verify_recovery():
    fault=read(ROOT/'fault_injected.json');name=fault['stage'];after=records(name)
    preserved=all(sha(ROOT/'outputs'/name/f'{i}.json')==h for i,h in fault['committed_sha256'].items())
    before_attempt=fault['attempt']
    retained_ids=[i for i,r in after.items() if r['attempt']==before_attempt]
    restarted_attempts={r['attempt'] for i,r in after.items() if i not in fault['committed_sha256']}
    ref=records('seed3407_reference')
    fields=('token_ids','text','finish_reason','stop_reason','input_tokens')
    differences={i:[k for k in fields if r[k]!=ref[i][k]] for i,r in after.items()}
    passed=preserved and set(retained_ids)==set(fault['committed_sha256']) and len(restarted_attempts)==1 and not any(differences.values())
    result=dict(checked_at=now(),passed=passed,preserved_committed_files=preserved,
                committed_before_kill=len(fault['committed_sha256']),completed_after_restart=len(after),
                retained_ids=retained_ids,new_worker_attempts=sorted(restarted_attempts),
                interrupted_requests=fault['observed_inflight'],differences=differences)
    write(ROOT/'recovery_verification.json',result)
    assert passed,'Actual recovery test failed'


def run():
    plan=make_plan();assert read(ROOT/'provenance/input_parity.json')['passed'];preserve_check()
    for spec in plan['stages']:
        name=spec['name']
        if spec.get('conditional') and any(r['output_tokens']==32768 and r['finish_reason']=='length' for r in records('seed3407_reference').values()):
            assert not records(name)
            write(ROOT/'skipped'/f'{name}.json',dict(reason='Natural 32,768-token length termination already observed in seed3407_reference. No forced-EOS diagnostic needed.',time=now()))
            continue
        got=records(name)
        if set(got)==set(spec['ids']):
            if spec.get('inject_interrupt'):verify_recovery()
            continue
        if (ROOT/'PAUSE').exists():state('paused',stage=name);return
        active=gpu_idle()
        if active:state('waiting_for_gpu',next_stage=name,compute_processes=active);return
        state('running',stage=name,seed=spec['seed'],completed=len(got),expected=len(spec['ids']),max_tokens=spec['max_tokens'])
        rc,injected=worker(spec)
        if rc==75 or (ROOT/'PAUSE').exists():state('paused',stage=name);return
        if injected:
            assert rc==-signal.SIGKILL
            fault=read(ROOT/'fault_injected.json')
            assert all(sha(ROOT/'outputs'/name/f'{i}.json')==h for i,h in fault['committed_sha256'].items())
            event('planned_recovery_ready',stage=name,retained_ids=list(fault['committed_sha256']))
            time.sleep(5)
            active=gpu_idle()
            if active:state('waiting_for_gpu',next_stage=name,compute_processes=active);return
            if (ROOT/'PAUSE').exists():state('paused',stage=name);return
            rc,again=worker(spec);assert not again
        if rc==75:state('paused',stage=name);return
        if rc:raise RuntimeError(f'{name} failed with exit {rc}; committed items retained.')
        if spec.get('inject_interrupt'):verify_recovery()
        result=summarize()
        if result['differences']:preserve_check();state('mismatch',stage=name,differences=result['differences']);return
        time.sleep(3)
    preserve_check();final=summarize()
    state('complete' if final['passed'] else 'incomplete',summary='summary.json',comparisons=final['comparison_count'])


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run','status','pause','resume']);a=p.parse_args()
    if a.action=='plan':
        plan=make_plan();print('stages',len(plan['stages']));return
    if a.action=='status':
        print((ROOT/'status.json').read_text() if (ROOT/'status.json').exists() else '{}')
        if (ROOT/'plan.json').exists():
            for s in read(ROOT/'plan.json')['stages']:print(s['name'],len(records(s['name'])),'/',len(s['ids']))
        return
    if a.action=='pause':
        (ROOT/'PAUSE').touch();event('user_pause_requested');return
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='resume':
            (ROOT/'PAUSE').unlink(missing_ok=True);event('user_resume_requested')
        try:run()
        except Exception as e:
            state('error',message=str(e));raise


if __name__=='__main__':main()
