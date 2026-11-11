"""Durable isolated setup/pilot controller. PAUSE stops at the next batch boundary."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from pilot import ROOT, STAGES, append, now, read, records, stage_items, summarize, write

PYTHON = ROOT / '.venv/bin/python'
OLD = Path('/home/rlatjswo/mmdl_test/.venv/bin/python')


def state(s, **kwargs):
    write(ROOT / 'status.json', dict(state=s, updated_at=now(), pid=os.getpid(), **kwargs))


def original_unchanged():
    before = read(ROOT / 'provenance/original_environment_before.json')
    old_root = OLD.parents[2]
    after = dict(python=str(OLD), files={p:hashlib.sha256((old_root / p).read_bytes()).hexdigest() for p in before['files']},
                 pip_freeze=subprocess.check_output([str(OLD),'-m','pip','freeze'], text=True))
    write(ROOT / 'provenance/original_environment_after.json', after)
    comparison = dict(checked_at=now(), files_equal=after['files']==before['files'],
                      pip_freeze_equal=after['pip_freeze']==before['pip_freeze'])
    comparison['passed'] = comparison['files_equal'] and comparison['pip_freeze_equal']
    write(ROOT / 'provenance/original_environment_preserved.json', comparison)
    if not comparison['passed']:
        raise RuntimeError('Original environment changed externally or unexpectedly; inspect before continuing.')


def run_logged(args, log):
    with (ROOT / log).open('a') as f:
        f.write('\nSTART ' + now() + ' ' + repr(args) + '\n')
        f.flush()
        return subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT).returncode


def setup(resolution_pid):
    state('resolving_dependencies', resolution_pid=resolution_pid)
    # /proc exists for zombie children too, so inspect status rather than kill(0).
    while resolution_pid:
        stat = Path(f'/proc/{resolution_pid}/stat')
        if not stat.exists() or stat.read_text().split(') ')[1].startswith('Z'):
            break
        if (ROOT / 'PAUSE').exists():
            state('paused', phase='waiting_for_dependency_resolution')
            return
        time.sleep(15)
    report = ROOT / 'provenance/install_resolution.json'
    if not report.exists():
        raise RuntimeError('pip dependency dry run did not produce a successful resolution report; see setup_dry_run.log.')
    resolved = read(report)
    pins = sorted(f"{r['metadata']['name']}=={r['metadata']['version']}" for r in resolved['install'])
    (ROOT / 'provenance/resolved_requirements.txt').write_text('\n'.join(pins)+'\n')
    state('installing')
    rc = run_logged([PYTHON,'-m','pip','install','--report',ROOT/'provenance/installed_report.json',
                     '-r',ROOT/'provenance/resolved_requirements.txt'], 'install.log')
    if rc:
        raise RuntimeError(f'Isolated installation failed ({rc}); see install.log.')
    rc = run_logged([PYTHON,'-m','pip','check'], 'pip_check.log')
    if rc:
        raise RuntimeError('pip check failed; no GPU inference is allowed.')
    (ROOT / 'provenance/pip_freeze.txt').write_text(subprocess.check_output([str(PYTHON),'-m','pip','freeze'],text=True))
    original_unchanged()
    state('input_preflight')
    rc = run_logged([PYTHON,'-B',ROOT/'pilot.py','preflight'], 'preflight.log')
    if rc:
        raise RuntimeError('New environment input/flag check failed; see preflight.log.')
    state('ready_for_gpu', note='Launch run only after the visualization thread has finished and GPU is free.')


def gpu_idle():
    p = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'], text=True,capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    active = [s for s in p.stdout.splitlines() if s.strip()]
    return not active, active


def run():
    assert read(ROOT/'provenance/input_parity.json')['passed']
    original_unchanged()
    bundle = read(ROOT/'inputs/frozen.json')
    for name, batch, cap, order in STAGES:
        expected = len(stage_items(bundle,name,order))
        got = len(records(ROOT/'outputs'/f'{name}.jsonl'))
        if got == expected:
            continue
        if (ROOT/'PAUSE').exists():
            state('paused', stage=name)
            return
        idle, processes = gpu_idle()
        if not idle:
            state('waiting_for_gpu', next_stage=name, gpu_processes=processes)
            # Return safely; caller may resume once the other task releases its GPU.
            return
        state('running', stage=name, completed_stages=sum((ROOT/'outputs'/f'{s[0]}.jsonl').exists() and len(records(ROOT/'outputs'/f'{s[0]}.jsonl'))==len(stage_items(bundle,s[0],s[3])) for s in STAGES), total_stages=len(STAGES))
        rc = run_logged([PYTHON,'-B',ROOT/'pilot.py','stage','--name',name], f'{name}.log')
        if rc == 75:
            state('paused', stage=name)
            return
        if rc:
            raise RuntimeError(f'{name} failed ({rc}); saved completed outputs are retained.')
        partial = summarize()
        if any(not r['passed'] for r in partial['comparisons']):
            original_unchanged()
            state('mismatch', stage=name, summary='summary.json')
            return
        time.sleep(3)  # Allow process teardown to release CUDA allocations.
    original_unchanged()
    final = summarize()
    state('complete' if final['passed'] else 'mismatch', summary='summary.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['setup','run','status','pause','resume'])
    p.add_argument('--resolution-pid', type=int, default=0)
    a = p.parse_args()
    if a.action == 'status':
        print(json.dumps(read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}, indent=2))
        if (ROOT/'inputs/frozen.json').exists():
            bundle = read(ROOT/'inputs/frozen.json')
            for name,_,_,order in STAGES:
                print(name, len(records(ROOT/'outputs'/f'{name}.jsonl')), '/', len(stage_items(bundle,name,order)))
        return
    if a.action == 'pause':
        (ROOT/'PAUSE').touch()
        append(ROOT/'events.jsonl',dict(time=now(),event='pause_requested'))
        return
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == 'resume':
            (ROOT/'PAUSE').unlink(missing_ok=True)
            append(ROOT/'events.jsonl',dict(time=now(),event='resume_requested'))
        try:
            setup(a.resolution_pid) if a.action == 'setup' else run()
        except Exception as e:
            state('error', message=str(e))
            raise


if __name__ == '__main__':
    main()
