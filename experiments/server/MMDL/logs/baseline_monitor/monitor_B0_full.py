"""Read-only evaluation inspection; writes only separate monitor records. No inference."""
import collections
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/home/rlatjswo/MMDL')
OUT = ROOT / 'results/baseline/B0_full'
DATA = Path('/home/rlatjswo/mmdl-data/mmmu_val')
MON = ROOT / 'logs/baseline_monitor'
MON.mkdir(parents=True, exist_ok=True)
lock = (MON / 'monitor.lock').open('a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('Another monitor owns the lock; skipped.')
    raise SystemExit(0)

def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()

def read(p):
    return json.loads(p.read_text()) if p.exists() else {}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def cmd(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'returncode': r.returncode, 'output': (r.stdout+r.stderr).strip()}
    except Exception as e:
        return {'returncode': None, 'output': str(e)}

def append(p, value):
    with p.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n')
        f.flush()
        os.fsync(f.fileno())

old = read(MON/'state.json')
if old.get('status') == 'completed':
    print(json.dumps({'status': 'completed', 'schedule': old.get('schedule'), 'action': 'Pause this automation only; never restart evaluation.'}))
    raise SystemExit(0)
stamp = now()
config = read(OUT/'config.json')
identity = config['identity']
ids = set(identity['ids'])
samples = [json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x.strip()]
issues = []
records = []
pred = OUT/'predictions.jsonl'
for attempt in range(3):
    raw = pred.read_bytes() if pred.exists() else b''
    records, malformed = [], []
    lines = raw.splitlines()
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            malformed.append(n)
    if malformed == [len(lines)] and attempt < 2:
        time.sleep(1)
        continue
    break
if malformed:
    issues.append({'jsonl_parse_errors': malformed, 'last_line_only': malformed == [len(lines)]})
latest = {r['id']: r for r in records}
counts = collections.Counter(latest.get(i, {}).get('status', 'pending') for i in ids)
unknown = sorted(set(latest)-ids)
if unknown:
    issues.append({'unexpected_ids': unknown})
if ids != {s['id'] for s in samples} or len(identity['ids']) != 900 or len(samples) != 900:
    issues.append('Expected ID set differs from 900-item validation dataset')
processes = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = proc.joinpath('cmdline').read_bytes().decode().strip('\x00').split('\x00')
        if not any(a.endswith('/baseline_eval.py') or a == 'baseline_eval.py' for a in args):
            continue
        if 'run' not in args:
            continue
        cwd = proc.joinpath('cwd').resolve()
        target = (cwd / args[args.index('--out')+1]).resolve()
        if target != OUT:
            continue
        processes.append({'pid': int(proc.name), 'args': args, 'cwd': str(cwd),
                          'ps': cmd(['ps','-p',proc.name,'-o','pid,ppid,lstart,etime,args'])['output'],
                          'start_ticks': proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]})
    except (OSError, ValueError, IndexError):
        continue
if len(processes)>1:
    issues.append('Multiple evaluation writers for B0_full; do not resume')
panes = cmd(['tmux','list-panes','-s','-t','baseline','-F','#{session_name}:#{window_index}.#{pane_index} #{pane_id} #{pane_pid} #{pane_current_command} #{pane_dead}'])
pane_files = []
(MON/'panes').mkdir(exist_ok=True)
if panes['returncode'] == 0:
    for line in panes['output'].splitlines():
        pane = line.split()[1]
        capture = cmd(['tmux','capture-pane','-p','-J','-t',pane,'-S','-200'])
        dest = MON/'panes'/f'{stamp.replace(":", "-")}_{pane.replace("%", "pane")}.txt'
        dest.write_text(capture['output'])
        pane_files.append(str(dest))
fixed = [OUT/'config.json', OUT/'environment.json', ROOT/'baseline_tools/baseline_eval.py', ROOT/'logs/baseline_server.json', DATA/'manifest.json', DATA/'samples.jsonl']
fixed += list((DATA/'official_mmmu').glob('*.py'))
hashes = {str(p):sha(p) for p in fixed}
baseline = old.get('initial_hashes', hashes)
changed = [p for p,h in baseline.items() if hashes.get(p) != h]
if changed:
    issues.append({'fixed_files_changed': changed})
manifest = read(DATA/'manifest.json')
if sha(ROOT/'baseline_tools/baseline_eval.py') != identity['script_sha256'] or sha(DATA/'manifest.json') != identity['data_manifest_sha256'] or sha(DATA/'samples.jsonl') != manifest['samples_sha256']:
    issues.append('Recorded code or dataset checksum mismatch')
for name,h in manifest['official_scorer']['sha256'].items():
    if sha(DATA/'official_mmmu'/name) != h:
        issues.append('Official parser checksum mismatch: '+name)
server = read(ROOT/'logs/baseline_server.json')
if server != config['server_record']:
    issues.append('Server record changed')
health = cmd(['curl','--connect-timeout','5','--max-time','10','-sS','-o','/dev/null','-w','%{http_code}','http://127.0.0.1:8000/health'])
models = cmd(['curl','--connect-timeout','5','--max-time','10','-sS','http://127.0.0.1:8000/v1/models'])
preflight = read(OUT/'preflight.json')
summary = read(OUT/'summary.json')
csv_rows = list(csv.DictReader((OUT/'sample_results.csv').open(encoding='utf-8-sig'))) if (OUT/'sample_results.csv').exists() else []
disk = os.statvfs(ROOT)
free_gib = disk.f_bavail*disk.f_frsize/1024**3
if free_gib < 10:
    issues.append('Disk below 5 GiB: preserve originals' if free_gib<5 else 'Disk below 10 GiB')
last_write = dt.datetime.fromtimestamp(pred.stat().st_mtime,dt.timezone.utc).astimezone().isoformat() if pred.exists() else None
stage = 'inference' if processes and records else 'preflight' if processes else 'stopped'
if summary.get('full_validation_complete') and not processes:
    stage = 'awaiting_completion_verification'
delta = counts['ok']-old.get('counts',{}).get('ok',counts['ok'])
if old.get('checked_at') and delta == 0 and (dt.datetime.fromisoformat(stamp)-dt.datetime.fromisoformat(old['checked_at'])).total_seconds()>=1800:
    issues.append('No new success in 30 minutes; inspect phase and progress; never terminate a live evaluation')
state = {**old, 'monitor_started_at': old.get('monitor_started_at',stamp), 'checked_at':stamp,
         'host':socket.gethostname(), 'user':cmd(['whoami'])['output'], 'status': 'blocked' if changed or unknown or len(processes)>1 or free_gib<5 else old.get('status','monitoring'),
         'stage':stage, 'processes':processes, 'initial_processes':old.get('initial_processes',processes),
         'tmux':panes, 'pane_files':pane_files, 'data_path':str(DATA), 'output_path':str(OUT),
         'counts':{'ok':counts['ok'],'error':counts['error'],'pending':counts['pending']}, 'new_successes':delta,
         'last_id':records[-1]['id'] if records else None, 'last_request_started_at':records[-1].get('timestamp') if records else None,
         'last_result_file_write_at':last_write, 'auto_recovery_count':old.get('auto_recovery_count',0),
         'initial_hashes':baseline, 'current_hashes':hashes, 'prediction_snapshot':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
         'previous_prediction_prefix_unchanged': None, 'attempts':len(records),'duplicate_attempts':len(records)-len(latest),
         'historical_errors':sum(r.get('status')=='error' for r in records),
         'length_stops_latest':sum(r.get('status')=='ok' and r.get('finish_reason')=='length' for r in latest.values()),
         'empty_responses':sum(r.get('status')=='ok' and not r.get('raw_response','').strip() for r in latest.values()),
         'parser_fallbacks_last_csv':sum(r.get('parser_random_fallback')=='True' for r in csv_rows),
         'csv_completed':sum(r.get('status')=='ok' for r in csv_rows),
         'summary_snapshot':summary, 'preflight':{k:v for k,v in preflight.items() if k!='samples'},
         'health':health,'models':models,'gpu':cmd(['nvidia-smi']), 'ram':cmd(['free','-h']),
         'disk':cmd(['df','-h',str(ROOT)]),'disk_free_gib':free_gib,
         'kernel_errors':cmd(['journalctl','-k','--since','30 minutes ago','--no-pager','-g','NVRM|Xid|Out of memory|oom-kill|CUDA']),
         'issues':issues, 'exit_code':'not recorded by original shell', 'schedule':old.get('schedule',{'status':'pending_registration'})}
previous = old.get('prediction_snapshot')
if previous:
    state['previous_prediction_prefix_unchanged'] = len(raw)>=previous['bytes'] and hashlib.sha256(raw[:previous['bytes']]).hexdigest()==previous['sha256']
    if not state['previous_prediction_prefix_unchanged']:
        issues.append('Original prediction prefix changed'); state['status']='blocked'
append(MON/'checks.jsonl',state)
(MON/'incidents.jsonl').touch(exist_ok=True)
if issues:
    append(MON/'incidents.jsonl',{'at':stamp,'issues':issues})
temp = MON/'state.json.tmp'
temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
temp.replace(MON/'state.json')
print(json.dumps({k:state[k] for k in ['checked_at','status','stage','counts','new_successes','last_id','last_result_file_write_at','processes','preflight','length_stops_latest','parser_fallbacks_last_csv','csv_completed','issues','schedule']},ensure_ascii=False,indent=2))
