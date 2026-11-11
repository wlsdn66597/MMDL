"""Explicit, restartable evaluation-only controller. Frozen experiment code is unchanged."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/plain_mmmu900_20261102'
RUN_NAMES = [f'plain_epoch_{e}_seed3407{suffix}' for e in (2, 3, 4)
             for suffix in ('', '_repeat')]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def journal(path):
    """Refuse corrupt records; only an unterminated final line may be repaired."""
    if not path.exists():
        return [], None
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows = []
    offset = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if index == len(lines) - 1 and not line.endswith(b'\n'):
                return rows, {'kind': 'incomplete_tail', 'offset': offset}
            raise ValueError(f'Corrupt complete JSONL record: {path}:{index + 1}')
        if not isinstance(row, dict):
            raise ValueError(f'Non-object JSONL record: {path}:{index + 1}')
        rows.append(row)
        offset += len(line)
    if lines and not data.endswith(b'\n'):
        return rows, {'kind': 'missing_newline', 'offset': len(data)}
    return rows, None


def repair_journal(path, archive):
    rows, damage = journal(path)
    if not damage:
        return
    original = path.read_bytes()
    backup = archive / (path.name + '.before-repair')
    with backup.open('xb') as f:
        f.write(original); f.flush(); os.fsync(f.fileno())
    repaired = (original[:damage['offset']] if damage['kind'] == 'incomplete_tail'
                else original + b'\n')
    atomic(archive / (path.name + '.repair.json'), dict(damage,
           source=str(path), original_sha256=digest(backup), complete_records=len(rows)))
    temp = path.with_name(path.name + '.repair-' + uuid.uuid4().hex)
    with temp.open('xb') as f:
        f.write(repaired); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def snapshot(directory, archive):
    archive.mkdir()
    hashes = {}
    for p in directory.iterdir():
        if p.is_file() and p.name != '.run.lock':
            target = archive / p.name
            with p.open('rb') as src, target.open('xb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
                dst.flush(); os.fsync(dst.fileno())
            hashes[p.name] = digest(target)
            if hashes[p.name] != digest(p):
                raise RuntimeError(f'File changed while snapshotting: {p}')
    atomic(archive / 'snapshot_hashes.json', hashes)


def check_append_only(directory, saved):
    for name, prefix in saved.items():
        with (directory / name).open('rb') as f:
            if f.read(len(prefix)) != prefix:
                raise ValueError(f'Saved responses were modified: {directory / name}')


def require_prefix(rows, ids):
    actual = [r['id'] for r in rows]
    if actual != ids[:len(actual)] or len(actual) != len(set(actual)):
        raise ValueError('Saved prediction IDs are not a unique frozen prefix')


def existing_workers():
    found = []
    for p in Path('/proc').glob('[0-9]*'):
        if int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            stat = (p / 'stat').read_text().split(') ', 1)[1].split()
            if stat[0] == 'Z':
                continue
            is_eval = any(a.endswith('/evaluate.py') for a in args) and str(OUT) in args
            is_controller = any(Path(a).name in (
                'plain_mmmu900_campaign_20261102.py', 'recover_plain_mmmu900_20261104.py',
                'resume_plain_mmmu900.py', 'plain_epoch_export_20261102.py') for a in args)
            if is_eval or is_controller:
                found.append({'pid': int(p.name), 'argv': args})
        except (OSError, UnicodeError):
            continue
    return found


def child(command, log, lock_fd, cpu=False, timeout=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='' if cpu else '0',
               VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    env.pop('PYTHONPATH', None); env.pop('VLLM_BATCH_INVARIANT', None)
    with log.open('x') as f:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, pass_fds=(lock_fd,))
        try:
            code = p.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'Child exit {code}; see {log}')
        except BaseException:
            # Only this invocation's process group, including evaluator children.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise


def backend():
    sys.path.insert(0, str(ROOT))
    import plain_mmmu900_campaign_20261102 as c
    from plain_epoch_export_20261102 import validate_export, model_path
    cfg, source, protocol, configs, ids, refs, plan = c.preflight()
    plain_dir = c.output_directory(cfg, 'plain')
    era_dir = c.output_directory(cfg, 'era')
    c.verify_pair_manifest(read(era_dir / 'run_manifest.json'), read(plain_dir / 'run_manifest.json'))
    proof = read(OUT / 'training_pair_verification.json')
    assert proof['passed'] and proof['plain_manifest_sha256'] == digest(plain_dir / 'run_manifest.json')
    assert proof['era_manifest_sha256'] == digest(era_dir / 'run_manifest.json')
    assert c.helper_hashes() == plan['helper_hashes']
    # Honor the preservation receipt from the earlier suspend recovery.
    old = OUT / 'recovery_20261104/prepare_receipt.json'
    if old.exists():
        for name, h in read(old)['preserved_files'].items():
            assert digest(ROOT / name) == h, name
        for name, h in read(old)['snapshot_files'].items():
            assert digest(old.parent / name) == h, name
        saved = (old.parent / 'interrupted_repeat/predictions.jsonl').read_bytes()
        check_append_only(OUT / 'plain_epoch_3_seed3407_repeat', {'predictions.jsonl': saved})
    sys.path.insert(0, str(source / 'src'))
    spec = importlib.util.spec_from_file_location('frozen_resume_evaluate', source / 'src/evaluate.py')
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    return c, cfg, source, protocol, configs, ids, refs, plan, validate_export, model_path, frozen


def inspect_run(dest, b, proof):
    c, cfg, source, protocol, configs, ids, refs, plan, _, _, frozen = b
    if not dest.exists():
        return {'state': 'new', 'saved': 0}
    rows, tail = journal(dest / 'predictions.jsonl')
    judges, judge_tail = journal(dest / 'judge_attempts.jsonl')
    require_prefix(rows, ids)
    frozen.check_records(rows, configs[3407], partial=True)
    for row, base in zip(rows, refs['baseline']['records']):
        for key in c.validate_pair.__globals__['INPUT_FIELDS']:
            assert row.get(key) == base.get(key), (row['id'], key)
        assert row['model_response']['input_tokens'] == base['model_response']['input_tokens']
    frozen.saved_judges(rows, judges, configs[3407])
    launch = dest / 'launcher.json'
    if launch.exists():
        saved = read(launch)
        assert proof is not None and saved['checkpoint_sha256'] == proof['hashes']
        assert saved['argv'] == command_for(dest.name, b, proof)
        assert saved['source_hashes'] == plan['source_hashes']
        assert saved['python_probe'] == plan['python_probe'] and saved['fresh_output_verified']
        assert saved['vllm_environment'] == {'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'}
    elif any(p.name != '.run.lock' for p in dest.iterdir()):
        raise ValueError(f'Missing launcher provenance: {dest}')
    if (dest / 'manifest.json').exists():
        m = read(dest / 'manifest.json')
        assert proof is not None and m['configuration'] == configs[3407] and m['mode'] == 'inference'
        assert m['model_identity']['path'] == proof['model']
        for key, h in m['model_identity']['weights_sha256'].items():
            assert proof['hashes'][key] == h
        assert m['model_identity']['config_sha256'] == digest(Path(proof['model']) / 'config.json')
        code = {str(p.relative_to(source)): digest(p) for p in (source / 'src').glob('*') if p.is_file()}
        assert m['code_sha256'] == code
        signature = hashlib.sha256(json.dumps({'configuration': configs[3407],
                    'model_identity': m['model_identity'], 'code': code}, sort_keys=True).encode()).hexdigest()
        assert m['fingerprint'] == signature
    elif rows or judges:
        raise ValueError(f'Missing manifest for saved responses: {dest}')
    complete = (dest / 'results.json').exists() and (dest / 'summary.json').exists()
    if complete:
        if tail or judge_tail:
            raise ValueError('A completed run has a damaged journal; no automatic repair')
        run = c.load_run(dest, 3407, configs[3407], ids, 'plain')
        c.validate_pair(refs['baseline']['records'], run['records'])
    return {'state': 'complete' if complete else 'partial', 'saved': len(rows),
            'judge_attempts_saved': len(judges), 'tail_repair_needed': bool(tail or judge_tail)}


def command_for(name, b, proof):
    _, cfg, source, protocol, *_ = b
    return [cfg['evaluation']['python'], str(source / 'src/evaluate.py'), '--config',
            str(protocol / 'config.seed3407.json'), '--run-name', name, '--output-root', str(OUT),
            '--checkpoint', proof['model'], '--model-cache', cfg['evaluation']['model_cache'],
            '--dataset-cache', cfg['evaluation']['dataset_cache']]


def write_same_or_new(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing validation receipt differs: {path}')
    else:
        atomic(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='CPU-only inventory; no inference or file repair')
    mode.add_argument('--resume', action='store_true', help='Explicitly resume unfinished evaluation')
    parser.add_argument('--repair-tail', action='store_true', help='Archive and repair only an unterminated last JSONL record')
    args = parser.parse_args()
    if args.check and args.repair_tail:
        parser.error('--repair-tail requires --resume')
    import fcntl
    with (OUT / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        workers = existing_workers()
        if workers:
            raise RuntimeError(f'Existing evaluation/controller processes; refusing duplicate: {workers}')
        b = backend()
        c, cfg, source, protocol, configs, ids, refs, plan, validate_export, model_path, frozen = b
        proofs = {e: validate_export(cfg, e) if model_path(cfg, e).exists() else None for e in (2, 3, 4)}
        states = {name: inspect_run(OUT / name, b, proofs[int(name.split('_')[2])]) for name in RUN_NAMES}
        print(json.dumps({'checked_at': now(), 'training_restarted': False, 'runs': states}, indent=2), flush=True)
        if args.check:
            return
        home = OUT / 'resume_attempts'; home.mkdir(exist_ok=True)
        attempt = home / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        attempt.mkdir()
        latest = OUT / 'resume_status.json'
        def state(stage, **extra):
            value = dict(stage=stage, updated_at=now(), attempt=str(attempt), **extra)
            atomic(attempt / 'status.json', value); atomic(latest, value)
        atomic(attempt / 'start.json', dict(started_at=now(), inventory=states,
               script_sha256=digest(__file__), plan_sha256=digest(OUT / 'plan.json'),
               original_helper_hashes=plan['helper_hashes'], repair_tail_authorized=args.repair_tail))
        preserved = {str(p): digest(p) for name in RUN_NAMES if states[name]['state'] == 'complete'
                     for p in (OUT / name).iterdir() if p.is_file() and p.name != '.run.lock'}
        atomic(attempt / 'completed_before.json', preserved)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            if any(s['state'] != 'complete' for s in states.values()):
                state('cuda_check')
                child([cfg['evaluation']['python'], '-c', 'import torch; x=torch.ones(1,device="cuda"); y=x+1; torch.cuda.synchronize(); assert y.item()==2; print("CUDA ready")'],
                      attempt / 'cuda.log', lock.fileno(), timeout=60)
            plain = {}
            for epoch in (2, 3, 4):
                proof = proofs[epoch]
                if proof is None:
                    state(f'export_epoch_{epoch}')
                    child([str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'tools/plain_epoch_export_20261102.py'),
                           '--config', str(c.CONFIG), '--epoch', str(epoch), '--execute'],
                          attempt / f'export_{epoch}.log', lock.fileno(), cpu=True)
                    proof = validate_export(cfg, epoch)
                write_same_or_new(OUT / f'export_epoch_{epoch}.json', proof)
                for repeat in (False, True):
                    name = f'plain_epoch_{epoch}_seed3407' + ('_repeat' if repeat else '')
                    dest = OUT / name
                    if states[name]['state'] != 'complete':
                        dest.mkdir(exist_ok=True)
                        before = attempt / (name + '.before')
                        snapshot(dest, before)
                        for fn in ('predictions.jsonl', 'judge_attempts.jsonl'):
                            _, tail = journal(dest / fn)
                            if tail:
                                if not args.repair_tail:
                                    raise ValueError('Unterminated JSONL tail; retry explicitly with --repair-tail')
                                repair_journal(dest / fn, before)
                        if not (dest / 'launcher.json').exists():
                            atomic(dest / 'launcher.json', dict(argv=command_for(name, b, proof), fresh_output_verified=True,
                                   python_probe=plan['python_probe'], checkpoint_sha256=proof['hashes'],
                                   vllm_environment={'VLLM_ENABLE_V1_MULTIPROCESSING':'0','VLLM_WORKER_MULTIPROC_METHOD':'spawn'},
                                   source_hashes=plan['source_hashes'], training_provenance=proof,
                                   campaign_script_sha256=plan['helper_hashes']['tools/plain_mmmu900_campaign_20261102.py'],
                                   resume_script_sha256=digest(__file__)))
                        inspect_run(dest, b, proof)
                        saved = {fn: (dest / fn).read_bytes() for fn in ('predictions.jsonl', 'judge_attempts.jsonl') if (dest / fn).exists()}
                        state(name, saved_responses=states[name]['saved'])
                        child(command_for(name, b, proof), attempt / (name + '.log'), lock.fileno())
                        check_append_only(dest, saved)
                    run = c.load_run(dest, 3407, configs[3407], ids, 'plain')
                    c.validate_pair(refs['baseline']['records'], run['records'])
                    for a, r in zip(refs['baseline']['records'], run['records']):
                        assert a['model_response']['input_tokens'] == r['model_response']['input_tokens']
                    if not repeat:
                        plain[f'plain_epoch_{epoch}'] = run
                    else:
                        receipt = c.compare_repeats(OUT / f'plain_epoch_{epoch}_seed3407', dest)
                        atomic(attempt / f'epoch_{epoch}_repeatability.json', receipt)
                        assert receipt['passed'], f'Epoch {epoch} repeatability failed; subsequent epochs blocked'
                        write_same_or_new(OUT / f'epoch_{epoch}_repeatability.json', receipt)
            for name, h in preserved.items():
                assert digest(name) == h, name
            assert digest(__file__) == read(attempt / 'start.json')['script_sha256']
            assert c.helper_hashes() == plan['helper_hashes']
            c.references(cfg, configs, ids)
            c.summarize(refs, plain, plan)
            completion = dict(state='training_and_seed3407_scoring_complete', completed_at=now(),
                              primary_runs=3, primary_responses=2700, repeat_runs=3, repeat_responses_excluded=2700,
                              all_repeats_passed=True, old_baseline_era_preserved=True, seeds=[3407],
                              semantic_review='pending', artifacts={m:r['artifacts_sha256'] for m,r in plain.items()})
            if not (OUT / 'completed.json').exists():
                atomic(OUT / 'completed.json', completion)
            atomic(attempt / 'completed.json', completion)
            atomic(attempt / 'exit.json', {'code': 0, 'finished_at': now()})
            state('complete', semantic_review='pending')
        except BaseException:
            atomic(attempt / 'failed.json', {'failed_at': now(), 'traceback': traceback.format_exc()})
            atomic(attempt / 'exit.json', {'code': 1, 'finished_at': now()})
            state('stopped', reason='Inspect this attempt failure/log; explicit resume required')
            raise


if __name__ == '__main__':
    main()
