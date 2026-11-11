"""Durable resume helpers; the frozen evaluation code remains unchanged."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from mmmu_multiseed_report import read, write

REMAINING = ('baseline_seed123', 'baseline_seed2026', 'baseline_seed9876')


def event(root, action, **details):
    record = {'time_unix': time.time(), 'action': action, **details}
    with (Path(root) / 'control_events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def repair_jsonl(path):
    """Keep every valid row; quarantine only an invalid unterminated last row."""
    path = Path(path)
    if not path.exists():
        return {'records': 0, 'repaired': False}
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    offset = 0
    rows = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('JSONL record must be an object')
        except (ValueError, UnicodeDecodeError):
            if index != len(lines)-1 or line.endswith(b'\n'):
                raise ValueError(f'Corrupt complete row in {path}; refusing recovery')
            backup = path.with_name(path.name + f'.interrupted-{time.time_ns()}.bak')
            shutil.copy2(path, backup)
            with path.open('r+b') as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            return {'records': len(rows), 'repaired': True, 'backup': str(backup), 'discarded_incomplete_bytes': len(line)}
        rows.append(record)
        offset += len(line)
    if content and not content.endswith(b'\n'):
        # A complete final JSON object without a delimiter must not merge with the next append.
        with path.open('ab') as handle:
            handle.write(b'\n')
            handle.flush()
            os.fsync(handle.fileno())
    return {'records': len(rows), 'repaired': False}


def check_manifest(folder, config, identity, code):
    folder = Path(folder)
    path = folder / 'manifest.json'
    if not path.exists():
        if folder.exists() and any(p.name != '.run.lock' for p in folder.iterdir()):
            raise ValueError('Existing run has no manifest: ' + str(folder))
        return
    manifest = read(path)
    expected = hashlib.sha256(json.dumps({'configuration': config, 'model_identity': identity, 'code': code}, sort_keys=True).encode()).hexdigest()
    if manifest['fingerprint'] != expected or manifest['mode'] != 'inference':
        raise ValueError('Resume refused: code, configuration or checkpoint differs')


def proc(pid):
    try:
        folder = Path('/proc') / str(pid)
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        uid = folder.stat().st_uid
        return {'pid': int(pid), 'state': fields[0], 'pgid': int(fields[2]),
                'start_ticks': fields[19], 'uid': uid,
                'argv': (folder / 'cmdline').read_bytes().decode().strip('\0').split('\0') if uid == os.getuid() else []}
    except (FileNotFoundError, ProcessLookupError):
        return None


def active_group(pgid):
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        item = proc(int(path.name))
        if item and item['pgid'] == pgid and item['state'] != 'Z':
            if item['uid'] != os.getuid():
                raise RuntimeError('Process group includes a different user')
            members.append(item['pid'])
    return members


def campaign_process(root, status):
    item = proc(status.get('driver_pid', -1))
    if not item or item['state'] == 'Z':
        return None
    argv = item['argv']
    allowed = {str(Path(root) / name) for name in ('run_mmmu_multiseed.py', 'resume_mmmu_multiseed.py')}
    if item['uid'] != os.getuid() or item['pgid'] != item['pid'] or not allowed.intersection(argv):
        raise RuntimeError('Campaign PID identity check failed')
    if '--root' not in argv or argv[argv.index('--root')+1] != str(root):
        raise RuntimeError('Campaign root identity check failed')
    return item
