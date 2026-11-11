"""Pause/resume remaining seeds without touching the running 3407 check."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from mmmu_multiseed_report import read, write
from mmmu_resume_utils import REMAINING, active_group, campaign_process, event, proc


def pause(root):
    status = read(root / 'status.json')
    if status.get('current_run') not in REMAINING:
        raise RuntimeError('Pause is enabled only for seeds 123, 2026 and 9876; current 3407 check is protected')
    item = campaign_process(root, status)
    if item is None:
        raise RuntimeError('No live campaign driver; inspect state before resuming')
    pgid = item['pgid']
    write(root / 'pause.request.json', {'requested_at_unix': time.time(), 'driver_pid': item['pid'], 'run': status['current_run']})
    event(root, 'pause_requested', run=status['current_run'], driver_pid=item['pid'], start_ticks=item['start_ticks'])
    latest = proc(item['pid'])
    if not latest or latest['start_ticks'] != item['start_ticks']:
        raise RuntimeError('Driver changed during pause request')
    os.killpg(pgid, signal.SIGINT)
    for sig, timeout in [(None, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)]:
        if sig is not None and active_group(pgid):
            os.killpg(pgid, sig)
        until = time.monotonic() + timeout
        while active_group(pgid) and time.monotonic() < until:
            time.sleep(0.2)
    if active_group(pgid):
        raise RuntimeError('Campaign processes still alive; do not resume')
    status = read(root / 'status.json')
    status.update(state='paused', phase='paused_by_user', paused_at_unix=time.time(), error=None)
    write(root / 'status.json', status)
    event(root, 'paused', run=status['current_run'], process_group_stopped=True)
    print(json.dumps(status))


def resume(root):
    status = read(root / 'status.json')
    if campaign_process(root, status) is not None:
        raise RuntimeError('Campaign is already running; no second process will be started')
    if not (root / 'repeatability.json').exists() or not read(root / 'repeatability.json')['passed']:
        raise RuntimeError('Resume requires a passed 3407 repeatability check')
    if status['state'] == 'complete':
        print('All seeds already complete')
        return
    session = 'mm_multiseed_resume_' + str(time.time_ns())
    command = 'bash ' + shlex.quote(str(root / 'resume_mmmu_multiseed.sh'))
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
    event(root, 'resume_submitted', tmux_session=session)
    print(json.dumps({'submitted': True, 'tmux_session': session, 'note': 'Check status.json and driver.log for preflight and actual inference'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'pause', 'resume'])
    parser.add_argument('--root', default='/home/rlatjswo/mmmu_multiseed_20261011')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.action == 'status':
        status = read(root / 'status.json')
        status['live_driver'] = campaign_process(root, status)
        path = root / (status.get('current_run') or '') / 'predictions.jsonl'
        if path.exists():
            with path.open('rb') as handle:
                status['prediction_lines'] = sum(line.endswith(b'\n') for line in handle)
        print(json.dumps(status, indent=2))
        return
    with (root / '.control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'pause': pause, 'resume': resume}[args.action](root)


if __name__ == '__main__':
    main()
