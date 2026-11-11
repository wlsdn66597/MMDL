"""Resume only the three non-diagnostic seeds with the frozen evaluator."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import run_mmmu_multiseed as original
from mmmu_multiseed_report import SEEDS, aggregate, compare_repeat, read, validate, write
from mmmu_resume_utils import REMAINING, check_manifest, event, repair_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    lock = original.evaluate.run_lock(root / '.campaign.lock')
    status = read(root / 'status.json')
    prior = dict(status)
    status.update(state='running', phase='resume_preflight', driver_pid=os.getpid(), error=None,
                  resume_started_at_utc=original.prepare.now())
    def mark(state=None, phase=None):
        if state:
            status['state'] = state
        if phase:
            status['phase'] = phase
        status['updated_at_utc'] = original.prepare.now()
        write(root / 'status.json', status)
    mark()
    event(root, 'resume_started', previous_state=prior, driver_pid=os.getpid())
    try:
        repeat = root / 'baseline_seed3407_repeat'
        config3407 = read(root / 'protocol/config.seed3407.json')
        expected = {r['id'] for r in read(root / 'protocol/evaluation_items.json')}
        validate(repeat, expected, config3407)
        if not compare_repeat(original.BASELINE, repeat, root / 'repeatability.json')['passed']:
            mark('blocked_reproducibility', 'repeatability_mismatch')
            return
        # Partial result files are never repaired while an evaluator is still running.
        if (root / 'pause.request.json').exists():
            (root / 'pause.request.json').unlink()
        original.preflight(root)
        code = {str(p.relative_to(original.REPO)): original.prepare.digest(p)
                for p in sorted((original.REPO / 'src').glob('*')) if p.is_file()}
        identity = original.evaluate.model_identity(SimpleNamespace(checkpoint=str(original.CHECKPOINT)), config3407)
        status['completed_new_runs'] = ['baseline_seed3407_repeat']
        for seed, name in zip(SEEDS[2:], REMAINING):
            folder = root / name
            config = read(root / f'protocol/config.seed{seed}.json')
            check_manifest(folder, config, identity, code)
            if (folder / 'summary.json').exists():
                validate(folder, expected, config)
                status['completed_new_runs'].append(name)
                event(root, 'skip_completed_seed', run=name)
                continue
            recovered = {name_: repair_jsonl(folder / name_) for name_ in ('predictions.jsonl', 'judge_attempts.jsonl')}
            predictions = original.prepare.read_jsonl(folder / 'predictions.jsonl')
            original.evaluate.check_records(predictions, config, partial=True)
            original.evaluate.saved_judges(predictions, original.prepare.read_jsonl(folder / 'judge_attempts.jsonl'), config)
            event(root, 'resume_seed', run=name, persisted_predictions=len(predictions), recovery=recovered,
                  note='Interrupted item/attempt may be regenerated; no claim of uninterrupted bitwise equivalence')
            status.update(current_run=name, saved_predictions_at_resume=len(predictions), run_started_at_utc=original.prepare.now())
            mark('running', 'evaluate')
            aggregate(root, original.BASELINE, original.SEED42)
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            used, utilization = map(int, gpu.splitlines()[0].split(','))
            if used > 256 or utilization != 0:
                raise RuntimeError('GPU is busy; refusing concurrent evaluation')
            command = [sys.executable, str(original.REPO / 'src/evaluate.py'), '--config', str(root / f'protocol/config.seed{seed}.json'),
                       '--checkpoint', str(original.CHECKPOINT), '--run-name', name, '--output-root', str(root),
                       '--model-cache', str(original.CACHE / 'hub'), '--dataset-cache', str(original.CACHE / 'datasets')]
            start = time.monotonic()
            event(root, 'evaluate_command', run=name, argv=command)
            with (root / f'{name}.log').open('a', encoding='utf-8') as log:
                log.write('\n=== RESUME ' + original.prepare.now() + ' ===\n')
                log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=original.REPO)
            data = validate(folder, expected, config)
            assert original.evaluate.summaries(data['records'], config) == data['summary']
            write(folder / 'verification.json', {'passed': True, 'records': 900, 'all_graded': True,
                  'resumed': bool(predictions), 'saved_predictions_at_resume': len(predictions),
                  'last_segment_elapsed_seconds': time.monotonic()-start, 'finished_at_utc': original.prepare.now()})
            status['completed_new_runs'].append(name)
            mark()
            aggregate(root, original.BASELINE, original.SEED42)
        status['current_run'] = None
        mark('complete', 'completed')
        aggregate(root, original.BASELINE, original.SEED42)
        archive = root.parent / (root.name + '.tar.gz')
        subprocess.run(['tar', '-czf', str(archive), root.name], cwd=root.parent, check=True)
        archive.with_name(archive.name + '.sha256').write_text(original.prepare.digest(archive) + '  ' + archive.name + '\n')
    except BaseException:
        status['error'] = traceback.format_exc()
        mark('paused' if (root / 'pause.request.json').exists() else 'failed')
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
