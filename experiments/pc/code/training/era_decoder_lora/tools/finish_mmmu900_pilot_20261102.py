"""Finish the in-flight epoch4 repeat; never launch another inference seed.

The original controller is SIGSTOP'd by the explicit user scope change.
This separate CPU-only finisher preserves the original code/plan/artifacts.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from era_lora.common import load_config, sha256_file, write_json
from era_lora.evaluate import validate_source
from era_lora.compare import load_run, compare_repeats, compare_records, validate_pair, repetition_metrics
from epoch_format_campaign import validate_export
from mmmu900_epoch_campaign import metrics, format_metrics

OUT = ROOT / 'runs/mmmu900_epochs_20261031'


def read(path):
    return json.loads(Path(path).read_text())


def save(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    write_json(tmp, data)
    tmp.replace(path)


def state(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.read_text().split(') ', 1)[1].split()[0] if path.exists() else None


def main():
    import fcntl
    with (OUT / '.pilot_finish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scope = read(OUT / 'scope_change_20261102.json')
        controller, evaluator = scope['controller_pid'], scope['preserved_evaluator_pid']
        assert state(controller) == 'T', 'Queue controller must remain stopped'
        assert not any(OUT.glob('epoch_*_seed42*')), 'Unexpected secondary seed output'
        save('pilot_status.json', {'state': 'waiting_for_epoch4_repeat', 'cancelled_seeds': scope['cancelled_seeds'],
            'semantic_review': 'pending', 'finisher_sha256': sha256_file(Path(__file__))})
        # This detached tmux helper only awaits process exit; no response-count polling.
        while state(evaluator) not in (None, 'Z'):
            assert state(controller) == 'T', 'Stopped controller was unexpectedly resumed'
            time.sleep(15)
        repeat = OUT / 'epoch_4_seed3407_repeat'
        assert (repeat / 'results.json').exists(), 'Repeat evaluator exited without final grading'
        save('pilot_status.json', {'state': 'verifying_completed_seed3407', 'semantic_review': 'pending'})
        cfg = load_config(ROOT / 'configs/era_4epoch_20261030.json')
        source, protocol, configs, hashes = validate_source(cfg)
        plan = read(OUT / 'plan.json')
        assert hashes == plan['source_hashes']
        assert sha256_file(ROOT / 'tools/mmmu900_epoch_campaign.py') == plan['script_sha256']
        ids = [x['id'] for x in read(protocol / 'evaluation_items.json')]
        base = load_run(cfg['evaluation']['baseline_runs']['3407'], 3407, configs[3407], ids, 'baseline')
        assert base['artifacts_sha256'] == plan['baseline_runs']['3407']['hashes']
        baseline_check = compare_repeats(base['path'], OUT / 'baseline_seed3407_replay')
        assert baseline_check['passed'], baseline_check
        loaded = {'baseline': base}
        receipts = {'baseline': baseline_check}
        for epoch in (2, 3, 4):
            proof = validate_export(cfg, epoch)
            assert proof == plan['exports'][str(epoch)]
            name = f'epoch_{epoch}'
            current = load_run(OUT / f'{name}_seed3407', 3407, configs[3407], ids, 'era')
            again = load_run(OUT / f'{name}_seed3407_repeat', 3407, configs[3407], ids, 'era')
            validate_pair(base['records'], current['records'])
            check = compare_repeats(current['path'], again['path'])
            save(f'pilot/epoch_{epoch}_repeatability.json', check)
            assert check['passed'], f'Epoch {epoch} repeat mismatch'
            loaded[name] = current
            receipts[name] = check
        summary = {'scope': 'single_seed_pilot', 'seed': 3407, 'unique_questions': 900,
            'candidate_runs': 3, 'candidate_responses': 2700, 'seeds_cancelled': scope['cancelled_seeds'],
            'metrics': {name: metrics(run['records']) for name, run in loaded.items()},
            'paired': {}, 'semantic_review': 'pending', 'inference_and_training_launched_by_finisher': False}
        for left, right in [('baseline', 'epoch_2'), ('baseline', 'epoch_3'), ('baseline', 'epoch_4'),
                            ('epoch_2', 'epoch_3'), ('epoch_2', 'epoch_4'), ('epoch_3', 'epoch_4')]:
            summary['paired'][left + '_to_' + right] = compare_records(loaded[left]['records'], loaded[right]['records'])
        save('pilot/summary.json', summary)
        # Full 900-item response inventory, not an invented semantic grade.
        records = []
        for index, question_id in enumerate(ids):
            row = {'id': question_id, 'subject': base['records'][index]['subject'], 'models': {}}
            for name, run in loaded.items():
                record = run['records'][index]
                assert record['id'] == question_id
                row['models'][name] = {'record': record, 'format': format_metrics(record),
                    'repetition': repetition_metrics(record['model_response']['raw']),
                    'E': None, 'R': None, 'A': None, 'semantic_review_status': 'not_reviewed'}
            records.append(row)
        save('pilot/all900_response_inventory.json', records)
        selected = set(plan['semantic_sample']['ids'])
        save('pilot/prespecified30_responses.json', [r for r in records if r['id'] in selected])
        save('pilot/verification.json', {'repeatability': receipts,
            'artifacts': {name: run['artifacts_sha256'] for name, run in loaded.items()},
            'original_plan_sha256': sha256_file(OUT / 'plan.json'),
            'scope_change_sha256': sha256_file(OUT / 'scope_change_20261102.json'),
            'all_900_ids_and_paired_inputs_validated': True, 'original_artifacts_unchanged': True})
        # The child has exited. Only the exact stopped queue controller is retired.
        assert state(controller) == 'T'
        cmd = Path(f'/proc/{controller}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert cmd == scope['controller_cmd']
        assert Path(f'/proc/{controller}/cwd').resolve() == ROOT
        os.kill(controller, signal.SIGTERM)
        os.kill(controller, signal.SIGCONT)  # Deliver the pending TERM, not resume work.
        for _ in range(20):
            if state(controller) in (None, 'Z'):
                break
            time.sleep(.25)
        assert state(controller) in (None, 'Z'), 'Queue controller did not exit'
        assert not any(OUT.glob('epoch_*_seed42*'))
        complete = {'state': 'seed3407_inference_and_scoring_complete', 'semantic_review': 'pending',
            'seed': 3407, 'primary_runs': 3, 'primary_responses': 2700,
            'baseline_replay_and_all_three_repeats_passed': True,
            'cancelled_seeds': scope['cancelled_seeds'], 'controller_terminated_by_user_scope_change': True,
            'note': 'Original campaign.exit may be143 for intentional queue termination; use this receipt for pilot completion.'}
        save('pilot_completed.json', complete)
        save('pilot_status.json', complete)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('pilot_failed.json', {'state': 'failed', 'traceback': traceback.format_exc()})
        raise
