"""Frozen-baseline evaluation of epoch 2/3/4; no training or source mutation.

This is a checkpoint comparison, not the separate paired plain/ERA experiment.
All five baseline seeds, a fresh baseline replay, and candidate repeats are used.
Semantic review remains an explicitly pending, image-grounded analysis task.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from era_lora.common import load_config, write_json, sha256_file
from era_lora.evaluate import validate_source, checkpoint_preflight, VERSIONS, SEEDS
from era_lora.compare import load_run, validate_pair, compare_records, compare_repeats, arm_metrics
from epoch_format_campaign import validate_export

OUT = ROOT / 'runs/mmmu900_epochs_20261031'
CONFIG = ROOT / 'configs/era_4epoch_20261030.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(name, value):
    """Atomic status writes let completion monitors avoid partial JSON."""
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def format_metrics(row):
    raw = row['model_response']['raw']
    legacy = bool(re.search(r'(?ms)^Evidence:\s*\S.*?^Rule:\s*\S.*?^Application:\s*\S', raw))
    headings = list(re.finditer(r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?(Evidence|Rule|Application)(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?', raw))
    names = [m[1].lower() for m in headings]
    ordered = names == ['evidence', 'rule', 'application']
    final = re.search(r'(?im)^[ \t]*(?:\*\*)?Final answer[ \t]*:[ \t]*(?:\*\*)?([^\n]+)[ \t]*\s*\Z', raw)
    nonempty = False
    if ordered:
        ends = [headings[1].start(), headings[2].start(), final.start() if final else len(raw)]
        nonempty = all(bool(re.search(r'\w', raw[m.end():end])) for m, end in zip(headings, ends))
    valid_final = bool(final and re.search(r'\w', final[1]) and (not headings or final.start() > headings[-1].end()))
    complete = ordered and nonempty and valid_final and row['model_response']['finish_reason'] == 'stop'
    return {'legacy_strict_headings': legacy, 'ordered_unique_headers': ordered,
            'nonempty_sections': nonempty, 'terminal_final_answer': valid_final,
            'complete_era_format': complete}


def metrics(records):
    result = arm_metrics(records)
    flags = [format_metrics(r) for r in records]
    for key in flags[0]:
        result[key + '_count'] = sum(f[key] for f in flags)
        result[key + '_percent'] = 100 * result[key + '_count'] / len(records)
    result['by_subject'] = {}
    for subject in sorted({r['subject'] for r in records}):
        items = [r for r in records if r['subject'] == subject]
        result['by_subject'][subject] = {'items': len(items), 'correct': sum(r['grading']['final']['hit'] for r in items),
            'format_success': sum(format_metrics(r)['complete_era_format'] for r in items)}
    return result


def preflight():
    cfg = load_config(CONFIG)
    source, protocol, configs, source_hashes = validate_source(cfg)
    frozen = read(protocol / 'evaluation_items.json')
    ids = [r['id'] for r in frozen]
    baselines = {}
    for seed in SEEDS:
        baseline = load_run(cfg['evaluation']['baseline_runs'][str(seed)], seed, configs[seed], ids, 'baseline')
        for record, expected in zip(baseline['records'], frozen):
            payload = {k: record[k] for k in ('question','options','reference_answer','input_prompt','image_fields')}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            assert digest == expected['input_sha256'], record['id']
        baselines[str(seed)] = {'path': baseline['path'], 'hashes': baseline['artifacts_sha256']}
    proofs = {str(epoch): validate_export(cfg, epoch) for epoch in (2, 3, 4)}
    base_hashes = checkpoint_preflight(cfg['model']['local_path'], allow_baseline_snapshot=True)
    probe_code = "import sys,json,importlib.metadata as m; print(json.dumps({'python':list(sys.version_info[:2]),'versions':{n:m.version(n).split('+')[0] for n in " + repr(list(VERSIONS)) + "}}))"
    probe = json.loads(subprocess.check_output([cfg['evaluation']['python'], '-c', probe_code], text=True))
    assert probe == {'python': [3, 12], 'versions': VERSIONS}
    excluded = set(read(ROOT / 'configs/era_diagnostic_ids.json'))
    selected = []
    for subject in configs[3407]['dataset']['configs']:
        choices = [r['id'] for r in frozen if r['subject'] == subject and r['id'] not in excluded]
        selected.append(min(choices, key=lambda x: hashlib.sha256(('era-review-20261031:' + x).encode()).hexdigest()))
    plan = {'kind': 'era_epoch_checkpoint_comparison', 'training': False,
        'seeds': SEEDS, 'epochs': [2, 3, 4], 'items_per_run': 900,
        'primary_generations': 13500, 'reproducibility_generations': 3600,
        'source_hashes': source_hashes, 'baseline_runs': baselines, 'exports': proofs,
        'base_hashes': base_hashes, 'python_probe': probe,
        'script_sha256': sha256_file(Path(__file__)),
        'semantic_sample': {'ids': selected, 'seed': 3407, 'one_per_subject': True,
            'selection_before_new_outputs': True, 'previous_eight_excluded': True,
            'review_status': 'pending', 'human_adjudication': False},
        'format_definition': 'unique ordered E/R/A headings, nonempty sections, terminal Final answer line, normal stop',
        'legacy_format_also_reported': True,
        'scope_note': 'Plain LoRA is not trained; this does not establish ERA versus paired plain SFT benefit.'}
    return cfg, source, protocol, configs, ids, plan


def run_one(cfg, source, protocol, configs, ids, plan, name, seed, epoch=None):
    destination = OUT / name
    checkpoint = cfg['model']['local_path'] if epoch is None else plan['exports'][str(epoch)]['model']
    hashes = plan['base_hashes'] if epoch is None else plan['exports'][str(epoch)]['hashes']
    command = [cfg['evaluation']['python'], str(source / 'src/evaluate.py'), '--config', str(protocol / f'config.seed{seed}.json'),
        '--run-name', name, '--output-root', str(OUT), '--checkpoint', checkpoint,
        '--model-cache', cfg['evaluation']['model_cache'], '--dataset-cache', cfg['evaluation']['dataset_cache']]
    launcher = {'argv': command, 'fresh_output_verified': True, 'python_probe': plan['python_probe'],
        'checkpoint_sha256': hashes, 'vllm_environment': {'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'},
        'source_hashes': plan['source_hashes'], 'training_provenance': plan['exports'].get(str(epoch)),
        'campaign_script_sha256': plan['script_sha256']}
    if destination.exists():
        assert read(destination / 'launcher.json') == launcher, 'Foreign or changed run directory'
    else:
        destination.mkdir()
        write_json(destination / 'launcher.json', launcher)
    save('status.json', {'state': 'running', 'stage': name, 'semantic_review': 'pending'})
    if not (destination / 'results.json').exists():
        env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env.pop('VLLM_BATCH_INVARIANT', None)
        with (OUT / 'logs' / (name + '.log')).open('a') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    loaded = load_run(destination, seed, configs[seed], ids, 'baseline' if epoch is None else 'era')
    reference = load_run(cfg['evaluation']['baseline_runs'][str(seed)], seed, configs[seed], ids, 'baseline')
    validate_pair(reference['records'], loaded['records'])
    # Processor input token counts/grids must agree as well as text/image metadata.
    for a, b in zip(reference['records'], loaded['records']):
        for field in ('input_tokens', 'image_grid_thw', 'resized_image_sizes_wh'):
            if field in a:
                assert a[field] == b[field], (a['id'], field)
    save(name + '/format_metrics.json', metrics(loaded['records']))
    return loaded


def summarize(cfg, configs, ids, plan):
    summaries, runs, pairs = {}, {}, {}
    for arm in ('baseline', 'epoch_2', 'epoch_3', 'epoch_4'):
        summaries[arm], runs[arm] = {}, {}
        for seed in SEEDS:
            path = cfg['evaluation']['baseline_runs'][str(seed)] if arm == 'baseline' else OUT / f'{arm}_seed{seed}'
            if not (Path(path) / 'results.json').exists():
                continue
            run = load_run(path, seed, configs[seed], ids, 'baseline' if arm == 'baseline' else 'era')
            runs[arm][seed] = run
            summaries[arm][str(seed)] = metrics(run['records'])
            if arm != 'baseline':
                pairs[f'baseline_to_{arm}_seed{seed}'] = compare_records(runs['baseline'][seed]['records'], run['records'])
    for left, right in (('epoch_2', 'epoch_3'), ('epoch_2', 'epoch_4'), ('epoch_3', 'epoch_4')):
        for seed in set(runs[left]) & set(runs[right]):
            pairs[f'{left}_to_{right}_seed{seed}'] = compare_records(runs[left][seed]['records'], runs[right][seed]['records'])
    aggregate = {}
    for arm, per_seed in summaries.items():
        if len(per_seed) == 5:
            aggregate[arm] = {key: {'mean': statistics.mean(m[key] for m in per_seed.values()),
                'sample_sd': statistics.stdev(m[key] for m in per_seed.values()),
                'min': min(m[key] for m in per_seed.values()), 'max': max(m[key] for m in per_seed.values())}
                for key in ('accuracy_percent', 'complete_era_format_percent', 'legacy_strict_headings_percent')}
    save('comparison.json', {'per_seed': summaries, 'aggregate': aggregate, 'paired': pairs,
        'unique_items': 900, 'note': 'Five seeds are repeated measurements, not 4500 independent questions.'})
    chosen = set(plan['semantic_sample']['ids'])
    review = []
    for arm, per_seed in runs.items():
        if 3407 in per_seed:
            for row in per_seed[3407]['records']:
                if row['id'] in chosen:
                    review.append({'arm': arm, 'seed': 3407, 'record': row,
                        'format': format_metrics(row), 'E': None, 'R': None, 'A': None, 'review_status': 'pending'})
    save('semantic_review/responses.json', review)
    lines = ['# MMMU 900 epoch comparison', '', 'Automatic accuracy/format scores only; E/R/A semantic review is pending.', '',
        '| Model | Seed | Correct / 900 | Accuracy | ERA complete | Legacy headers |', '|---|---:|---:|---:|---:|---:|']
    for arm, per_seed in summaries.items():
        for seed, m in per_seed.items():
            lines.append(f"| {arm} | {seed} | {m['correct']} | {m['accuracy_percent']:.2f}% | {m['complete_era_format_count']}/900 | {m['legacy_strict_headings_count']}/900 |")
    (OUT / 'scores.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def campaign():
    cfg, source, protocol, configs, ids, plan = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (OUT / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'plan.json').exists():
            assert read(OUT / 'plan.json') == plan, 'Campaign inputs/code changed'
        else:
            save('plan.json', plan)
        (OUT / 'logs').mkdir(exist_ok=True)
        summarize(cfg, configs, ids, plan)
        run_one(cfg, source, protocol, configs, ids, plan, 'baseline_seed3407_replay', 3407)
        check = compare_repeats(cfg['evaluation']['baseline_runs']['3407'], OUT / 'baseline_seed3407_replay')
        save('baseline_replay_receipt.json', check)
        assert check['passed'], 'Baseline replay mismatch; see receipt before continuing'
        # All three primary seed outputs become available before secondary seeds.
        for epoch in (2, 3, 4):
            for suffix in ('', '_repeat'):
                run_one(cfg, source, protocol, configs, ids, plan, f'epoch_{epoch}_seed3407{suffix}', 3407, epoch)
            check = compare_repeats(OUT / f'epoch_{epoch}_seed3407', OUT / f'epoch_{epoch}_seed3407_repeat')
            save(f'epoch_{epoch}_repeatability.json', check)
            assert check['passed'], f'Epoch {epoch} repeatability mismatch'
            summarize(cfg, configs, ids, plan)
        for seed in SEEDS[1:]:
            for epoch in (2, 3, 4):
                run_one(cfg, source, protocol, configs, ids, plan, f'epoch_{epoch}_seed{seed}', seed, epoch)
                summarize(cfg, configs, ids, plan)
        # Recheck frozen archives, exports, and original baseline hashes after execution.
        assert preflight()[-1] == plan
        save('completed.json', {'state': 'inference_and_scoring_complete', 'primary_runs': 15,
            'primary_generations': 13500, 'reproducibility_runs': 4, 'original_artifacts_unchanged': True,
            'semantic_review': 'pending', 'semantic_sample_questions': 30})
        save('status.json', read(OUT / 'completed.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        plan = preflight()[-1]
        print(json.dumps({'preflight': 'passed', 'seeds': plan['seeds'], 'epochs': plan['epochs'],
            'primary_generations': plan['primary_generations'], 'semantic_ids': plan['semantic_sample']['ids']}))
        return
    try:
        campaign()
    except Exception:
        save('failed.json', {'state': 'failed', 'traceback': traceback.format_exc()})
        save('status.json', read(OUT / 'failed.json'))
        raise


if __name__ == '__main__':
    main()
