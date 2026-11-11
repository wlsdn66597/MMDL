"""Read-only paired analysis of saved frozen MMMU runs; never calls a model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from .common import config_digest, load_config, resolve_path, sha256_file, write_json
from .evaluate import SEEDS, SOURCE_HASHES, VERSIONS, read_json, validate_source, validate_training_pair

HISTORICAL_RESULTS = {
    3407: '3791f54f30b9e127fa995f2f442ab0460c988addb069377596ea5830a7a8f0d8',
    42: '7db40fbd31ad7a6646d061bccc3c755d1f08030485f3d0543582611b9ac4290e',
    123: '951dbb2fedb845141d37c5ac4de577a40b3ca620128bab6b019629dbd211aa15',
    2026: '579a0a819145bec8e08667ff2eff31fadb6da9d2d9e284ba10d9676a03cba8db',
    9876: '9d4af522d731d8afff318a865c5edd03d5dd0dfa28d0f39da173462f6b59484d',
}
BASE_WEIGHTS = {
    'model-00001-of-00002.safetensors': '30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9',
    'model-00002-of-00002.safetensors': '046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6',
}
INPUT_FIELDS = ('id', 'subject', 'dataset_config', 'dataset_index', 'question_type',
                'question', 'options', 'reference_answer', 'image_fields',
                'original_image_sizes_wh', 'input_prompt', 'rendered_chat')
VOLATILE = {'elapsed_seconds', 'created_at', 'timestamp', 'started_at', 'completed_at'}


def scrub_timing(value):
    if isinstance(value, dict):
        return {k: scrub_timing(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [scrub_timing(v) for v in value]
    return value


def validate_records(records, expected_ids=None):
    ids = [r['id'] for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate evaluation sample IDs')
    if expected_ids is not None and ids != list(expected_ids):
        raise ValueError('Evaluation sample IDs/order differ from frozen items')
    for r in records:
        final = r['grading']['final']
        response = r['model_response']
        if final.get('status') != 'graded' or final.get('hit') not in (0, 1):
            raise ValueError(f'Incomplete grading: {r["id"]}')
        if response.get('finish_reason') not in ('stop', 'length'):
            raise ValueError(f'Unsupported termination: {r["id"]}')
        if response['finish_reason'] == 'length' and final['hit'] != 0:
            raise ValueError('Length-limited responses must remain incorrect')
        if not isinstance(response.get('raw'), str) or response.get('output_tokens', -1) < 0:
            raise ValueError('Missing raw response or output token count')
    return {r['id']: r for r in records}


def input_signature(row):
    return {k: row.get(k) for k in INPUT_FIELDS}


def load_run(path, seed=None, expected_config=None, expected_ids=None, arm=None):
    path = Path(path)
    result_path = path / 'results.json'
    result = read_json(result_path)
    if result.get('schema_version') != 'mmmu-frozen-evaluation-v1':
        raise ValueError(f'Unsupported result schema: {path}')
    manifest = read_json(path / 'manifest.json')
    if result['manifest'] != manifest:
        raise ValueError('Embedded result manifest differs from manifest.json')
    if manifest.get('mode') != 'inference':
        raise ValueError('Saved-response regrading cannot be reused as fresh inference')
    config = manifest['configuration']
    if seed is not None and config['sampling']['seed'] != seed:
        raise ValueError('Result seed differs from requested seed')
    if expected_config is not None and config != expected_config:
        raise ValueError('Evaluation configuration drift; paired comparison refused')
    environment = manifest['environment']
    if environment.get('versions') != VERSIONS or environment.get('torch_cuda') != '12.8' or environment.get('gpu') != 'NVIDIA GeForce RTX 4090':
        raise ValueError('Evaluation runtime differs from executed baseline')
    if not environment.get('python', '').startswith('3.12.') or not environment.get('platform', '').startswith('Linux-'):
        raise ValueError('Evaluation Python/platform differs from baseline protocol')
    checksum = sha256_file(result_path)
    historical = arm == 'baseline' and checksum == HISTORICAL_RESULTS.get(seed)
    if arm in ('plain', 'era') and checksum in HISTORICAL_RESULTS.values():
        raise ValueError('An untrained baseline cannot substitute for a trained arm')
    launcher = None
    # Historical outputs are anchored by exact audited artifact hashes. New runs
    # require the launcher's environment and fresh-run receipt in addition.
    if checksum not in HISTORICAL_RESULTS.values():
        launcher = read_json(path / 'launcher.json')
        flags = launcher.get('vllm_environment', {})
        if not launcher.get('fresh_output_verified') or flags.get('VLLM_ENABLE_V1_MULTIPROCESSING') != '0' or 'VLLM_BATCH_INVARIANT' in flags:
            raise ValueError('Missing or incompatible frozen evaluation launch provenance')
        if launcher.get('python_probe') != {'python': [3, 12], 'versions': VERSIONS}:
            raise ValueError('Launcher dependency probe differs from baseline')
        identity = manifest['model_identity']
        if not identity.get('weights_sha256') or any(launcher.get('checkpoint_sha256', {}).get(k) != v for k, v in identity['weights_sha256'].items()):
            raise ValueError('Evaluated checkpoint hashes differ from preflight')
    code = manifest['code_sha256']
    for rel, digest in SOURCE_HASHES.items():
        if code.get(rel) != digest:
            if not (historical and seed == 42 and rel == 'src/evaluate.py'
                    and code.get(rel) == '4fab4285cac36abb651e70a494f0ae7cd90bf0230e8894e1cb54fce0c98e22ca'):
                raise ValueError(f'Evaluation source drift: {rel}')
    if arm == 'baseline' and not historical:
        identity = manifest['model_identity']
        if identity.get('weights_sha256') != BASE_WEIGHTS:
            raise ValueError('Baseline reuse needs audited historical result hash or exact original weight hashes')
    records = result['records']
    indexed = validate_records(records, expected_ids)
    summary = read_json(path / 'summary.json')
    correct = sum(r['grading']['final']['hit'] for r in records)
    if result['summary'] != summary or summary['total'] != len(records) or summary['correct'] != correct:
        raise ValueError('Saved summary does not agree with all result records')
    if not math.isclose(summary['accuracy_percent'], 100 * correct / len(records), abs_tol=1e-9):
        raise ValueError('Saved accuracy does not agree with result records')
    predictions = [json.loads(line) for line in (path / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [r['id'] for r in predictions] != list(indexed):
        raise ValueError('Prediction and result sample IDs/order differ')
    for predicted, record in zip(predictions, records):
        if input_signature(predicted) != input_signature(record) or predicted['model_response'] != record['model_response']:
            raise ValueError('Prediction and result inputs/responses differ')
    judge_file = path / 'judge_attempts.jsonl'
    judge_ids = {json.loads(line)['id'] for line in judge_file.read_text(encoding='utf-8').splitlines() if line.strip()} if judge_file.exists() else set()
    needed = {r['id'] for r in records if 'judge' in r['grading']['final'].get('method', '')}
    if not needed <= judge_ids or not judge_ids <= set(indexed):
        raise ValueError('Missing or foreign judge attempt records')
    artifacts = {p.name: sha256_file(p) for p in path.iterdir()
                 if p.is_file() and p.name in ('results.json', 'summary.json', 'manifest.json',
                                              'predictions.jsonl', 'judge_attempts.jsonl', 'launcher.json')}
    return {'path': str(path), 'manifest': manifest, 'records': records,
            'artifacts_sha256': artifacts, 'historical_baseline_verified': historical,
            'training_provenance': launcher.get('training_provenance') if launcher else None}


def validate_pair(left, right):
    a = validate_records(left)
    b = validate_records(right)
    if list(a) != list(b):
        raise ValueError('Paired runs have mismatched sample IDs/order')
    for key in a:
        if input_signature(a[key]) != input_signature(b[key]):
            raise ValueError(f'Paired input/prompt drift: {key}')
    return a, b


def repetition_metrics(raw):
    # Fixed mechanical diagnostics, not a semantic judge or a claim of reasoning quality.
    words = re.findall(r'\w+', raw.casefold())
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    repeated = len(grams) - len(set(grams))
    lines = [re.sub(r'\s+', ' ', line).strip().casefold() for line in raw.splitlines()]
    lines = [line for line in lines if len(line.split()) >= 6]
    counts = Counter(lines)
    return {'repeated_4gram_fraction': repeated / len(grams) if grams else 0.,
            'has_line_repeated_3_times': any(n >= 3 for n in counts.values())}


def arm_metrics(records):
    validate_records(records)
    tokens = sorted(r['model_response']['output_tokens'] for r in records)
    repetitions = [repetition_metrics(r['model_response']['raw']) for r in records]
    # These headings match the target schema. They only measure visible format adoption.
    era_pattern = re.compile(r'(?ims)^\s*Evidence\s*:.*?^\s*Rule\s*:.*?^\s*Application\s*:')
    n = len(records)
    length = sum(r['model_response']['finish_reason'] == 'length' for r in records)
    explicit = sum(r['grading']['final'].get('method') == 'explicit_final_parser' for r in records)
    judged = sum('judge' in r['grading']['final'].get('method', '') for r in records)
    return {'total': n, 'correct': sum(r['grading']['final']['hit'] for r in records),
        'accuracy_percent': 100 * sum(r['grading']['final']['hit'] for r in records) / n,
        'length_count': length, 'length_rate': length / n,
        'normal_stop_count': n - length, 'normal_stop_rate': (n - length) / n,
        'judge_count': judged, 'judge_rate': judged / n,
        'generation_seconds': sum(r['model_response'].get('elapsed_seconds', 0.) for r in records),
        'mean_output_tokens': statistics.mean(tokens), 'median_output_tokens': statistics.median(tokens),
        'p95_output_tokens': tokens[math.ceil(.95 * n) - 1],
        'explicit_final_parser_count': explicit, 'explicit_final_parser_rate': explicit / n,
        'era_heading_count': sum(bool(era_pattern.search(r['model_response']['raw'])) for r in records),
        'mean_repeated_4gram_fraction': statistics.mean(r['repeated_4gram_fraction'] for r in repetitions),
        'repeated_line_3plus_count': sum(r['has_line_repeated_3_times'] for r in repetitions)}


def compare_records(left, right):
    a, b = validate_pair(left, right)
    changes = {'wrong_to_right': [], 'right_to_wrong': [], 'right_to_right': [], 'wrong_to_wrong': []}
    for key in a:
        before, after = a[key]['grading']['final']['hit'], b[key]['grading']['final']['hit']
        changes[('right' if before else 'wrong') + '_to_' + ('right' if after else 'wrong')].append(key)
    am, bm = arm_metrics(left), arm_metrics(right)
    subjects = {}
    for subject in sorted({r['subject'] for r in left}):
        keys = [key for key in a if a[key]['subject'] == subject]
        before = sum(a[key]['grading']['final']['hit'] for key in keys)
        after = sum(b[key]['grading']['final']['hit'] for key in keys)
        subjects[subject] = {'total': len(keys), 'left_correct': before, 'right_correct': after,
            'accuracy_delta_pp': 100 * (after - before) / len(keys),
            'wrong_to_right': sum(not a[k]['grading']['final']['hit'] and b[k]['grading']['final']['hit'] for k in keys),
            'right_to_wrong': sum(a[k]['grading']['final']['hit'] and not b[k]['grading']['final']['hit'] for k in keys)}
    return {'left': am, 'right': bm, 'accuracy_delta_pp': bm['accuracy_percent'] - am['accuracy_percent'],
            'transition_counts': {k: len(v) for k, v in changes.items()}, 'transition_ids': changes,
            'by_subject': subjects}


def compare_repeats(first, second):
    a, b = load_run(first), load_run(second)
    validate_pair(a['records'], b['records'])
    differences = []
    for key in ('configuration', 'model_identity', 'code_sha256', 'environment'):
        if a['manifest'][key] != b['manifest'][key]:
            differences.append('manifest.' + key)
    for x, y in zip(a['records'], b['records']):
        for field in ('model_response', 'grading'):
            if scrub_timing(x[field]) != scrub_timing(y[field]):
                differences.append(x['id'] + '.' + field)
    judges = []
    for path in (first, second):
        file = Path(path) / 'judge_attempts.jsonl'
        judges.append([scrub_timing(json.loads(line)) for line in file.read_text(encoding='utf-8').splitlines() if line.strip()] if file.exists() else [])
    if judges[0] != judges[1]:
        differences.append('judge_attempts')
    return {'passed': not differences, 'records_compared': len(a['records']),
        'differences': differences, 'ignored_fields': sorted(VOLATILE),
        'first_artifacts_sha256': a['artifacts_sha256'], 'second_artifacts_sha256': b['artifacts_sha256']}


def require_seed_sets(runs, seeds=SEEDS):
    if set(runs) != {'baseline', 'plain', 'era'}:
        raise ValueError('Comparison requires baseline, plain and era arms')
    expected = {str(s) for s in seeds}
    for arm, entries in runs.items():
        if {str(s) for s in entries} != expected:
            raise ValueError(f'Missing/unexpected inference seeds for {arm}')


def compare_campaign(cfg, runs):
    _, protocol, configs, _ = validate_source(cfg)
    require_seed_sets(runs)
    frozen = read_json(protocol / 'evaluation_items.json')
    ids = [row['id'] for row in frozen]
    loaded, hashes = {}, {}
    for arm in ('baseline', 'plain', 'era'):
        loaded[arm], hashes[arm] = {}, {}
        for seed in SEEDS:
            item = load_run(runs[arm][str(seed)], seed, configs[seed], ids, arm)
            for record, expected in zip(item['records'], frozen):
                payload = {k: record[k] for k in ('question', 'options', 'reference_answer', 'input_prompt', 'image_fields')}
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if digest != expected['input_sha256']:
                    raise ValueError(f'Input differs from frozen baseline: {record["id"]}')
            loaded[arm][seed] = item
            hashes[arm][str(seed)] = {'path': item['path'], 'artifacts_sha256': item['artifacts_sha256'],
                'historical_baseline_verified': item['historical_baseline_verified']}
    # A single checkpoint must underlie every seed of each trained arm.
    for arm in ('plain', 'era'):
        identities = [loaded[arm][s]['manifest']['model_identity'] for s in SEEDS]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError(f'{arm} uses different checkpoints across inference seeds')
        lineages = [loaded[arm][s]['training_provenance'] for s in SEEDS]
        if not lineages[0] or any(proof != lineages[0] for proof in lineages[1:]):
            raise ValueError(f'{arm} has missing/different training provenance across seeds')
        if lineages[0]['identity']['config_sha256'] != config_digest(cfg):
            raise ValueError(f'{arm} was trained using a different experiment configuration')
    validate_training_pair({arm: loaded[arm][3407]['training_provenance'] for arm in ('plain', 'era')})
    result = {}
    for left, right in (('baseline', 'plain'), ('baseline', 'era'), ('plain', 'era')):
        by_seed = {str(s): compare_records(loaded[left][s]['records'], loaded[right][s]['records']) for s in SEEDS}
        deltas = [by_seed[str(s)]['accuracy_delta_pp'] for s in SEEDS]
        result[f'{right}_minus_{left}'] = {'by_seed': by_seed,
            'mean_accuracy_delta_pp': statistics.mean(deltas), 'sample_sd_delta_pp': statistics.stdev(deltas)}
    return {'schema_version': 1, 'model_calls': 0, 'seeds': SEEDS, 'comparisons': result,
        'artifact_provenance': hashes,
        'limitations': ['Five runs reuse the same 900 items; they are not 4500 independent questions.',
            'Inference-seed variability does not measure training-seed variability.',
            'ERA headings and repetition metrics are mechanical diagnostics, not semantic proof of ERA reasoning.',
            'Observed saved artifact parity does not establish deterministic behavior on another GPU/software environment.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--runs', help='JSON mapping arm -> seed string -> saved run directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeatability', nargs=2, metavar=('FIRST', 'SECOND'))
    args = parser.parse_args(argv)
    if args.repeatability:
        input_paths = list(args.repeatability)
        report = compare_repeats(*args.repeatability)
    else:
        if not args.config:
            parser.error('--config is required for campaign comparison')
        cfg = load_config(args.config)
        if args.runs:
            run_file = Path(args.runs).resolve()
            runs = read_json(run_file)
            runs = {arm: {str(seed): str((run_file.parent / path).resolve()) for seed, path in entries.items()}
                    for arm, entries in runs.items()}
        else:
            e = cfg['evaluation']
            root = Path(resolve_path(cfg, e['output_root']))
            runs = {'baseline': {str(s): str(resolve_path(cfg, e['baseline_runs'][str(s)])) for s in SEEDS}}
            runs.update({arm: {str(s): str(root / f'{arm}_seed{s}') for s in SEEDS} for arm in ('plain', 'era')})
        input_paths = [path for entries in runs.values() for path in entries.values()]
        report = compare_campaign(cfg, runs)
    output = Path(args.output).resolve()
    if any(output.is_relative_to(Path(path).resolve()) for path in input_paths):
        raise ValueError('Comparison output must be outside the read-only input run directories')
    write_json(Path(args.output), report)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'model_calls': 0,
                      'passed': report.get('passed')}, indent=2))
    if report.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
