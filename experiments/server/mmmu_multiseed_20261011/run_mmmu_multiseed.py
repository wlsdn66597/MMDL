"""Continue the frozen baseline campaign with a strict 3407 repeatability gate."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from mmmu_multiseed_report import SEEDS, aggregate, compare_repeat, read, validate, write

REPO = Path('/home/rlatjswo/mmdl_test')
OLD = Path('/home/rlatjswo/mmmu_baseline_20261009')
VISUAL = Path('/home/rlatjswo/mmmu_visual_tokens_seed42_20261010')
BASELINE = OLD / 'baseline_seed3407'
SEED42 = VISUAL / 'baseline_seed42'
CACHE = Path('/home/rlatjswo/mmdl-cache/huggingface')
CHECKPOINT = CACHE / 'hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'
sys.path.insert(0, str(REPO / 'src'))
import evaluate
import prepare


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(root):
    config = read(root / 'protocol/config.seed3407.json')
    contract = read(root / 'protocol/evaluation_contract.json')
    assert config == read(OLD / 'protocol/config.seed3407.json')
    assert contract == read(OLD / 'protocol/evaluation_contract.json')
    for seed in SEEDS:
        candidate = read(root / f'protocol/config.seed{seed}.json')
        expected = copy.deepcopy(config)
        expected['sampling']['seed'] = seed
        assert candidate == expected and candidate['judge']['sampling']['seed'] == 3407
        assert digest(root / f'protocol/config.seed{seed}.json') == contract['config_sha256'][f'config.seed{seed}.json']
    assert digest(root / 'protocol/evaluation_items.json') == contract['evaluation_items_sha256']
    hashes = read(OLD / 'protocol/linux_source_hashes.json')['sha256']
    for name, sha in hashes.items():
        if name.startswith('src/') or name == 'requirements.txt':
            assert digest(REPO / name) == sha, name
    visual_repo = Path('/home/rlatjswo/mmdl_visual_token_analysis')
    for name, sha in hashes.items():
        if not (name.startswith('src/') or name == 'requirements.txt'):
            continue
        content = (visual_repo / name).read_bytes().replace(b'\r\n', b'\n')
        if name == 'src/evaluate.py':
            lines = content.decode().splitlines(keepends=True)
            start = next(i for i, line in enumerate(lines) if "visual_replay={'prompt_token_ids'" in line)
            assert "'source':'vllm_exact_token_ids'}," in lines[start+4]
            content = ''.join(lines[:start] + lines[start+5:]).encode()
        assert hashlib.sha256(content).hexdigest() == sha, name
    old_manifest = read(BASELINE / 'manifest.json')
    visual_manifest = read(SEED42 / 'manifest.json')
    for folder, manifest in ((REPO, old_manifest), (visual_repo, visual_manifest)):
        for name, sha in manifest['code_sha256'].items():
            assert digest(folder / name) == sha, (str(folder), name)
    environment = prepare.runtime_check(config)
    for manifest in (old_manifest, visual_manifest):
        assert environment == manifest['environment'], 'Runtime environment changed'
    for name, sha in old_manifest['model_identity']['weights_sha256'].items():
        assert prepare.digest(CHECKPOINT / name) == sha, name
    assert digest(CHECKPOINT / 'config.json') == old_manifest['model_identity']['config_sha256']
    auxiliary = read(OLD / 'provenance/model-auxiliary-hashes.json')
    snapshots = read(OLD / 'provenance/preparation.json')['model_snapshots']
    for model, files in auxiliary.items():
        for name, sha in files.items():
            assert prepare.digest(Path(snapshots[model]) / name) == sha, (model, name)
    # Preserve frozen judge weights as well as target weights.
    preparation = read(OLD / 'provenance/preparation.json')
    for model, files in preparation['model_sha256'].items():
        for name, sha in files.items():
            assert prepare.digest(Path(snapshots[model]) / name) == sha, (model, name)
    write(root / 'provenance/original-preparation.json', preparation)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    used, util = map(int, gpu.splitlines()[0].split(','))
    assert used <= 256 and util == 0, f'GPU busy: {gpu}'
    assert shutil.disk_usage(root).free > 10 * 1024**3
    evaluate.parsers.selfcheck()
    datasets, rows = prepare.load_data(config, str(CACHE / 'datasets'), offline=True)
    expected = {r['id']: r for r in read(root / 'protocol/evaluation_items.json')}
    previous = {(r['id'], r['field']): r for r in read(OLD / 'provenance/input-validation.json')['images']}
    image_count = 0
    for row in rows:
        record = evaluate.data_row(row)
        source = datasets[row['subject']][row['dataset_index']]
        fields = [f'image_{i}' for i in range(1, 8) if source.get(f'image_{i}') is not None]
        record.update(input_prompt=evaluate.prompt_for(row, config), image_fields=fields)
        payload = {k: record[k] for k in ('question', 'options', 'reference_answer', 'input_prompt', 'image_fields')}
        assert hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == expected[row['id']]['input_sha256']
        for field in fields:
            rgb = source[field].convert('RGB')
            prior = previous[row['id'], field]
            assert list(rgb.size) == prior['size_wh'] and hashlib.sha256(rgb.tobytes()).hexdigest() == prior['rgb_bytes_sha256']
            image_count += 1
    assert len(rows) == len(expected) == 900 and image_count == len(previous)
    (root / 'provenance').mkdir(exist_ok=True)
    subprocess.run(['tar', '-czf', str(root / 'provenance/evaluation-source.tar.gz'), 'src', 'requirements.txt'], cwd=REPO, check=True)
    (root / 'provenance/pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (root / 'provenance/nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi', '-q'], text=True))
    write(root / 'provenance/preflight.json', {
        'passed': True, 'records': 900, 'images': image_count, 'frozen_inputs_and_rgb_match': True,
        'original_source_hashes_match': True, 'seed42_only_token_capture_difference': True,
        'target_and_judge_weight_and_auxiliary_hashes_match': True, 'environment': environment,
        'git_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: digest(REPO / name) for name in old_manifest['code_sha256']},
        'controller_sha256': {p.name: digest(p) for p in root.glob('*.py')},
        'environment_variables': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'VLLM_ENABLE_V1_MULTIPROCESSING', 'VLLM_BATCH_INVARIANT', 'HF_HOME')},
        'reuse': {'3407': str(BASELINE), '42': str(SEED42)},
        'scope': 'Baseline only. No changed checkpoint, prompt, judge, output budget or heatmap rerun.'})
    aggregate(root, BASELINE, SEED42)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    lock = evaluate.run_lock(root / '.campaign.lock')
    status = {'state': 'running', 'phase': 'preflight', 'driver_pid': os.getpid(),
              'started_at_utc': prepare.now(), 'planned_seeds': SEEDS,
              'new_runs': ['baseline_seed3407_repeat', 'baseline_seed123', 'baseline_seed2026', 'baseline_seed9876'],
              'completed_new_runs': [], 'current_run': None, 'error': None}
    def mark(state=None, phase=None):
        if state:
            status['state'] = state
        if phase:
            status['phase'] = phase
        status['updated_at_utc'] = prepare.now()
        write(root / 'status.json', status)
        print(json.dumps(status), flush=True)
    mark()
    try:
        preflight(root)
        if args.preflight_only:
            mark('prepared', 'preflight_passed')
            aggregate(root, BASELINE, SEED42)
            return
        # Never quietly resume or overwrite a reproduction run / partial seed run.
        for name in status['new_runs']:
            assert not (root / name).exists(), f'Fresh directory required: {name}'
        expected = {r['id'] for r in read(root / 'protocol/evaluation_items.json')}
        for seed, name in [(3407, 'baseline_seed3407_repeat')] + [(s, f'baseline_seed{s}') for s in SEEDS[2:]]:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            used, util = map(int, gpu.splitlines()[0].split(','))
            assert used <= 256 and util == 0, f'GPU busy before {name}: {gpu}'
            status['current_run'] = name
            status['run_started_at_utc'] = prepare.now()
            mark('running', 'repeatability_evaluate' if seed == 3407 else 'evaluate')
            aggregate(root, BASELINE, SEED42)
            command = [sys.executable, str(REPO / 'src/evaluate.py'), '--config', str(root / f'protocol/config.seed{seed}.json'),
                       '--checkpoint', str(CHECKPOINT), '--run-name', name, '--output-root', str(root),
                       '--model-cache', str(CACHE / 'hub'), '--dataset-cache', str(CACHE / 'datasets')]
            write(root / f'{name}.command.json', {'argv': command, 'started_at_utc': prepare.now()})
            started = time.monotonic()
            with (root / f'{name}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=REPO)
            data = validate(root / name, expected, read(root / f'protocol/config.seed{seed}.json'))
            assert evaluate.summaries(data['records'], data['manifest']['configuration']) == data['summary']
            write(root / name / 'verification.json', {'passed': True, 'records': 900, 'all_graded': True,
                                                       'elapsed_seconds': time.monotonic() - started, 'finished_at_utc': prepare.now()})
            status['completed_new_runs'].append(name)
            if seed == 3407:
                mark(phase='compare_repeatability')
                result = compare_repeat(BASELINE, root / name, root / 'repeatability.json')
                if not result['passed']:
                    mark('blocked_reproducibility', 'repeatability_mismatch')
                    aggregate(root, BASELINE, SEED42)
                    return
            aggregate(root, BASELINE, SEED42)
        status['current_run'] = None
        mark('complete', 'completed')
        aggregate(root, BASELINE, SEED42)
        subprocess.run(['tar', '-czf', str(root.parent / (root.name + '.tar.gz')), root.name], cwd=root.parent, check=True)
        archive = root.parent / (root.name + '.tar.gz')
        (root.parent / (archive.name + '.sha256')).write_text(prepare.digest(archive) + '  ' + archive.name + '\n')
    except BaseException:
        status['error'] = traceback.format_exc()
        mark('failed')
        try:
            aggregate(root, BASELINE, SEED42)
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
