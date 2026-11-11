"""One fresh 900-item baseline run, followed by all-item visual replay/report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import evaluate
import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--baseline-root', required=True)
    args = parser.parse_args()
    root = Path(args.campaign).resolve()
    old = Path(args.baseline_root).resolve()
    config_path = root / 'protocol/config.seed42.json'
    config = prepare.config(config_path)
    started = time.time()
    phase = 'preflight'
    run_name = 'baseline_seed42'
    lock = evaluate.run_lock(root / '.campaign.lock')

    def status(state, error=None):
        prepare.write(root / 'status.json', {'state': state, 'phase': phase, 'run_name': run_name,
            'full_evaluation_runs_requested': 1, 'started_at_unix': started,
            'updated_at': prepare.now(), 'elapsed_seconds': time.time() - started,
            'driver_pid': os.getpid(), 'error': error})

    def execute(command, log):
        with (root / log).open('w', encoding='utf-8') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)

    status('running')
    try:
        if (root / run_name).exists():
            raise ValueError('Fresh baseline directory already exists; do not generate twice')
        if prepare.read(root / 'pilot_stress/progress.json')['state'] != 'complete':
            raise ValueError('GPU pilot including longest old context must pass first')
        pilot_manifest = prepare.read(root / 'pilot_stress/manifest.json')
        pilot_fingerprint = hashlib.sha256(json.dumps({'config': config,
            'results_sha256': prepare.digest(old / 'baseline_seed3407/results.json'),
            'script_sha256': prepare.digest(prepare.ROOT / 'src/visual_attention.py'),
            'pilot': True, 'chunk_size': 256}, sort_keys=True).encode()).hexdigest()
        if pilot_manifest['fingerprint'] != pilot_fingerprint:
            raise ValueError('Pilot was produced by a different script/configuration')
        old_config = prepare.read(old / 'protocol/config.seed3407.json')
        old_config['sampling']['seed'] = 42
        if old_config != config or config['judge']['sampling']['seed'] != 3407:
            raise ValueError('Only the target inference seed may change')
        environment = prepare.runtime_check(config)
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        used, utilization = map(int, gpu.splitlines()[0].split(','))
        if used > 256 or utilization != 0:
            raise ValueError(f'GPU is occupied: {gpu}')
        execute([sys.executable, '-m', 'unittest', 'discover', '-s', str(prepare.ROOT / 'src'),
                 '-p', 'test_visual_attention.py', '-v'], 'tests.log')
        execute([sys.executable, str(prepare.ROOT / 'src/evaluate.py'), '--config', str(config_path), '--selfcheck'], 'parser_selfcheck.log')
        expected_hashes = prepare.read(old / 'protocol/linux_source_hashes.json')['sha256']
        for name, expected in expected_hashes.items():
            if not (name.startswith('src/') or name == 'requirements.txt'):
                continue
            content = (prepare.ROOT / name).read_bytes().replace(b'\r\n', b'\n')
            if name == 'src/evaluate.py':
                lines = content.decode().splitlines(keepends=True)
                start = next(i for i, line in enumerate(lines) if "visual_replay={'prompt_token_ids'" in line)
                if "'source':'vllm_exact_token_ids'}," not in lines[start + 4]:
                    raise ValueError('Unexpected evaluation modification')
                content = ''.join(lines[:start] + lines[start + 5:]).encode()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Frozen baseline source changed beyond token capture: {name}')
        phase = 'verify_inputs'
        status('running')
        datasets, rows = prepare.load_data(config, os.environ['MMMU_DATASET_CACHE'], offline=True)
        expected = {r['id']: r for r in prepare.read(root / 'protocol/evaluation_items.json')}
        previous_images = {(r['id'], r['field']): r for r in prepare.read(old / 'provenance/input-validation.json')['images']}
        count = 0
        for row in rows:
            record = evaluate.data_row(row)
            source = datasets[row['subject']][row['dataset_index']]
            fields = [f'image_{i}' for i in range(1, 8) if source.get(f'image_{i}') is not None]
            record.update(input_prompt=evaluate.prompt_for(row, config), image_fields=fields)
            canonical = {k: record[k] for k in ('question', 'options', 'reference_answer', 'input_prompt', 'image_fields')}
            digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if digest != expected[row['id']]['input_sha256']:
                raise ValueError('Frozen input mismatch: ' + row['id'])
            for field in fields:
                rgb = source[field].convert('RGB')
                prior = previous_images[row['id'], field]
                if list(rgb.size) != prior['size_wh'] or hashlib.sha256(rgb.tobytes()).hexdigest() != prior['rgb_bytes_sha256']:
                    raise ValueError('Frozen image mismatch: ' + row['id'])
                count += 1
        if len(rows) != len(expected) or len(rows) != 900 or count != len(previous_images):
            raise ValueError('Expected the same 900 questions and images')
        sources = {str(p.relative_to(prepare.ROOT)): prepare.digest(p) for p in (prepare.ROOT / 'src').glob('*.py')}
        provenance = {'config': config, 'config_sha256': prepare.digest(config_path), 'inputs': 900, 'images': count,
            'all_frozen_inputs_match': True, 'all_previous_rgb_hashes_match': True,
            'source_sha256': sources, 'environment': environment,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=prepare.ROOT, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=prepare.ROOT, text=True),
            'only_evaluation_change': 'save exact input/output token IDs and image preprocessing metadata'}
        prepare.write(root / 'provenance/preflight.json', provenance)
        phase = 'evaluate'
        status('running')
        command = [sys.executable, str(prepare.ROOT / 'src/evaluate.py'), '--config', str(config_path),
            '--run-name', run_name, '--output-root', str(root), '--model-cache', os.environ['MMMU_MODEL_CACHE'],
            '--dataset-cache', os.environ['MMMU_DATASET_CACHE']]
        execute(command, 'evaluation.log')
        result = prepare.read(root / run_name / 'results.json')
        records = result['records']
        evaluate.check_records(records, config)
        if any(r['grading']['final']['status'] != 'graded' or 'visual_replay' not in r for r in records):
            raise ValueError('Incomplete grading or exact replay token IDs')
        if sum(r['grading']['final']['hit'] for r in records) != result['summary']['correct']:
            raise ValueError('Score aggregation mismatch')
        prepare.write(root / 'provenance/evaluation-validation.json', {'records': 900,
            'all_graded': True, 'all_exact_replay_ids': True, 'correct': result['summary']['correct']})
        phase = 'heatmaps'
        status('running')
        execute([sys.executable, str(prepare.ROOT / 'src/visual_attention.py'), '--config', str(config_path),
            '--results', str(root / run_name / 'results.json'), '--output', str(root / 'analysis'),
            '--model-cache', os.environ['MMMU_MODEL_CACHE'], '--dataset-cache', os.environ['MMMU_DATASET_CACHE'],
            '--baseline-results', str(old / 'baseline_seed3407/results.json')], 'heatmaps.log')
        summary = prepare.read(root / 'analysis/analysis_summary.json')
        if summary['heatmaps_complete'] != 900:
            raise ValueError('Heatmap coverage is not 900/900')
        phase = 'completed'
        status('complete')
    except Exception:
        status('failed', traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
