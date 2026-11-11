"""Verify downloaded seed42 results, numeric attention arrays and image files."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/mmmu_visual_tokens_seed42_20261010'
sys.path.insert(0, str(ROOT / 'code/evaluation/mmdl-visual-token-analysis/src'))
import evaluate

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    result = read(RUN / 'baseline_seed42/results.json')
    summary = read(RUN / 'baseline_seed42/summary.json')
    config = read(RUN / 'protocol/config.seed42.json')
    records = result['records']
    predictions = [json.loads(x) for x in (RUN / 'baseline_seed42/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id = {r['id']: r for r in predictions}
    previous = read(ROOT / 'results/mmmu_first_baseline_20261009/baseline_seed3407/results.json')
    old = {r['id']: r for r in previous['records']}
    expected = {r['id'] for r in read(RUN / 'protocol/evaluation_items.json')}
    checks = {}
    checks['state_complete'] = read(RUN / 'status.json')['state'] == 'complete'
    checks['unique_900_ids'] = len(records) == len(predictions) == len(by_id) == len(expected) == 900 and {r['id'] for r in records} == expected == set(by_id) == set(old)
    checks['all_summary_fields_match'] = summary == result['summary'] == evaluate.summaries(records, config)
    checks['all_length_incorrect'] = all(not r['grading']['final']['hit'] for r in records if r['model_response']['finish_reason'] == 'length')
    preflight = read(RUN / 'provenance/preflight.json')
    checks['frozen_inputs_rgb_match'] = preflight['all_frozen_inputs_match'] and preflight['all_previous_rgb_hashes_match']
    checks['seed_and_judge'] = config['sampling']['seed'] == 42 and config['judge']['sampling']['seed'] == 3407
    old_config = read(ROOT / 'results/mmmu_first_baseline_20261009/protocol/config.seed3407.json')
    old_config['sampling']['seed'] = 42
    checks['only_sampling_seed_changed'] = old_config == config
    analysis_manifest = read(RUN / 'analysis/manifest.json')
    analysis_summary = read(RUN / 'analysis/analysis_summary.json')
    progress = read(RUN / 'analysis/progress.json')
    checks['heatmap_complete_900'] = progress['state'] == 'complete' and progress['completed'] == progress['expected'] == analysis_summary['heatmaps_complete'] == 900
    metadata_ids = {p.parent.name for p in (RUN / 'analysis').glob('*/metadata.json')}
    checks['metadata_ids_exact'] = metadata_ids == expected
    transitions = collections.Counter()
    group_values = collections.defaultdict(lambda: collections.defaultdict(list))
    methods = collections.defaultdict(lambda: {'count': 0, 'correct': 0})
    images_checked = arrays_checked = probes_checked = 0
    for r in records:
        rid = r['id']
        capture = r['visual_replay']
        assert capture == by_id[rid]['visual_replay'], rid
        assert r['model_response'] == by_id[rid]['model_response'], rid
        assert len(capture['prompt_token_ids']) == r['model_response']['input_tokens'], rid
        assert len(capture['output_token_ids']) == r['model_response']['output_tokens'], rid
        assert capture['source'] == 'vllm_exact_token_ids', rid
        assert len(capture['image_grid_thw']) == len(r['image_fields']), rid
        case = RUN / 'analysis' / rid
        meta = read(case / 'metadata.json')
        assert meta['id'] == rid and meta['fingerprint'] == analysis_manifest['fingerprint'], rid
        assert meta['record_sha256'] == hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest(), rid
        assert meta['token_source'] == 'vllm_exact_token_ids', rid
        assert meta['grading'] == r['grading'], rid
        assert (case / 'index.html').stat().st_size > 0, rid
        assert meta['layers_zero_based'] == [8, 17, 26, 35], rid
        token_count = sum(t * (h//2) * (w//2) for t, h, w in capture['image_grid_thw'])
        with np.load(case / 'attention.npz', allow_pickle=False) as data:
            assert set(data.files) == {f'{name}_layer_{layer}' for name in meta['probes'] for layer in meta['layers_zero_based']}, rid
            for name, probe in meta['probes'].items():
                values = []
                for layer in meta['layers_zero_based']:
                    vector = data[f'{name}_layer_{layer}']
                    assert vector.shape == (token_count,), rid
                    assert np.isfinite(vector).all() and (vector >= 0).all() and vector.sum() <= 1.00001, rid
                    values.append(vector)
                    arrays_checked += 1
                average = np.mean(values, axis=0)
                assert math.isclose(float(average.sum()), probe['total_image_attention_mass'], abs_tol=1e-6), rid
                offset = 0
                for index, im in enumerate(probe['images']):
                    shape = [capture['image_grid_thw'][index][1] // 2, capture['image_grid_thw'][index][2] // 2]
                    assert im['grid_hw'] == shape, rid
                    n = math.prod(shape)
                    assert math.isclose(float(average[offset:offset+n].sum()), im['image_attention_mass'], abs_tol=1e-6), rid
                    offset += n
                    with Image.open(case / im['file']) as jpeg:
                        assert list(jpeg.size) == r['original_image_sizes_wh'][index], rid
                        jpeg.verify()
                    images_checked += 1
                assert offset == token_count, rid
                probes_checked += 1
                group = 'correct' if r['grading']['final']['hit'] else 'incorrect'
                for label in [group, 'finish_' + r['model_response']['finish_reason'], 'subject_' + r['subject']]:
                    group_values[label][name].append(float(average.sum()))
        for index, size in enumerate(r['original_image_sizes_wh']):
            with Image.open(case / f'original_image_{index+1}.jpg') as jpeg:
                assert list(jpeg.size) == size, rid
                jpeg.verify()
            images_checked += 1
        before = bool(old[rid]['grading']['final']['hit'])
        after = bool(r['grading']['final']['hit'])
        transitions[f'{"correct" if before else "incorrect"}_to_{"correct" if after else "incorrect"}'] += 1
        method = r['grading']['final']['method']
        methods[method]['count'] += 1
        methods[method]['correct'] += int(after)
    checks['all_exact_tokens_and_prediction_records_match'] = True
    checks['all_attention_arrays_grids_and_image_files_valid'] = True
    checks['transitions_match'] = dict(transitions) == analysis_summary['transitions_from_seed3407']
    for group, probes in group_values.items():
        for name, values in probes.items():
            reference = analysis_summary['attention_by_group'][group][name]
            assert reference['count'] == len(values)
            assert math.isclose(reference['mean_image_attention_mass'], sum(values)/len(values), abs_tol=1e-8)
    checks['attention_group_means_recomputed'] = True
    assert all(checks.values()), checks
    attempts = [json.loads(x) for x in (RUN / 'baseline_seed42/judge_attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    stats = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'checks': checks,
             'records': 900, 'correct': summary['correct'], 'image_files_verified': images_checked,
             'attention_arrays_verified': arrays_checked, 'query_probes_verified': probes_checked,
             'transitions': dict(transitions), 'methods': dict(methods), 'judge_attempts': len(attempts),
             'judge_unique_ids': len({a['id'] for a in attempts}),
             'generation_seconds_sum': sum(r['model_response']['elapsed_seconds'] for r in records),
             'length_generation_seconds_sum': sum(r['model_response']['elapsed_seconds'] for r in records if r['model_response']['finish_reason']=='length'),
             'outputs_created_at_utc': result['created_at'], 'analysis_manifest_created_at_utc': analysis_manifest['created_at'],
             'end_status': read(RUN / 'status.json')}
    (RUN / 'verification.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    main()
