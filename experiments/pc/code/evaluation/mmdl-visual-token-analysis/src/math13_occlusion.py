"""Predeclared, equal-area Math_13 input interventions; regenerate from scratch.

This pilot estimates sensitivity to three particular gray masks, not pixelwise
causal importance. The untouched baseline files are inputs only.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image

import evaluate
import parsers
import prepare

ITEM = 'validation_Math_13'
SEEDS = [42, 3407, 20261014]
MASK_SEED = 20261014
SIDE = 96
FILL = (127, 127, 127)


def rgb_hash(im):
    return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()


def mask_image(im, box):
    a = np.asarray(im.convert('RGB')).copy()
    if box:
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height
        a[y0:y1, x0:x1] = FILL
    return Image.fromarray(a)


def rectangle_attention(weights, box, width, height):
    # Exact area overlap with the original token grid, not bilinear colors.
    if box is None:
        return 0.0
    h, w = weights.shape
    x0, y0, x1, y1 = box
    xs = np.linspace(0, width, w + 1)
    ys = np.linspace(0, height, h + 1)
    xf = np.maximum(0, np.minimum(xs[1:], x1) - np.maximum(xs[:-1], x0)) / (width / w)
    yf = np.maximum(0, np.minimum(ys[1:], y1) - np.maximum(ys[:-1], y0)) / (height / h)
    return float((weights * yf[:, None] * xf[None, :]).sum())


def prepare_experiment(campaign, output):
    config_path = campaign / 'protocol/config.seed42.json'
    config = prepare.config(config_path)
    source_manifest = prepare.read(campaign / 'baseline_seed42/manifest.json')
    paths = source_manifest['cache_paths']
    records = prepare.read(campaign / 'baseline_seed42/results.json')['records']
    baseline = next(r for r in records if r['id'] == ITEM)
    datasets, rows = prepare.load_data(config, paths['dataset'], offline=True)
    row = next(r for r in rows if r['id'] == ITEM)
    source = datasets[row['subject']][row['dataset_index']]
    assert baseline['image_fields'] == ['image_1']
    original = source['image_1'].convert('RGB')
    assert original.size == (502, 491)
    assert evaluate.prompt_for(row, config) == baseline['input_prompt']
    assert evaluate.data_row(row)['reference_answer'] == baseline['reference_answer'] == 'C'
    old_validation = Path('/home/rlatjswo/mmmu_baseline_20261009/provenance/input-validation.json')
    expected = next(r for r in prepare.read(old_validation)['images']
                    if r['id'] == ITEM and r['field'] == 'image_1')
    assert rgb_hash(original) == expected['rgb_bytes_sha256']
    case = campaign / 'analysis' / ITEM
    meta = prepare.read(case / 'metadata.json')
    shape = meta['probes']['before_final_answer']['images'][0]['grid_hw']
    with np.load(case / 'attention.npz') as data:
        attn = np.mean([data[f'before_final_answer_layer_{i}']
                        for i in meta['layers_zero_based']], axis=0).reshape(shape)
    h, w = shape
    peak_y, peak_x = np.unravel_index(attn.argmax(), attn.shape)
    left = max(0, min(original.width - SIDE, round((peak_x + .5) * original.width / w - SIDE / 2)))
    top = max(0, min(original.height - SIDE, round((peak_y + .5) * original.height / h - SIDE / 2)))
    rng = random.Random(MASK_SEED)
    rx, ry = rng.randrange(original.width - SIDE + 1), rng.randrange(original.height - SIDE + 1)
    boxes = {
        'original': None,
        'attention_high': [left, top, left + SIDE, top + SIDE],
        'geometry_D': [112, 128, 208, 224],
        'random_region': [rx, ry, rx + SIDE, ry + SIDE],
    }
    descriptions = {
        'original': 'Unmodified pinned dataset RGB image; lossless PNG.',
        'attention_high': '96px square centered/clamped at the highest final-answer attention token.',
        'geometry_D': 'Manually selected before inference: label D, inner upper-left vertex, adjoining edges.',
        'random_region': 'One uniformly sampled valid square; fixed PRNG seed; no rejection based on image content or answer.',
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    conditions = []
    for name, box in boxes.items():
        im = mask_image(original, box)
        a, b = np.asarray(original), np.asarray(im)
        changed = np.any(a != b, axis=2)
        allowed = np.zeros(changed.shape, dtype=bool)
        if box:
            x0, y0, x1, y1 = box
            allowed[y0:y1, x0:x1] = True
            assert int(allowed.sum()) == SIDE * SIDE
        assert not np.any(changed & ~allowed)
        assert np.array_equal(a[~allowed], b[~allowed])
        assert im.size == original.size
        path = output / 'images' / f'{name}.png'
        im.save(path)
        assert rgb_hash(Image.open(path)) == rgb_hash(im)
        mass = rectangle_attention(attn, box, *original.size)
        conditions.append({'name': name, 'box_xyxy_exclusive': box,
            'description': descriptions[name], 'image': str(path.relative_to(output)),
            'rgb_sha256': rgb_hash(im), 'png_sha256': prepare.digest(path),
            'mask_area_pixels': int(allowed.sum()), 'changed_pixels': int(changed.sum()),
            'mask_area_fraction': float(allowed.mean()), 'outside_mask_exactly_unchanged': True,
            'baseline_attention_mass_in_rectangle': mass,
            'fraction_of_baseline_image_attention': mass / float(attn.sum())})
    experiment = {'item': ITEM, 'seeds': SEEDS, 'mask_location_seed': MASK_SEED,
        'mask_side_pixels': SIDE, 'mask_rgb': FILL, 'conditions': conditions,
        'expected_generations': len(conditions) * len(SEEDS),
        'configuration': config, 'cache_paths': paths,
        'source_config_sha256': prepare.digest(config_path),
        'source_record_sha256': hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest(),
        'source_attention_sha256': prepare.digest(case / 'attention.npz'),
        'source_rgb_matches_pinned_baseline': True,
        'baseline_image_attention_mass': float(attn.sum()),
        'original_size_wh': list(original.size),
        'method': 'equal-area gray occlusion; full fresh autoregressive generation; paired seeds',
        'scoring': 'Frozen target parser. Length is incorrect. Pending judge is unresolved, never guessed.',
        'limitations': ['3 seeds and one random mask: exploratory case study only.',
            'Gray fill introduces an artificial feature as well as removing information.',
            'Geometry_D masks one local vertex, not all geometry.',
            'No prior reasoning text is supplied to any condition.',
            'Not a dense saliency map; no claim about untested pixels.']}
    fingerprint = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    if (output / 'experiment.json').exists():
        assert prepare.read(output / 'experiment.json')['fingerprint'] == fingerprint, 'Experiment changed'
    experiment['fingerprint'] = fingerprint
    prepare.write(output / 'experiment.json', experiment)
    prepare.write(output / 'source_record.json', baseline)
    shutil.copy2(case / 'before_final_answer_image_1.jpg', output / 'images/baseline_attention.jpg')
    shutil.copy2(case / 'attention.npz', output / 'baseline_attention.npz')
    prepare.write(output / 'source_attention_metadata.json', meta)
    print(json.dumps({'prepared': str(output), 'conditions': conditions, 'expected': experiment['expected_generations']}), flush=True)
    return experiment, baseline


def summarize(experiment, records):
    by_key = {(r['seed'], r['condition']): r for r in records}
    assert len(by_key) == len(records)
    totals = []
    for condition in experiment['conditions']:
        name = condition['name']
        rr = [by_key[s, name] for s in SEEDS if (s, name) in by_key]
        pairs = [(r, by_key.get((r['seed'], 'original'))) for r in rr]
        valid = [(a, b) for a, b in pairs if b is not None and
                 a['grading']['status'] == b['grading']['status'] == 'graded']
        flips = [(a['seed'], b['grading']['extracted_answer'], a['grading']['extracted_answer'])
                 for a, b in valid if a['grading']['extracted_answer'] != b['grading']['extracted_answer']]
        totals.append({'condition': name, 'completed': len(rr),
            'correct': sum(r['grading']['hit'] == 1 for r in rr),
            'unresolved': sum(r['grading']['status'] != 'graded' for r in rr),
            'length': sum(r['model_response']['finish_reason'] == 'length' for r in rr),
            'answers': {str(r['seed']): r['grading']['extracted_answer'] for r in rr},
            'paired_answer_changes': flips,
            'correct_to_wrong': sum(b['grading']['hit'] == 1 and a['grading']['hit'] == 0 for a, b in valid),
            'wrong_to_correct': sum(b['grading']['hit'] == 0 and a['grading']['hit'] == 1 for a, b in valid),
            'elapsed_seconds': sum(r['model_response']['elapsed_seconds'] for r in rr)})
    original42 = by_key.get((42, 'original'))
    return {'expected': experiment['expected_generations'], 'completed': len(records),
        'conditions': totals, 'all_scored': all(r['grading']['status'] == 'graded' for r in records),
        'original_seed42_matches_historical_output_ids': original42.get('historical_output_ids_equal') if original42 else None,
        'generation_seconds': sum(r['model_response']['elapsed_seconds'] for r in records)}


def run(experiment, baseline, output):
    c = experiment['configuration']
    environment = prepare.runtime_check(c)
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    model = prepare.snapshot(c, experiment['cache_paths']['model'])
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    rt = c['runtime']
    records = prepare.read_jsonl(output / 'predictions.jsonl')
    assert len({(r['seed'], r['condition']) for r in records}) == len(records)
    assert all(r['fingerprint'] == experiment['fingerprint'] for r in records)
    done = {(r['seed'], r['condition']) for r in records}
    if len(done) == experiment['expected_generations']:
        prepare.write(output / 'summary.json', summarize(experiment, records))
        return
    provenance = {'environment': environment, 'script_sha256': prepare.digest(__file__),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=prepare.ROOT, text=True).strip(),
        'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=prepare.ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=prepare.ROOT, text=True),
        'model_snapshot': str(model), 'started_at': prepare.now(), 'pid': os.getpid()}
    if (output / 'provenance.json').exists():
        old = prepare.read(output / 'provenance.json')
        assert old['script_sha256'] == provenance['script_sha256'], 'Code changed while resuming'
    else:
        prepare.write(output / 'provenance.json', provenance)
    llm = LLM(model=str(model), dtype=rt['dtype'], quantization=None,
        tensor_parallel_size=rt['tensor_parallel_size'], max_model_len=rt['max_model_len'],
        gpu_memory_utilization=rt['gpu_memory_utilization'], max_num_seqs=rt['max_num_seqs'],
        max_num_batched_tokens=rt['max_num_batched_tokens'], enforce_eager=rt['enforce_eager'],
        limit_mm_per_prompt={'image': rt['max_images_per_prompt']}, seed=c['sampling']['seed'],
        mm_processor_kwargs={'size': {'shortest_edge': c['vision']['min_pixels'],
            'longest_edge': c['vision']['max_pixels']}, 'do_resize': False})
    for seed in SEEDS:
        for condition in experiment['conditions']:
            name = condition['name']
            if (seed, name) in done:
                continue
            prepare.write(output / 'status.json', {'state': 'running', 'phase': 'generate',
                'completed': len(records), 'expected': experiment['expected_generations'],
                'current': {'seed': seed, 'condition': name}, 'updated_at': prepare.now(), 'pid': os.getpid()})
            start = time.perf_counter()
            im = Image.open(output / condition['image']).convert('RGB')
            assert rgb_hash(im) == condition['rgb_sha256']
            messages = [{'role': 'user', 'content': [dict(type='image', image=im, **c['vision']),
                {'type': 'text', 'text': baseline['input_prompt']}]}]
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            assert rendered == baseline['rendered_chat']
            images, videos, kwargs = process_vision_info(messages,
                image_patch_size=processor.image_processor.patch_size,
                return_video_kwargs=True, return_video_metadata=True)
            assert videos is None
            kwargs['do_resize'] = False
            encoded = processor(text=[rendered], images=images, return_tensors='pt', **kwargs)
            assert encoded['input_ids'][0].tolist() == baseline['visual_replay']['prompt_token_ids']
            assert encoded['image_grid_thw'].tolist() == baseline['visual_replay']['image_grid_thw']
            sample = SamplingParams(**dict(c['sampling'], seed=seed), max_tokens=c['max_new_tokens'], stop_token_ids=[])
            generated = llm.generate([{'prompt': rendered, 'multi_modal_data': {'image': images},
                'mm_processor_kwargs': kwargs}], sample, use_tqdm=False)[0]
            out = generated.outputs[0]
            assert list(generated.prompt_token_ids) == baseline['visual_replay']['prompt_token_ids']
            assert out.finish_reason in ('stop', 'length')
            assert out.finish_reason != 'length' or len(out.token_ids) == c['max_new_tokens']
            record = {k: copy.deepcopy(baseline[k]) for k in
                ('id', 'subject', 'question_type', 'question', 'options', 'reference_answer', 'input_prompt')}
            record.update(seed=seed, condition=name, fingerprint=experiment['fingerprint'],
                image_rgb_sha256=condition['rgb_sha256'],
                visual_replay={'prompt_token_ids': list(generated.prompt_token_ids),
                    'output_token_ids': list(out.token_ids), 'image_grid_thw': encoded['image_grid_thw'].tolist(),
                    'processed_image_sizes_wh': [list(i.size) for i in images], 'source': 'vllm_exact_token_ids'},
                model_response={'raw': out.text, 'processed': out.text.split('</think>')[-1].strip(),
                    'finish_reason': out.finish_reason, 'stop_reason': out.stop_reason,
                    'output_tokens': len(out.token_ids), 'input_tokens': len(generated.prompt_token_ids),
                    'elapsed_seconds': time.perf_counter() - start}, completed_at=prepare.now())
            record['grading'] = parsers.grade_response(record)
            if seed == 42 and name == 'original':
                record['historical_output_ids_equal'] = list(out.token_ids) == baseline['visual_replay']['output_token_ids']
            prepare.append(output / 'predictions.jsonl', record)
            (output / 'responses').mkdir(exist_ok=True)
            (output / 'responses' / f'{name}_seed{seed}.txt').write_text(out.text, encoding='utf-8')
            records.append(record)
            done.add((seed, name))
            prepare.write(output / 'summary.json', summarize(experiment, records))
            print(json.dumps({'completed': len(records), 'expected': experiment['expected_generations'],
                'condition': name, 'seed': seed, 'tokens': len(out.token_ids),
                'seconds': record['model_response']['elapsed_seconds'], 'grading': record['grading']}), flush=True)
            del encoded
    assert len(records) == experiment['expected_generations']
    prepare.write(output / 'status.json', {'state': 'complete', 'phase': 'generation_and_target_parser',
        'completed': len(records), 'expected': experiment['expected_generations'],
        'all_scored': all(r['grading']['status'] == 'graded' for r in records), 'updated_at': prepare.now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = evaluate.run_lock(output / '.experiment.lock')
    try:
        experiment, baseline = prepare_experiment(Path(args.campaign).resolve(), output)
        if not args.prepare_only:
            run(experiment, baseline, output)
    except Exception:
        prepare.write(output / 'status.json', {'state': 'failed', 'updated_at': prepare.now(),
            'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    main()
