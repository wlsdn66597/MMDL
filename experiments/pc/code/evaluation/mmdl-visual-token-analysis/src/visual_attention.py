"""Teacher-forced Qwen3-VL attention probes; never generate or change scores.

Pinned Transformers 4.57.0 SDPA receives post-QK-normalization, post-RoPE Q/K.
Recompute only selected query rows in FP32, normalize over ALL causal keys,
then retain image keys. The original SDPA call still computes hidden states.
"""
import argparse
import hashlib
import html
import json
import math
import re
import time
from pathlib import Path

import prepare
import evaluate


def anchors(tokenizer, prompt_length, output_ids):
    """Query t-1 predicts token t; final answer anchor excludes answer value."""
    if not output_ids:
        raise ValueError('Cannot probe an empty response')
    result = {'before_first_token': prompt_length - 1,
              'before_last_token': prompt_length + len(output_ids) - 2}
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    matches = list(re.finditer(r'final\s+answer\s*:\s*\S', text, re.I))
    if matches:
        # Prefix lengths are measured by decoding actual generated IDs, never
        # replacing their tokenization with a new encoding of response text.
        char_index = matches[-1].end() - 1
        lo, hi = 1, len(output_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(tokenizer.decode(output_ids[:mid], skip_special_tokens=False)) > char_index:
                hi = mid
            else:
                lo = mid + 1
        result['before_final_answer'] = prompt_length + lo - 2
    return result


def selected_attention(query, key, mask, positions, offset, groups, scaling):
    """Return [selected queries, keys], mean over heads; batch must be one."""
    import torch
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError('Only unpadded batch size one is supported')
    rows = torch.tensor(positions, device=query.device, dtype=torch.long)
    key_positions = torch.arange(key.shape[-2], device=query.device)
    causal = key_positions[None, :] <= (rows + offset)[:, None]
    total = torch.zeros((len(positions), key.shape[-2]), device=query.device, dtype=torch.float32)
    for head in range(query.shape[1]):
        logits = query[0, head, rows].float() @ key[0, head // groups].float().T
        logits *= scaling
        logits.masked_fill_(~causal, float('-inf'))
        if mask is not None:
            if mask.ndim != 4:
                raise ValueError('Unsupported attention mask')
            selected_mask = mask[0, head if mask.shape[1] > 1 else 0,
                                 rows if mask.shape[-2] > 1 else 0, :key.shape[-2]]
            if selected_mask.dtype == torch.bool:
                logits.masked_fill_(~selected_mask, float('-inf'))
            else:
                logits += selected_mask.float()
        probabilities = logits.softmax(-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError('Non-finite attention probabilities')
        total += probabilities / query.shape[1]
    if not torch.allclose(total.sum(-1), torch.ones(len(positions), device=query.device), atol=2e-5):
        raise ValueError('Attention does not sum to one over causal keys')
    return total


class AttentionTap:
    def __init__(self, layers, positions, image_positions):
        self.layers = layers
        self.positions = positions
        self.image_positions = image_positions
        self.offset = 0
        self.maps = {}

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.registry = ALL_ATTENTION_FUNCTIONS
        self.original = self.registry['sdpa']
        self.registry.register('sdpa', self.forward)
        return self

    def __exit__(self, *exc):
        self.registry.register('sdpa', self.original)

    def forward(self, module, query, key, value, attention_mask, **kwargs):
        layer = getattr(module, 'layer_idx', None)
        if layer in self.layers:
            if key.shape[-2] != self.offset + query.shape[-2]:
                raise ValueError('Unexpected cache/query positions')
            active = {name: pos - self.offset for name, pos in self.positions.items()
                      if self.offset <= pos < self.offset + query.shape[-2]}
            if active:
                weights = selected_attention(query, key, attention_mask, list(active.values()),
                                             self.offset, module.num_key_value_groups, kwargs['scaling'])
                for index, name in enumerate(active):
                    self.maps[(name, layer)] = weights[index, self.image_positions].detach().cpu().numpy()
        return self.original(module, query, key, value, attention_mask, **kwargs)


def image_layout(input_ids, image_token_id, grids, merge):
    positions = [i for i, token in enumerate(input_ids) if token == image_token_id]
    shapes = []
    offset = 0
    for t, h, w in grids:
        if t != 1 or h % merge or w % merge:
            raise ValueError('Only valid still-image merged grids are supported')
        shape = (h // merge, w // merge)
        count = math.prod(shape)
        segment = positions[offset:offset + count]
        if len(segment) != count or segment != list(range(segment[0], segment[0] + count)):
            raise ValueError('Image token count/contiguity differs from grid')
        shapes.append(shape)
        offset += count
    if offset != len(positions):
        raise ValueError('Unexpected extra image tokens')
    return positions, shapes


def overlay(rgb, grid, maximum):
    import numpy as np
    from PIL import Image
    # One common scale across every anchor/image of a question. Zero is
    # transparent; no per-image contrast boosting that could imply attention.
    enlarged = np.asarray(Image.fromarray(grid.astype('float32')).resize(rgb.size, Image.Resampling.BILINEAR))
    intensity = np.clip(enlarged / maximum, 0, 1) if maximum > 0 else enlarged * 0
    stops = np.array([[30, 60, 180], [0, 180, 210], [255, 220, 50], [230, 30, 30]], dtype=float)
    z = intensity * 3
    index = np.minimum(z.astype(int), 2)
    color = stops[index] * (1 - (z - index)[..., None]) + stops[index + 1] * (z - index)[..., None]
    alpha = (0.6 * np.sqrt(intensity))[..., None]
    blended = np.asarray(rgb).astype(float) * (1 - alpha) + color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype('uint8'))


def prepare_item(row, source, config, processor):
    from qwen_vl_utils import process_vision_info
    fields = [f'image_{i}' for i in range(1, 8) if source.get(f'image_{i}') is not None]
    originals = [source[field].convert('RGB') for field in fields]
    text = evaluate.prompt_for(row, config)
    messages = [{'role': 'user', 'content': [dict(type='image', image=im, **config['vision'])
                for im in originals] + [{'type': 'text', 'text': text}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size,
                                                return_video_kwargs=True, return_video_metadata=True)
    if videos is not None:
        raise ValueError('Unexpected video')
    kwargs['do_resize'] = False
    encoded = processor(text=[rendered], images=images, return_tensors='pt', **kwargs)
    return originals, encoded, rendered


def replay(model, encoded, output_ids, tokenizer, chunk_size):
    import torch
    length = encoded['input_ids'].shape[1]
    positions = anchors(tokenizer, length, output_ids)
    image_positions, shapes = image_layout(encoded['input_ids'][0].tolist(), model.config.image_token_id,
                                           encoded['image_grid_thw'].tolist(), model.config.vision_config.spatial_merge_size)
    depth = model.config.text_config.num_hidden_layers
    layers = sorted(set([depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]))
    inputs = encoded.to(model.device)
    tap = AttentionTap(layers, positions, image_positions)
    with torch.inference_mode(), tap:
        output = model(**inputs, use_cache=True, logits_to_keep=1,
                       cache_position=torch.arange(length, device=model.device))
        cache = output.past_key_values
        del output
        # Feed only through the last required query. Generated special tokens
        # stay literal token IDs and are not reinterpreted as fresh image input.
        needed = max(positions.values()) - length + 1
        for start in range(0, needed, chunk_size):
            ids = output_ids[start:min(needed, start + chunk_size)]
            tap.offset = length + start
            end = tap.offset + len(ids)
            output = model(input_ids=torch.tensor([ids], device=model.device), past_key_values=cache,
                           attention_mask=torch.ones((1, end), device=model.device, dtype=torch.long),
                           cache_position=torch.arange(tap.offset, end, device=model.device),
                           use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            del output
        del cache
    expected = {(name, layer) for name in positions for layer in layers}
    if set(tap.maps) != expected:
        raise ValueError('Missing attention probes')
    return tap.maps, positions, layers, shapes


def save_item(destination, record, originals, maps, positions, layers, shapes, fingerprint, token_source):
    import numpy as np
    destination.mkdir(parents=True, exist_ok=True)
    averaged = {name: np.mean([maps[name, layer] for layer in layers], axis=0) for name in positions}
    maximum = max(float(values.max()) for values in averaged.values())
    arrays = {f'{name}_layer_{layer}': values for (name, layer), values in maps.items()}
    np.savez_compressed(destination / 'attention.npz', **arrays)
    probes = {}
    for name, values in averaged.items():
        offset = 0
        images = []
        for index, (rgb, shape) in enumerate(zip(originals, shapes)):
            count = math.prod(shape)
            grid = values[offset:offset + count].reshape(shape)
            offset += count
            filename = f'{name}_image_{index + 1}.jpg'
            overlay(rgb, grid, maximum).save(destination / filename, quality=92)
            mass = float(grid.sum())
            conditional = grid.ravel() / mass if mass > 0 else grid.ravel()
            entropy = float(-(conditional * np.log(conditional.clip(1e-30))).sum())
            images.append({'file': filename, 'grid_hw': list(shape), 'image_attention_mass': mass,
                           'conditional_entropy': entropy,
                           'top_10_percent_mass_fraction': float(np.sort(conditional)[-max(1, math.ceil(count * .1)):].sum())})
        probes[name] = {'query_position': positions[name], 'total_image_attention_mass': float(values.sum()), 'images': images}
    for index, rgb in enumerate(originals):
        rgb.save(destination / f'original_image_{index + 1}.jpg', quality=92)
    metadata = {'id': record['id'], 'subject': record['subject'], 'fingerprint': fingerprint,
                'token_source': token_source, 'method': 'teacher_forced_transformers_sdpa_qk_fp32',
                'layers_zero_based': layers, 'head_aggregation': 'arithmetic_mean',
                'layer_aggregation': 'arithmetic_mean_of_four_sampled_layers',
                'scale_max_probability': maximum, 'probes': probes,
                'finish_reason': record['model_response']['finish_reason'],
                'grading': record.get('grading'), 'record_sha256': hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()}
    # Metadata is the completion marker; write it after all numeric/image files.
    prepare.write(destination / 'metadata.json', metadata)
    return metadata


def build_report(output, records, metadata, baseline=None):
    from collections import Counter
    rows = []
    transitions = Counter()
    transition_ids = {}
    groups = {}
    previous = {r['id']: r for r in prepare.read(baseline)['records']} if baseline else {}
    for record in records:
        info = metadata.get(record['id'])
        if not info:
            continue
        grade = record.get('grading', {}).get('final', {})
        hit = grade.get('hit')
        if record['id'] in previous and hit is not None:
            old_hit = previous[record['id']]['grading']['final']['hit']
            change = f'{"correct" if old_hit else "incorrect"}_to_{"correct" if hit else "incorrect"}'
            transitions[change] += 1
            transition_ids.setdefault(change, []).append(record['id'])
        case = output / record['id']
        parts = [f'<h1>{html.escape(record["id"])}</h1>',
                 '<p>Teacher-forced attention replay. Red = higher within-question attention; not causal importance. '
                 'One scale across all images/probes in this question. Layer/head means; see metadata and raw NPZ.</p>',
                 f'<p>Correct: {hit}; finish: {html.escape(info["finish_reason"])}</p>',
                 f'<pre>{html.escape(record["question"])}</pre>',
                 f'<pre>{html.escape(json.dumps(record["options"], ensure_ascii=False, indent=2))}</pre>',
                 f'<p>Reference: {html.escape(str(record["reference_answer"]))}</p>']
        for index in range(len(next(iter(info['probes'].values()))['images'])):
            parts.append(f'<h2>Image {index+1}</h2><div class="grid"><figure><img src="original_image_{index+1}.jpg"><figcaption>Original</figcaption></figure>')
            for name, probe in info['probes'].items():
                image = probe['images'][index]
                parts.append(f'<figure><img src="{image["file"]}"><figcaption>{name}<br>Image attention mass: {image["image_attention_mass"]:.6f}</figcaption></figure>')
            parts.append('</div>')
        parts += ['<h2>Model response</h2>', f'<pre>{html.escape(record["model_response"]["raw"])}</pre>',
                  '<p><a href="metadata.json">Metadata</a> · <a href="attention.npz">Raw attention</a></p>']
        style = '<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:16px;background:#f4f6fa;color:#19263b}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0;background:white;padding:12px}img{max-width:100%;max-height:800px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccc}</style>'
        (case / 'index.html').write_text('<!doctype html>' + style + ''.join(parts), encoding='utf-8')
        label = 'correct' if hit else 'incorrect' if hit is not None else 'ungraded'
        for name, probe in info['probes'].items():
            for group in [label, 'finish_' + info['finish_reason'], 'subject_' + record['subject']]:
                groups.setdefault(group, {}).setdefault(name, []).append(probe['total_image_attention_mass'])
        rows.append(f'<tr><td><a href="{record["id"]}/index.html">{html.escape(record["id"])}</a></td><td>{label}</td><td>{info["finish_reason"]}</td></tr>')
    group_statistics = {group: {name: {'count': len(values), 'mean_image_attention_mass': sum(values) / len(values),
                               'min': min(values), 'max': max(values)} for name, values in probes.items()}
                        for group, probes in groups.items()}
    summary = {'expected': len(records), 'heatmaps_complete': len(metadata), 'transitions_from_seed3407': dict(transitions),
               'transition_ids': transition_ids, 'attention_by_group': group_statistics,
               'limits': ['Attention is descriptive, not causal importance.',
                          'Replay uses Transformers SDPA; generation uses vLLM. Numerical equivalence is not asserted.',
                          'Four sampled decoder layers and three query anchors are not whole-response attention.',
                          'Final-answer anchor is absent when no explicit final answer is present.',
                          'Last-token probe for length-terminated responses is not a final-answer probe.']}
    prepare.write(output / 'analysis_summary.json', summary)
    graded = [r for r in records if r.get('grading', {}).get('final', {}).get('hit') is not None]
    correct = sum(bool(r['grading']['final']['hit']) for r in graded)
    truncated = sum(r['model_response']['finish_reason'] == 'length' for r in records)
    report = ['# Baseline 시각 토큰 분석', '',
              f'- Heatmap 완료: {len(metadata)}/{len(records)}문항',
              f'- 정답: {correct}/{len(graded)}' + (f' ({correct / len(graded) * 100:.2f}%)' if graded else ''),
              f'- 출력 상한 종료: {truncated}문항', '',
              '## seed 3407 대비 문항 변화', '',
              '| 변화 | 문항 수 |', '|---|---:|']
    report += [f'| {key} | {value} |' for key, value in transitions.items()]
    report += ['', '## 정오답별 이미지 attention 비중', '',
               '| 그룹 | 관찰 위치 | 문항 수 | 평균 이미지 attention 비중 |', '|---|---|---:|---:|']
    for group in ['correct', 'incorrect', 'finish_length']:
        report += [f'| {group} | {name} | {value["count"]} | {value["mean_image_attention_mass"]:.6f} |'
                   for name, value in group_statistics.get(group, {}).items()]
    report += ['', '## 해석 범위', '',
               '새 seed의 생성 응답을 고정한 teacher-forcing 재생이다. 생성은 vLLM, 분석은 Transformers SDPA이므로 생성 당시 attention의 직접 기록으로 부르지 않는다.',
               '문항별로 네 decoder layer의 head 평균을 다시 평균했다. 모든 원시 layer별 배열은 attention.npz에 보존한다.',
               '이미지 전체로 향하는 attention의 비중과 이미지 안의 상대적인 분포를 구분한다. 정답 그룹의 비중이 높거나 낮다는 관찰만으로 인과관계를 주장하지 않는다.',
               '색상 범위는 한 문항의 모든 이미지·관찰 위치에 공통이며, 서로 다른 문항의 색 진하기를 직접 비교하지 않는다. 밝은 테두리/공백은 attention sink일 수 있으므로 핵심 의미 영역으로 단정하지 않는다.',
               'Final answer가 없으면 해당 probe가 없다. length 종료의 마지막 probe는 미완성 응답 끝부분이다.',
               '이 보고서는 자동 집계다. 900문항의 이미지 판독·추론 오류 원인을 전수 수동 판정한 결과가 아니다.', '',
               '[문항별 원본·heatmap·전체 응답 보기](index.html)']
    (output / 'analysis_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (output / 'index.html').write_text('<!doctype html>' + style + '<h1>MMMU visual token analysis</h1>'
        + f'<p>Heatmaps: {len(metadata)}/{len(records)}. '
        + 'Teacher-forced replay · four decoder layers · mean over heads.</p>'
        + '<p>Attention indicates internal weighting, not proof of which region caused an answer.</p>'
        + '<input id="filter" placeholder="Filter question ID or correct/incorrect" style="width:70%;padding:12px">'
        + '<table><thead><tr><th>Question</th><th>Grade</th><th>Termination</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
        + '<script>document.getElementById("filter").oninput=e=>document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-cache', required=True)
    parser.add_argument('--dataset-cache', required=True)
    parser.add_argument('--baseline-results')
    parser.add_argument('--pilot', action='store_true', help='Single/multi-image and longest-context old records; no new generation.')
    parser.add_argument('--chunk-size', type=int, default=256)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('chunk-size must be positive')
    config = prepare.config(args.config)
    prepare.runtime_check(config)
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if transformers.__version__ != '4.57.0':
        raise ValueError('Attention adapter is pinned to Transformers 4.57.0')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = evaluate.run_lock(output / '.analysis.lock')
    records = prepare.read(args.results)['records']
    if args.pilot:
        short = [r for r in records if r['model_response']['output_tokens'] < 512]
        single = next(r for r in short if len(r['image_fields']) == 1)
        multiple = next(r for r in short if len(r['image_fields']) > 1)
        longest = max(records, key=lambda r: r['model_response']['input_tokens'] + r['model_response']['output_tokens'])
        records = list({r['id']: r for r in [single, multiple, longest]}.values())
    else:
        evaluate.check_records(records, config)
        if any('visual_replay' not in r for r in records):
            raise ValueError('Production replay requires exact generation token IDs for all 900 records')
    datasets, data_rows = prepare.load_data(config, args.dataset_cache, offline=True)
    lookup = {r['id']: r for r in data_rows}
    model_path = prepare.snapshot(config, args.model_cache)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                        attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    fingerprint = hashlib.sha256(json.dumps({'config': config, 'results_sha256': prepare.digest(args.results),
        'script_sha256': prepare.digest(__file__), 'pilot': args.pilot, 'chunk_size': args.chunk_size}, sort_keys=True).encode()).hexdigest()
    manifest = {'fingerprint': fingerprint, 'created_at': prepare.now(), 'pilot': args.pilot, 'expected': len(records),
                'model_revision': config['model']['revision'], 'transformers': transformers.__version__,
                'dtype': 'bfloat16', 'attention_recompute_dtype': 'float32', 'chunk_size': args.chunk_size}
    if (output / 'manifest.json').exists() and prepare.read(output / 'manifest.json')['fingerprint'] != fingerprint:
        raise ValueError('Analysis fingerprint changed; choose a new output directory')
    prepare.write(output / 'manifest.json', manifest)
    completed = {}
    for record in records:
        case = output / record['id']
        marker = case / 'metadata.json'
        if marker.exists():
            info = prepare.read(marker)
            if info['fingerprint'] != fingerprint:
                raise ValueError('Stale case metadata')
            if (case / 'attention.npz').is_file() and all((case / im['file']).is_file() for p in info['probes'].values() for im in p['images']):
                completed[record['id']] = info
                continue
        started = time.perf_counter()
        row = lookup[record['id']]
        source = datasets[row['subject']][row['dataset_index']]
        originals, encoded, rendered = prepare_item(row, source, config, processor)
        if rendered != record['rendered_chat']:
            raise ValueError('Rendered prompt differs from generation')
        if args.pilot:
            ids = processor.tokenizer.encode(record['model_response']['raw'], add_special_tokens=False)
            token_source = 'pilot_reencoded_old_response_not_exact_generation_ids'
        else:
            saved = record['visual_replay']
            if saved['prompt_token_ids'] != encoded['input_ids'][0].tolist() or saved['image_grid_thw'] != encoded['image_grid_thw'].tolist():
                raise ValueError('Replay prompt token IDs/grid differ from generation')
            ids = saved['output_token_ids']
            if len(ids) != record['model_response']['output_tokens']:
                raise ValueError('Output token count mismatch')
            token_source = saved['source']
        maps, positions, layers, shapes = replay(model, encoded, ids, processor.tokenizer, args.chunk_size)
        info = save_item(case, record, originals, maps, positions, layers, shapes, fingerprint, token_source)
        completed[record['id']] = info
        prepare.write(output / 'progress.json', {'state': 'running', 'completed': len(completed), 'expected': len(records),
                                              'last_id': record['id'], 'updated_at': prepare.now()})
        print(f'Heatmaps {len(completed)}/{len(records)} {record["id"]} seconds={time.perf_counter()-started:.2f}', flush=True)
        torch.cuda.empty_cache()
    summary = build_report(output, records, completed, args.baseline_results)
    if summary['heatmaps_complete'] != len(records):
        raise ValueError('Incomplete heatmap coverage')
    prepare.write(output / 'progress.json', {'state': 'complete', 'completed': len(completed), 'expected': len(records), 'updated_at': prepare.now()})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
