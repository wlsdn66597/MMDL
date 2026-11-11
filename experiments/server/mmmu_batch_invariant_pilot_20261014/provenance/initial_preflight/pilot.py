"""Isolated vLLM batch-invariance pilot. Never writes to the baseline environment."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ['VLLM_BATCH_INVARIANT'] = '1'  # Must precede every vLLM import.
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_LOG_BATCHSIZE_INTERVAL'] = '1'
os.environ['VLLM_LOG_STATS_INTERVAL'] = '1'
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['VLLM_CACHE_ROOT'] = str(ROOT / 'cache/vllm')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(ROOT / 'cache/torchinductor')
os.environ['TRITON_CACHE_DIR'] = str(ROOT / 'cache/triton')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def append(p, data):
    with Path(p).open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def sha(x):
    return hashlib.sha256(x).hexdigest()


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def prepared_vl(bundle):
    from PIL import Image
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(bundle['model_path'], local_files_only=True)
    requests, checks = {}, []
    for item in bundle['items']:
        images = [Image.open(ROOT / p).convert('RGB') for p in item['image_paths']]
        assert [sha(im.tobytes()) for im in images] == item['rgb_sha256']
        messages = [{'role':'user', 'content':[dict(type='image', image=im, **bundle['config']['vision']) for im in images] + [{'type':'text', 'text':item['input_prompt']}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed, _, kw = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size, return_video_kwargs=True, return_video_metadata=True)
        kw['do_resize'] = False
        encoded = processor(text=[rendered], images=processed, return_tensors='pt', **kw)
        actual = dict(rendered_chat=rendered, prompt_token_ids=encoded['input_ids'][0].tolist(),
                      image_grid_thw=encoded['image_grid_thw'].tolist(),
                      pixel_values_sha256=sha(encoded['pixel_values'].cpu().contiguous().numpy().tobytes()),
                      pixel_values_shape=list(encoded['pixel_values'].shape),
                      processed_rgb_sha256=[sha(im.tobytes()) for im in processed],
                      processed_image_sizes_wh=[list(im.size) for im in processed])
        differences = [k for k,v in actual.items() if v != item[k]]
        checks.append(dict(id=item['id'], passed=not differences, differences=differences,
                           prompt_tokens=len(actual['prompt_token_ids']), images=len(images),
                           actual_pixel_values_sha256=actual['pixel_values_sha256']))
        requests[item['id']] = dict(prompt=rendered, multi_modal_data={'image':processed}, mm_processor_kwargs=kw)
    return requests, checks


def prepared_judge(bundle):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(bundle['judge_path'], local_files_only=True)
    requests, checks = {}, []
    for item in bundle['judges']:
        msgs = [{'role':'user', 'content':item['input_prompt']}]
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        passed = ids == item['prompt_token_ids'] and rendered == item['rendered_chat']
        requests[item['id']] = dict(prompt_token_ids=ids)
        checks.append(dict(id=item['id'], passed=passed, prompt_tokens=len(ids)))
    return requests, checks


def preflight():
    bundle = read(ROOT / 'inputs/frozen.json')
    _, vl = prepared_vl(bundle)
    _, judge = prepared_judge(bundle)
    import vllm.envs as envs
    flag = bool(getattr(envs, 'VLLM_BATCH_INVARIANT', False))
    versions = {p:metadata.version(p) for p in ('vllm','torch','transformers','qwen-vl-utils','datasets','Pillow')}
    result = dict(created_at=now(), recognized_batch_invariant=flag, versions=versions,
                  input_checks=vl, judge_input_checks=judge,
                  passed=flag and all(x['passed'] for x in vl+judge))
    write(ROOT / 'provenance/input_parity.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result['passed']:
        raise RuntimeError('Input parity or recognized batch-invariance flag failed; GPU generation is blocked.')


STAGES = [
    ('vl_reference', 1, 256, 'forward'),
    ('vl_pair', 2, 256, 'forward'),
    ('vl_four_reverse', 4, 256, 'reverse'),
    ('vl_four_mix', 4, 256, 'mix'),
    ('vl_repeat', 1, 256, 'forward'),
    ('vl_resume_subset', 2, 256, 'subset'),
    ('vl_extended_reference', 1, 2048, 'extended'),
    ('vl_extended_pair', 2, 2048, 'extended_reverse'),
    ('judge_reference', 1, 4096, 'forward'),
    ('judge_four_reverse', 4, 4096, 'reverse'),
    ('judge_repeat', 1, 4096, 'forward'),
]


def stage_items(bundle, name, order):
    items = list(bundle['judges'] if name.startswith('judge') else bundle['items'])
    if order.startswith('extended'):
        wanted = ['validation_Math_13', bundle['items'][1]['id'], bundle['items'][4]['id']]
        items = [next(r for r in items if r['id'] == k) for k in wanted]
    if order.endswith('reverse'):
        items.reverse()
    elif order == 'mix':
        items = items[::2] + items[1::2]
    elif order == 'subset':
        items = items[1::2]
    return items


def run_stage(name):
    if (ROOT / 'PAUSE').exists():
        raise SystemExit(75)
    bundle = read(ROOT / 'inputs/frozen.json')
    assert read(ROOT / 'provenance/input_parity.json')['passed']
    _, batch, cap, order = next(s for s in STAGES if s[0] == name)
    judge = name.startswith('judge')
    chosen = stage_items(bundle, name, order)
    output_path = ROOT / 'outputs' / f'{name}.jsonl'
    output_path.parent.mkdir(exist_ok=True)
    existing = records(output_path)
    assert len({r['id'] for r in existing}) == len(existing)
    assert all(r['id'] in {x['id'] for x in chosen} for r in existing)
    done = {r['id'] for r in existing}
    if len(done) == len(chosen):
        return
    requests, checks = prepared_judge(bundle) if judge else prepared_vl(bundle)
    assert all(x['passed'] for x in checks)
    import torch
    import vllm.envs as envs
    from vllm import LLM, SamplingParams
    assert bool(envs.VLLM_BATCH_INVARIANT)
    c = bundle['config']
    spec = c['judge'] if judge else c
    rt = spec['runtime']
    kwargs = dict(model=bundle['judge_path' if judge else 'model_path'], dtype='bfloat16', quantization=None,
                  tensor_parallel_size=1, seed=3407, max_model_len=rt['max_model_len'],
                  gpu_memory_utilization=0.9, max_num_seqs=batch,
                  max_num_batched_tokens=rt['max_num_batched_tokens'], enforce_eager=True,
                  enable_prefix_caching=False, disable_log_stats=False)
    if not judge:
        kwargs.update(limit_mm_per_prompt={'image':5}, mm_processor_cache_gb=0,
                      mm_processor_kwargs={'size':{'shortest_edge':1003520, 'longest_edge':4014080}, 'do_resize':False})
    started = now()
    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    write(ROOT / 'provenance' / f'{name}_engine.json', dict(started_at=started, pid=os.getpid(),
          llm_kwargs=kwargs, sampling=dict(spec['sampling'], max_tokens=cap),
          environment_flag=os.environ['VLLM_BATCH_INVARIANT'], recognized_flag=bool(envs.VLLM_BATCH_INVARIANT),
          gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
          torch_cuda=torch.version.cuda, init_seconds=time.perf_counter()-t0,
          engine_config=repr(llm.llm_engine.vllm_config),
          note='Prefix and multimodal processor caches disabled in all pilot stages to expose fresh computation.'))
    for i in range(0, len(chosen), batch):
        group = [r for r in chosen[i:i+batch] if r['id'] not in done]
        if not group:
            continue
        if (ROOT / 'PAUSE').exists():
            raise SystemExit(75)
        append(ROOT / 'events.jsonl', dict(time=now(), stage=name, event='batch_started', ids=[r['id'] for r in group], max_num_seqs=batch))
        params = [SamplingParams(**spec['sampling'], max_tokens=cap, **({} if judge else {'stop_token_ids':[]})) for r in group]
        began = time.perf_counter()
        outputs = llm.generate([requests[r['id']] for r in group], params, use_tqdm=False)
        elapsed = time.perf_counter()-began
        for item, result in zip(group, outputs, strict=True):
            if list(result.prompt_token_ids) != item['prompt_token_ids']:
                raise RuntimeError(f'Engine prompt IDs differ for {item["id"]}')
            o = result.outputs[0]
            row = dict(id=item['id'], stage=name, seed=3407, token_ids=list(o.token_ids), text=o.text,
                       finish_reason=o.finish_reason, stop_reason=o.stop_reason,
                       input_tokens=len(result.prompt_token_ids), max_tokens=cap,
                       output_tokens=len(o.token_ids), batch_ids=[r['id'] for r in group],
                       batch_elapsed_seconds=elapsed, created_at=now())
            append(output_path, row)
            done.add(item['id'])
        append(ROOT / 'events.jsonl', dict(time=now(), stage=name, event='batch_completed', completed=len(done), expected=len(chosen)))
        print(name, len(done), '/', len(chosen), flush=True)


def summarize():
    bundle = read(ROOT / 'inputs/frozen.json')
    comparisons, counts = [], {}
    complete = True
    for name, batch, cap, order in STAGES:
        expected = {r['id'] for r in stage_items(bundle, name, order)}
        data = records(ROOT / 'outputs' / f'{name}.jsonl')
        counts[name] = {'completed':len(data), 'expected':len(expected)}
        if len(data) != len(expected) or {r['id'] for r in data} != expected:
            complete = False
        reference = 'judge_reference' if name.startswith('judge') else ('vl_extended_reference' if 'extended' in name else 'vl_reference')
        ref = {r['id']:r for r in records(ROOT / 'outputs' / f'{reference}.jsonl')}
        if name == reference:
            continue
        for r in data:
            if r['id'] not in ref:
                complete = False
                continue
            differences = [k for k in ('token_ids','text','finish_reason','stop_reason','input_tokens') if r[k] != ref[r['id']][k]]
            comparisons.append(dict(stage=name, id=r['id'], passed=not differences, differences=differences))
    result = dict(created_at=now(), complete=complete, passed=complete and bool(comparisons) and all(r['passed'] for r in comparisons),
                  stage_counts=counts, comparisons=comparisons,
                  limitations=['Purposive 12-item sample; no benchmark accuracy estimate.',
                  'Main output cap 256; extended subset cap 2048. Full 32768-token generations not tested.',
                  'Seed 3407 only; fixed checkpoint, one RTX 4090, BF16, eager mode.',
                  'Fresh-process repeats and a saved-completion subset are tested; forced in-flight crash recovery is not tested.',
                  'Old vLLM 0.11.0 and new vLLM output equality is not required or claimed.',
                  'Batch grouping and maximum concurrency are recorded; kernel-level scheduling is not independently traced.'])
    write(ROOT / 'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preflight','stage','summarize'])
    p.add_argument('--name')
    a = p.parse_args()
    if a.action == 'preflight':
        preflight()
    elif a.action == 'stage':
        run_stage(a.name)
    else:
        summarize()
