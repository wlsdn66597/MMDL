"""Export a small frozen pilot from the OLD environment, without GPU inference."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OLD = Path('/home/rlatjswo/mmdl_test')
sys.path.insert(0, str(OLD / 'src'))
import prepare
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info


def digest_bytes(x):
    return hashlib.sha256(x).hexdigest()


def tensor_hash(x):
    return digest_bytes(x.detach().cpu().contiguous().numpy().tobytes())


def main():
    out = ROOT / 'inputs'
    if (out / 'frozen.json').exists():
        raise RuntimeError('Frozen input bundle already exists; do not overwrite.')
    out.mkdir(exist_ok=True)
    c = prepare.read('/home/rlatjswo/mmmu_multiseed_20261011/protocol/config.seed3407.json')
    base = Path('/home/rlatjswo/mmmu_visual_tokens_seed42_20261010/baseline_seed42')
    records = prepare.read_jsonl(base / 'predictions.jsonl')
    selected = []
    def add(r, reason):
        if r['id'] not in [v[0]['id'] for v in selected]:
            selected.append((r, reason))
    for n in range(1, 6):
        group = [r for r in records if len(r['image_fields']) == n]
        add(max(group, key=lambda r: r['model_response']['input_tokens']), f'longest input with {n} images')
    add(next(r for r in records if r['id'] == 'validation_Math_13'), 'previously inspected geometry example')
    add(min((r for r in records if r['question_type'] == 'open'), key=lambda r: r['model_response']['output_tokens']), 'short open answer')
    add(max(records, key=lambda r: (r['model_response']['finish_reason'] == 'length', r['model_response']['output_tokens'])), 'previous output reached length cap')
    for subject in ('Accounting', 'Art', 'Computer_Science', 'Clinical_Medicine', 'Physics', 'History'):
        if len(selected) >= 12:
            break
        add(min((r for r in records if r['subject'] == subject), key=lambda r:r['model_response']['output_tokens']), 'short response / subject diversity')
    assert len(selected) == 12
    datasets, _ = prepare.load_data(c, '/home/rlatjswo/mmdl-cache/huggingface/datasets', offline=True)
    cache = '/home/rlatjswo/mmdl-cache/huggingface/hub'
    model = prepare.snapshot(c, cache)
    judge = prepare.snapshot(c, cache, judge=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    items = []
    for record, reason in selected:
        raw = datasets[record['subject']][record['dataset_index']]
        assert raw['id'] == record['id']
        images = [raw[f].convert('RGB') for f in record['image_fields']]
        paths, hashes = [], []
        for i, im in enumerate(images):
            p = out / f'{record["id"]}_{i}.png'
            im.save(p)
            paths.append(str(p.relative_to(ROOT)))
            hashes.append(digest_bytes(im.tobytes()))
        messages = [{'role':'user','content':[dict(type='image', image=im, **c['vision']) for im in images] + [{'type':'text', 'text':record['input_prompt']}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed, _, kwargs = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size, return_video_kwargs=True, return_video_metadata=True)
        kwargs['do_resize'] = False
        encoded = processor(text=[rendered], images=processed, return_tensors='pt', **kwargs)
        token_ids = encoded['input_ids'][0].tolist()
        grid = encoded['image_grid_thw'].tolist()
        assert rendered == record['rendered_chat'], record['id']
        assert token_ids == record['visual_replay']['prompt_token_ids'], record['id']
        assert grid == record['visual_replay']['image_grid_thw'], record['id']
        item = dict(id=record['id'], subject=record['subject'], reason=reason, image_paths=paths,
                    rgb_sha256=hashes, input_prompt=record['input_prompt'], rendered_chat=rendered,
                    prompt_token_ids=token_ids, image_grid_thw=grid,
                    pixel_values_sha256=tensor_hash(encoded['pixel_values']),
                    pixel_values_shape=list(encoded['pixel_values'].shape),
                    processed_rgb_sha256=[digest_bytes(im.tobytes()) for im in processed],
                    processed_image_sizes_wh=[list(im.size) for im in processed],
                    old_output_tokens=record['model_response']['output_tokens'],
                    old_finish_reason=record['model_response']['finish_reason'])
        items.append(item)
        print('FROZEN', item['id'], len(images), len(token_ids), flush=True)
    attempts = prepare.read_jsonl(base / 'judge_attempts.jsonl')
    attempts = list({r['id']:r for r in attempts}.values())
    attempts.sort(key=lambda r:r['input_tokens'])
    selected_judges = [attempts[i] for i in (0, len(attempts)//3, len(attempts)*2//3, len(attempts)-1)]
    tokenizer = AutoTokenizer.from_pretrained(judge, local_files_only=True)
    judges = []
    for r in selected_judges:
        msgs = [{'role':'user', 'content':r['input_prompt']}]
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert rendered == r['rendered_chat'] and len(tokens) == r['input_tokens']
        judges.append(dict(id=r['id'], input_prompt=r['input_prompt'], rendered_chat=rendered, prompt_token_ids=tokens))
    prepare.write(out / 'frozen.json', dict(config=c, model_path=str(model), judge_path=str(judge),
                  selection_scope='Purposive 12-item pilot, image counts 1..5; not a representative accuracy sample.',
                  source_predictions_sha256=prepare.digest(base / 'predictions.jsonl'),
                  source_judge_attempts_sha256=prepare.digest(base / 'judge_attempts.jsonl'),
                  old_python=sys.executable, items=items, judges=judges))


if __name__ == '__main__':
    main()
