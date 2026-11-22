#!/usr/bin/env python3
"""Evaluate the base or a merged Qwen3-VL model with the team's MMMU-Pro recipe."""

import argparse
import ast
import asyncio
import hashlib
import importlib.metadata
import io
import json
import os
import re
import statistics
import string
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT = Path(__file__).resolve()
HF_HOME = Path(os.environ.get('HF_HOME', SCRIPT.parent / '.cache/huggingface')).expanduser().resolve()
DATASET = ('MMMU/MMMU_Pro', '563f3e84bb3b90893083a1f039cfa13077f2302b')
SUBSET = 'standard (10 options)'
SPLIT_SIZE = 1730
RECORDS_SHA256 = 'c967b5dff25a55e2280f359efed34441f95d5ed8647c80ee2ecd6a5388238ce0'
BASE = ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17')
JUDGE = ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218')
SEED = 3407
MAX_TOKENS = 32768
CONTEXT = 64000
SEQS = 16
MIN_PIXELS, MAX_PIXELS = 1003520, 4014080
LOOP_WINDOW, LOOP_N = 4096, 12
LOOP_THRESHOLD, LOOP_CONSECUTIVE = 0.5, 2
FINAL_FIELD = re.compile(r'final\s*answer\s*(?:is)?\s*[:：]\s*(.*)', re.I)

# The ZIP uses this prompt even on the ten-option subset. Keep it to reproduce the recipe.
JUDGE_PROMPT = (
    'You are an AI assistant who will help me to match an answer with several options of a single-choice question. '
    'You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. '
    'If the meaning of all options are significantly different from the answer, output Z. '
    'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
    'Example 1: \nQuestion: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
    'Answer: a cute teddy bear\nYour output: A\n'
    'Example 2: \nQuestion: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
    'Answer: Spider\nYour output: Z\n'
    'Example 3: \nQuestion: {}?\nOptions: {}\nAnswer: {}\nYour output: '
)


def write_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def json_line(row):
    return json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'


def record_runtime(run_dir):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 and the repository requirements.txt.')
    requirements = SCRIPT.parent.parent / 'requirements.txt'
    packages = {}
    for line in requirements.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, expected = line.split('==', 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f'{name} is missing. Install the repository requirements.txt.') from exc
        if actual.split('+', 1)[0] != expected:
            raise RuntimeError(f'{name} is {actual}; the evaluation environment expects {expected}.')
        packages[name] = actual
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader',
    ], text=True).strip()
    runtime = {
        'python': sys.version,
        'packages': packages,
        'gpu_driver_memory': gpu,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'evaluator_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        'records_sha256': RECORDS_SHA256,
    }
    path = run_dir / 'runtime.json'
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8')) != runtime:
            raise RuntimeError(f'Runtime differs from the saved run in {run_dir}. Use another --run-name.')
    else:
        write_text(path, json_line(runtime))


def read_predictions(path, records):
    if not path.exists():
        return {}
    predictions = {}
    with path.open('rb+') as stream:
        while line := stream.readline():
            if not line.endswith(b'\n'):
                stream.truncate(stream.tell() - len(line))
                print('Removed an unfinished prediction line from the previous run.', flush=True)
                break
            prediction = json.loads(line)
            rid = prediction['id']
            if rid not in records or rid in predictions:
                raise RuntimeError(f'Unexpected or duplicate prediction ID in {path}: {rid}')
            predictions[rid] = prediction
    return predictions


def weight_files(model):
    index = model / 'model.safetensors.index.json'
    if index.is_file():
        return sorted(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()))
    return ['model.safetensors']


def model_snapshot(spec):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo, revision = spec
    try:
        cached = Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError:
        cached = None
    required = ['config.json', 'tokenizer_config.json', 'tokenizer.json']
    if spec == BASE:
        required.append('preprocessor_config.json')
    if cached is not None and all((cached / file).is_file() for file in required):
        weights = weight_files(cached)
        if weights and all((cached / file).is_file() for file in weights):
            return cached
    print(f'Preparing model {repo}@{revision[:8]}', flush=True)
    return Path(snapshot_download(repo, revision=revision, allow_patterns=[
        '*.json', '*.safetensors', '*.txt', '*.model', '*.jinja',
    ]))


def bind_checkpoint(checkpoint, run_dir):
    if not checkpoint.is_dir() or (checkpoint / 'adapter_config.json').exists():
        raise RuntimeError(f'--checkpoint needs the extracted merged Qwen3-VL model folder: {checkpoint}')
    required = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json']
    weights = weight_files(checkpoint)
    missing = [name for name in required + weights
               if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
    if missing or not weights:
        raise RuntimeError(f'Incomplete merged checkpoint at {checkpoint}. Missing or empty files: {missing}')
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen3_vl' or config.get('quantization_config'):
        raise RuntimeError(f'Expected an unquantized Qwen3-VL checkpoint at {checkpoint}.')

    print(f'Verifying merged checkpoint: {checkpoint}', flush=True)
    identity = {}
    names = set(required + weights)
    for pattern in ('*.json', '*.jinja', '*.model', 'merges.txt'):
        names.update(file.name for file in checkpoint.glob(pattern) if file.is_file())
    for name in sorted(names):
        with (checkpoint / name).open('rb') as file:
            identity[name] = hashlib.file_digest(file, 'sha256').hexdigest()
    path = run_dir / 'checkpoint.json'
    if path.is_file():
        if json.loads(path.read_text(encoding='utf-8'))['files_sha256'] != identity:
            raise RuntimeError(f'Checkpoint differs from the saved run in {run_dir}. Use another --run-name.')
    else:
        if (run_dir / 'predictions.jsonl').exists():
            raise RuntimeError(f'The saved predictions in {run_dir} have no checkpoint record. Use another --run-name.')
        write_text(path, json_line({'path': str(checkpoint), 'files_sha256': identity}))


def load_dataset():
    from huggingface_hub import hf_hub_download
    from pyarrow import concat_tables
    from pyarrow.parquet import read_table

    repo, revision = DATASET
    files = [hf_hub_download(
        repo, filename=f'{SUBSET}/test-{i:05d}-of-00002.parquet',
        revision=revision, repo_type='dataset',
    ) for i in range(2)]
    table = concat_tables([read_table(path) for path in files])
    if len(table) != SPLIT_SIZE:
        raise RuntimeError(f'Expected {SPLIT_SIZE} questions, got {len(table)}.')
    return table


def question_prompt(question, options):
    letters = '\n'.join(f'{letter}. {value}' for letter, value in options.items())
    return (f'Question: {question}\nOptions:\n{letters}\n'
            'Please select the correct answer from the options above.\n'
            'On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.')


def build_records(table):
    columns = {f'image_{i}': table[f'image_{i}'] for i in range(1, 8) if f'image_{i}' in table.column_names}
    records = {}
    for index, row in enumerate(table.select(['id', 'question', 'options', 'answer']).to_pylist()):
        rid = row['id']
        if rid in records:
            raise RuntimeError(f'Duplicate dataset ID: {rid}')
        choices = row['options']
        choices = ast.literal_eval(choices) if isinstance(choices, str) else choices
        if not isinstance(choices, (list, tuple)) or not 1 <= len(choices) <= len(string.ascii_uppercase):
            raise RuntimeError(f'{rid}: invalid options in the pinned dataset: {choices!r}')
        options = dict(zip(string.ascii_uppercase, choices))
        gold = str(row['answer']).strip()
        fields = []
        for key, column in columns.items():
            image = column[index]
            if image.is_valid and image['bytes'].is_valid and image['bytes'].as_buffer().size:
                fields.append(key)
        if gold not in options:
            raise RuntimeError(f'{rid}: answer {gold!r} is outside the available options {list(options)}; '
                               f'{len(fields)} images present.')
        source_split, subject_id = rid.split('_', 1)
        records[rid] = dict(id=rid, dataset_index=index, image_fields=fields, source_split=source_split,
                            subject=subject_id.rsplit('_', 1)[0], question=row['question'], options=options,
                            reference_answer=gold, input_prompt=question_prompt(row['question'], options))
    return records, columns


def render_chat(record, processor):
    content = [dict(type='image') for _ in record['image_fields']]
    content.append(dict(type='text', text=record['input_prompt']))
    return processor.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                         add_generation_prompt=True)


def prepare_input(record, columns, prompt):
    if not record['image_fields']:
        return {'prompt': prompt}

    from PIL import Image, ImageOps
    from qwen_vl_utils import process_vision_info

    index = record['dataset_index']
    content = []
    for field in record['image_fields']:
        with Image.open(io.BytesIO(columns[field][index]['bytes'].as_py())) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        content.append(dict(type='image', image=image, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS))
    messages = [dict(role='user', content=content + [dict(type='text', text=record['input_prompt'])])]
    images, _, kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True,
                                           return_video_metadata=True)
    return dict(prompt=prompt, multi_modal_data={'image': images},
                mm_processor_kwargs={**(kwargs or {}), 'do_resize': False})


def ngram_repeat_rate(text):
    words = text.split()
    if len(words) < LOOP_N + 50:
        return 0.0
    grams = len(words) - LOOP_N + 1
    unique = {tuple(words[i:i + LOOP_N]) for i in range(grams)}
    return 1.0 - len(unique) / grams


async def generate(engine, prompt, params, rid):
    parts = []
    output_tokens = streak = 0
    input_tokens = finish = loop_at = None
    next_check = LOOP_WINDOW
    start = time.monotonic()
    async for event in engine.generate(prompt, params, rid):
        if input_tokens is None and event.prompt_token_ids is not None:
            input_tokens = len(event.prompt_token_ids)
        output = event.outputs[0]
        parts.append(output.text)
        output_tokens += len(output.token_ids)
        if output.finish_reason is not None:
            finish = output.finish_reason
        if output_tokens >= next_check and not event.finished:
            text = ''.join(parts)
            window = text[-int(LOOP_WINDOW * len(text) / output_tokens):]
            streak = streak + 1 if ngram_repeat_rate(window) >= LOOP_THRESHOLD else 0
            next_check += LOOP_WINDOW
            if streak >= LOOP_CONSECUTIVE:
                loop_at = output_tokens
                await engine.abort(rid)
                break
        if event.finished:
            break
    if loop_at is not None:
        status, finish = 'loop_stopped', 'abort'
    elif finish == 'length':
        status = 'output_token_limit'
    elif finish == 'stop':
        status = 'normal_stop'
    else:
        raise RuntimeError(f'{rid}: generation did not finish: {finish}')
    raw = ''.join(parts)
    return dict(raw=raw, processed=raw.split('</think>')[-1], finish_reason=finish, input_tokens=input_tokens,
                output_tokens=output_tokens, output_limit_reached=finish == 'length', generation_status=status,
                loop_stop_at_tokens=loop_at, elapsed_seconds=time.monotonic() - start)


async def infer(run_dir, smoke, checkpoint):
    from transformers import AutoProcessor
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind

    print(f'Preparing {DATASET[0]} {SUBSET} / test', flush=True)
    records, columns = build_records(load_dataset())
    image_counts = Counter(len(r['image_fields']) for r in records.values())
    option_counts = Counter(len(r['options']) for r in records.values())
    print(f'Dataset: {len(records)} questions, {image_counts[0]} text-only, '
          f'option counts {dict(sorted(option_counts.items()))}', flush=True)
    records_digest = hashlib.sha256(json_line(list(records.values())).encode('utf-8')).hexdigest()
    if records_digest != RECORDS_SHA256:
        raise RuntimeError(f'The MMMU-Pro evaluation records differ from the reference: {records_digest}')
    print(f'Verified reference records SHA-256: {records_digest}', flush=True)
    if smoke:
        records = dict(list(records.items())[:8])
    record_path = run_dir / 'records.json'
    rows = list(records.values())
    if not record_path.is_file():
        write_text(record_path, json_line(rows))
    elif json.loads(record_path.read_text(encoding='utf-8')) != rows:
        raise RuntimeError(f'Saved questions differ from the current dataset or prompt: {record_path}')
    path = run_dir / 'predictions.jsonl'
    completed = read_predictions(path, records)
    todo = [rid for rid in records if rid not in completed]
    print(f'Inference: {len(completed)}/{len(records)} saved, {len(todo)} remaining.', flush=True)
    if not todo:
        return

    model = checkpoint if checkpoint is not None else model_snapshot(BASE)
    print(f'Model: {model}', flush=True)
    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
    prompts = {rid: render_chat(records[rid], processor) for rid in todo}
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=str(model), dtype='bfloat16', seed=SEED, max_model_len=CONTEXT, gpu_memory_utilization=0.9,
        max_num_seqs=SEQS, max_num_batched_tokens=4096, enforce_eager=True,
        limit_mm_per_prompt={'image': max(image_counts)},
        mm_processor_kwargs={'size': {'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}, 'do_resize': False},
    ))
    semaphore = asyncio.Semaphore(SEQS * 2)
    count = len(completed)
    start = time.monotonic()
    try:
        with path.open('a', encoding='utf-8') as output:
            async def one(rid):
                nonlocal count
                async with semaphore:
                    prompt = await asyncio.to_thread(prepare_input, records[rid], columns, prompts[rid])
                    params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
                                            presence_penalty=1.5, seed=SEED, max_tokens=MAX_TOKENS,
                                            stop_token_ids=[], output_kind=RequestOutputKind.DELTA)
                    response = await generate(engine, prompt, params, rid)
                    output.write(json_line(dict(id=rid, seed=SEED, model_response=response)))
                    output.flush()
                    count += 1
                    if count % 25 == 0 or count == len(records):
                        minutes = (time.monotonic() - start) / 60
                        print(f'Inference {count}/{len(records)} ({minutes:.1f} min this run)', flush=True)

            async with asyncio.TaskGroup() as tasks:
                for rid in todo:
                    tasks.create_task(one(rid))
    finally:
        engine.shutdown()


def option_text_match(payload, options):
    normalized = payload.strip().lower().rstrip('.')
    hits = [key for key, value in options.items() if str(value).strip().lower().rstrip('.') == normalized]
    return hits[0] if len(hits) == 1 else None


def extract_answer(raw, options):
    # Multiple-choice parser from the teammate's evaluation ZIP.
    text = raw.split('</think>')[-1].replace('**', '').replace('__', '').replace('`', '')
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.M)
    for _ in range(2):
        text = re.sub(r'\\(?:boxed|text|mathrm|textbf)\{([^{}]*)\}', r'\1', text)
    text = text.replace('$', '')
    valid = set(options)
    lines = text.splitlines()
    payloads = []
    for i, line in enumerate(lines):
        match = FINAL_FIELD.match(line.strip().lstrip('>-*• ').strip())
        if match:
            payloads.append(match[1].strip() or next((line.strip() for line in lines[i + 1:] if line.strip()), ''))

    def field(payload):
        letters = set(re.findall(r'(?<![A-Za-z])\(?([A-Z])\)?(?![A-Za-z])', payload)) & valid
        if len(letters) == 1:
            return letters.pop(), 'final_field'
        if len(letters) > 1:
            return None, 'multiple_letters_in_final_field'
        answer = option_text_match(payload, options)
        return answer, 'final_field' if answer else 'final_field_unparsed'

    if payloads:
        answer, reason = field(payloads[-1])
        if answer is not None and any(field(previous)[0] not in (None, answer) for previous in payloads[:-1]):
            return None, 'conflicting_final_fields'
        return answer, reason
    text = text.strip()
    match = re.fullmatch(r'\(?([A-Z])\)?[\.\)]?', text)
    if match and match[1] in valid:
        return match[1], 'fallback_single_letter'
    match = re.fullmatch(r'\(?([A-Z])\)?[\.\):]\s+[^\n]{1,200}', text)
    if match and match[1] in valid:
        return match[1], 'fallback_letter_with_text'
    letters = set(re.findall(r'\(([A-Z])\)', text)) & valid
    if len(letters) == 1:
        return letters.pop(), 'fallback_paren_letter'
    answer = option_text_match(text, options)
    return answer, 'fallback_option_text' if answer else 'no_final_field'


def parse_judge(raw, valid):
    text = raw.split('</think>')[-1].strip()
    for pattern in (r'your output\s*[:：]\s*\(?([A-Z])\)?', r'output\s*[:：]\s*\(?([A-Z])\)?',
                    r'final answer\s*[:：]\s*\(?([A-Z])\)?'):
        matches = re.findall(pattern, text, flags=re.I)
        if matches:
            return matches[-1].upper()
    match = re.fullmatch(r'\(?([A-Z])\)?\.?', text)
    if match:
        return match[1]
    letters = [x for x in re.findall(r'(?<![A-Za-z])([A-Z])(?![A-Za-z])', text) if x in valid | {'Z'}]
    return letters[-1] if letters else None


def judge_responses(todo, records, predictions, path):
    if path.is_file():
        saved = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if [row['id'] for row in saved] != todo:
            raise RuntimeError(f'Saved judge results do not match the pending questions: {path}')
        return {row['id']: row['answer'] for row in saved}
    print(f'LLM judge: {len(todo)} questions', flush=True)
    if not todo:
        write_text(path, '')
        return {}

    from vllm import LLM, SamplingParams

    llm = LLM(model=str(model_snapshot(JUDGE)), dtype='bfloat16', max_model_len=32768,
              gpu_memory_utilization=0.92, seed=SEED)
    params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=4096, seed=SEED)
    conversations = []
    for rid in todo:
        record = records[rid]
        options = ' '.join(f'{key}. {value}' for key, value in record['options'].items())
        response = predictions[rid]['model_response']['processed']
        conversations.append([dict(role='user', content=JUDGE_PROMPT.format(record['question'], options, response))])
    outputs = llm.chat(conversations, params, chat_template_kwargs={'enable_thinking': False})
    decisions = []
    for rid, item in zip(todo, outputs, strict=True):
        raw = item.outputs[0].text
        decisions.append(dict(id=rid, answer=parse_judge(raw, set(records[rid]['options'])), judge_output=raw[-500:]))
    write_text(path, ''.join(map(json_line, decisions)))
    return {row['id']: row['answer'] for row in decisions}


def grade(run_dir, records, predictions):
    grades, pending = [], []
    for rid, prediction in predictions.items():
        response = prediction['model_response']
        status = response['generation_status']
        if status == 'loop_stopped':
            answer, reason = None, 'loop_stopped'
        elif status == 'output_token_limit' or response['output_limit_reached']:
            answer, reason = None, 'length'
        else:
            answer, reason = extract_answer(response['raw'], records[rid]['options'])
            if answer is None:
                pending.append(rid)
        grades.append(dict(id=rid, answer=answer, reason=reason, output_tokens=response['output_tokens'],
                           generation_status=status, reference_answer=records[rid]['reference_answer']))
    decisions = judge_responses(pending, records, predictions, run_dir / 'judge.jsonl')
    for result in grades:
        if result['id'] in decisions:
            result.update(answer=decisions[result['id']], reason='judge')
        result['verdict'] = 'correct' if result['answer'] == result['reference_answer'] else 'incorrect'
    write_text(run_dir / 'graded.jsonl', ''.join(map(json_line, grades)))
    correct = sum(result['verdict'] == 'correct' for result in grades)
    total = len(grades)
    statuses = dict(Counter(result['generation_status'] for result in grades))
    reasons = dict(Counter(result['reason'] for result in grades))
    lengths = [result['output_tokens'] for result in grades]
    summary = [f'MMMU-Pro {SUBSET} test: {correct}/{total} = {100 * correct / total:.2f}%',
               f'Generation status: {statuses}', f'Grading paths: {reasons}',
               f'Output tokens median: {statistics.median(lengths)}, over 2048: {sum(n > 2048 for n in lengths)}']
    for key in ('source_split', 'subject'):
        groups = defaultdict(Counter)
        for result in grades:
            groups[records[result['id']][key]][result['verdict']] += 1
        summary.append(f'By {key}')
        for name, counts in sorted(groups.items()):
            n = sum(counts.values())
            summary.append(f'  {name}: {counts["correct"]}/{n} = {100 * counts["correct"] / n:.1f}%')
    write_text(run_dir / 'summary.txt', '\n'.join(summary) + '\n')
    print(summary[0], flush=True)
    print(f'Saved {run_dir / "summary.txt"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Local folder with a merged Qwen3-VL checkpoint.')
    parser.add_argument('--run-name', help='Output name for this evaluation. Default: base or finetuned.')
    parser.add_argument('--smoke', action='store_true', help='Run the first 8 questions in a separate output directory.')
    parser.add_argument('--_infer', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_name is not None and not re.fullmatch(r'[\w-]+', args.run_name, re.ASCII):
        parser.error('--run-name must use only letters, numbers, underscores, or hyphens')
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    model_kind = 'base' if checkpoint is None else 'finetuned'
    tag = args.run_name or model_kind
    prefix = f'{model_kind}_{tag}' if args.run_name else model_kind
    mode = 'smoke' if args.smoke else 'eval'
    run_dir = SCRIPT.parent / 'outputs' / f'{prefix}_{mode}'
    os.environ.update(HF_HOME=str(HF_HOME), HF_HUB_CACHE=str(HF_HOME / 'hub'), HF_HUB_OFFLINE='0',
                      VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if args._infer:
        asyncio.run(infer(run_dir, args.smoke, checkpoint))
        return

    import fcntl

    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.parent / '.mmmu_pro_std10.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('An MMMU-Pro evaluation is already running.') from None
        record_runtime(run_dir)
        command = [sys.executable, '-u', str(SCRIPT), '--_infer']
        if checkpoint is not None:
            bind_checkpoint(checkpoint, run_dir)
            command += ['--checkpoint', str(checkpoint)]
        if args.run_name is not None:
            command += ['--run-name', args.run_name]
        if args.smoke:
            command.append('--smoke')
        # The inference process exits before the judge is loaded, releasing its VRAM.
        subprocess.run(command, check=True, pass_fds=(lock.fileno(),))
        records = {r['id']: r for r in json.loads((run_dir / 'records.json').read_text(encoding='utf-8'))}
        predictions = read_predictions(run_dir / 'predictions.jsonl', records)
        if predictions.keys() != records.keys():
            raise RuntimeError(f'Incomplete inference: {len(predictions)}/{len(records)}')
        grade(run_dir, records, predictions)


if __name__ == '__main__':
    main()
