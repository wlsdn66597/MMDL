"""One command: pinned inputs -> inference -> frozen parsers -> local judge -> scores."""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import string
import subprocess
import sys
import time

import prepare
import parsers

ROOT = prepare.ROOT


def run_lock(path):
    handle = path.open('a+b')
    try:
        if platform.system() == 'Windows':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('Another process is already writing this run.')
    return handle  # Retain until main exits; process termination releases the lock.


def prompt_for(row, c):
    options = row.get('options', [])
    options = ast.literal_eval(options) if isinstance(options, str) else options
    items = options.items() if isinstance(options,dict) else zip(string.ascii_uppercase,options)
    letters = ''.join(f'{letter}. {value}\n' for letter,value in items) if row['question_type']=='multiple-choice' else ''
    hint = row.get('hint')
    present = hint is not None and not (isinstance(hint,float) and math.isnan(hint))
    optional_hint = c['prompts']['optional_hint'].format(hint=hint) if present else ''
    base = c['prompts'][row['question_type']].format(optional_hint=optional_hint,
        question=row['question'],lettered_options=letters).rstrip()
    return base + '\n' + c['prompts']['suffixes'][row['question_type']]


def data_row(row):
    options = row.get('options', [])
    options = ast.literal_eval(options) if isinstance(options,str) else options
    return {'id':row['id'], 'subject':row['subject'], 'dataset_config':row['subject'],
        'dataset_index':row['dataset_index'], 'question_type':row['question_type'],
        'question':row['question'], 'options':dict(zip(string.ascii_uppercase,options)) if row['question_type']=='multiple-choice' else {},
        'reference_answer':row['answer']}


def check_records(records, c, partial=False):
    ids = set()
    for r in records:
        if r['id'] in ids or r['subject'] not in c['dataset']['configs']:
            raise RuntimeError('Duplicate ID or unexpected subject.')
        if not r['id'].startswith('validation_' + r['subject'] + '_'):
            raise RuntimeError('Unexpected split/id.')
        ids.add(r['id'])
        if r['model_response']['finish_reason'] not in ('stop','length'):
            raise RuntimeError('Unsupported/error generation termination.')
        if r['question_type'] not in ('open','multiple-choice'):
            raise RuntimeError('Unexpected question type.')
        if r['model_response']['finish_reason']=='length' and r['model_response']['output_tokens'] != c['max_new_tokens']:
            raise RuntimeError('Unexpected truncation budget.')
        parsers.choice_info(r)
    counts = Counter(r['subject'] for r in records)
    if any(n>30 for n in counts.values()) or (not partial and (len(records)!=900 or any(counts[s]!=30 for s in c['dataset']['configs']))):
        raise RuntimeError('Expected every one of 30 subjects with 30 questions.')


def model_identity(args, c):
    if not args.checkpoint:
        return {'repo':c['model']['repo'], 'revision':c['model']['revision']}
    path = Path(args.checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError('--checkpoint must be a local merged checkpoint directory.')
    mc = prepare.validate_weights(path,'qwen3_vl')
    dtype = mc.get('text_config',{}).get('dtype',mc.get('text_config',{}).get('torch_dtype',mc.get('torch_dtype')))
    if dtype not in (None,'bfloat16'):
        raise RuntimeError('Checkpoint must retain BF16.')
    shards = list(path.glob('*.safetensors'))
    return {'path':str(path), 'config_sha256':prepare.digest(path/'config.json'),
            'weights_sha256':{p.name:prepare.digest(p) for p in sorted(shards)}}


def infer(args, c, run_dir, manifest):
    prepare.runtime_check(c)
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    datasets, rows = prepare.load_data(c,manifest['cache_paths']['dataset'],offline=True)
    model = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else prepare.snapshot(c,manifest['cache_paths']['model'])
    if args.checkpoint:
        base = prepare.snapshot(c,manifest['cache_paths']['model'])
        if prepare.read(base/'config.json')['text_config'] != prepare.read(model/'config.json')['text_config']:
            # Fine-tuning may change metadata, but architecture dimensions must match.
            for key in ('hidden_size','num_hidden_layers','num_attention_heads','vocab_size'):
                if prepare.read(base/'config.json')['text_config'][key] != prepare.read(model/'config.json')['text_config'][key]:
                    raise RuntimeError(f'Checkpoint architecture differs: {key}')
    processor = AutoProcessor.from_pretrained(model,local_files_only=True)
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    if factor != 32:
        raise RuntimeError('Unexpected patch/merge size.')
    path = run_dir/'predictions.jsonl'
    existing = prepare.read_jsonl(path)
    check_records(existing,c,partial=True)
    lookup = {r['id']:r for r in rows}
    for r in existing:
        if r['id'] not in lookup or any(r[k] != data_row(lookup[r['id']])[k] for k in ('question','options','reference_answer','subject')):
            raise RuntimeError('Existing predictions do not match pinned dataset.')
        if r['input_prompt'] != prompt_for(lookup[r['id']],c):
            raise RuntimeError('Existing prompt differs from frozen configuration.')
    done = {r['id'] for r in existing}
    pending = [r for r in rows if r['id'] not in done]
    if not pending:
        print('Inference complete; reusing all 900 saved responses.',flush=True)
        return
    rt = c['runtime']
    llm = LLM(model=str(model),dtype=rt['dtype'],quantization=None,
        tensor_parallel_size=rt['tensor_parallel_size'],max_model_len=rt['max_model_len'],
        gpu_memory_utilization=rt['gpu_memory_utilization'],max_num_seqs=rt['max_num_seqs'],
        max_num_batched_tokens=rt['max_num_batched_tokens'],enforce_eager=rt['enforce_eager'],
        limit_mm_per_prompt={'image':rt['max_images_per_prompt']},seed=c['sampling']['seed'],
        mm_processor_kwargs={'size':{'shortest_edge':c['vision']['min_pixels'],'longest_edge':c['vision']['max_pixels']},'do_resize':False})
    sample = SamplingParams(**c['sampling'],max_tokens=c['max_new_tokens'],stop_token_ids=[])
    for row in pending:
        start = time.perf_counter()
        r = data_row(row)
        rgb = []; fields = []
        source = datasets[row['subject']][row['dataset_index']]
        for i in range(1,8):
            field = f'image_{i}'
            if source.get(field) is not None:
                rgb.append(source[field].convert('RGB')); fields.append(field)
        if not rgb or len(rgb)>rt['max_images_per_prompt']:
            raise RuntimeError('Image count outside configured budget.')
        text = prompt_for(row,c)
        messages = [{'role':'user','content':[dict(type='image',image=im,**c['vision']) for im in rgb]+[{'type':'text','text':text}]}]
        rendered = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images, videos, kwargs = process_vision_info(messages,image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,return_video_metadata=True)
        if videos is not None:
            raise RuntimeError('Unexpected video input.')
        kwargs['do_resize'] = False
        encoded = processor(text=[rendered],images=images,return_tensors='pt',**kwargs)
        input_tokens = int(encoded['input_ids'].shape[-1])
        if input_tokens+c['max_new_tokens']>rt['max_model_len']:
            raise RuntimeError(f'Input+output exceeds context: {row["id"]}; no silent truncation.')
        result = llm.generate([{'prompt':rendered,'multi_modal_data':{'image':images},'mm_processor_kwargs':kwargs}],sample,use_tqdm=False)[0]
        output = result.outputs[0]
        if len(result.prompt_token_ids) != input_tokens:
            raise RuntimeError('Processor/vLLM input token lengths disagree.')
        r.update(image_fields=fields, original_image_sizes_wh=[list(im.size) for im in rgb],
            input_prompt=text, rendered_chat=rendered,
            visual_replay={'prompt_token_ids':list(result.prompt_token_ids),
                'output_token_ids':list(output.token_ids),
                'image_grid_thw':encoded['image_grid_thw'].tolist(),
                'processed_image_sizes_wh':[list(im.size) for im in images],
                'source':'vllm_exact_token_ids'},
            model_response={'raw':output.text,'processed':output.text.split('</think>')[-1].strip(),
                'finish_reason':output.finish_reason,'stop_reason':output.stop_reason,
                'output_tokens':len(output.token_ids),'input_tokens':input_tokens,
                'elapsed_seconds':time.perf_counter()-start,'output_limit_reached':output.finish_reason=='length',
                'generation_status':'output_token_limit' if output.finish_reason=='length' else 'normal_stop'})
        check_records([r],c,partial=True)
        prepare.append(path,r)
        print(f'Inference {len(done)+1}/900 {r["id"]} tokens={len(output.token_ids)}',flush=True)
        done.add(r['id'])


def import_saved_results(path, run_dir, c):
    saved_results = prepare.read(path)
    if saved_results['settings']['model']['revision'] != c['model']['revision'] or saved_results['settings']['dataset']['revision'] != c['dataset']['revision']:
        raise RuntimeError('Saved results revision mismatch.')
    if any(saved_results['settings']['sampling'][k]!=v for k,v in c['sampling'].items()) or saved_results['settings']['sampling']['max_new_tokens']!=c['max_new_tokens']:
        raise RuntimeError('Saved results sampling mismatch.')
    if any(saved_results['settings']['vision'][k]!=v for k,v in c['vision'].items()):
        raise RuntimeError('Saved results vision budget mismatch.')
    records = saved_results['records']
    check_records(records,c)
    for r in records:
        hint = None
        if r['input_prompt'].startswith('Hint: '):
            hint = r['input_prompt'].split('\nQuestion: ',1)[0][6:]
        if prompt_for(dict(r,hint=hint),c) != r['input_prompt']:
            raise RuntimeError(f'Saved results input prompt differs: {r["id"]}')
    prediction_path = run_dir/'predictions.jsonl'
    existing = prepare.read_jsonl(prediction_path)
    if len(existing)>900 or any(a['id']!=b['id'] or a['model_response']!=b['model_response'] for a,b in zip(existing,records)):
        raise RuntimeError('Run already contains different responses.')
    prediction_fields = ('id','subject','dataset_config','dataset_index','image_fields',
        'original_image_sizes_wh','question_type','question','options','reference_answer',
        'input_prompt','rendered_chat','model_response')
    for r in records[len(existing):]:
        prepare.append(prediction_path,{k:copy.deepcopy(r[k]) for k in prediction_fields})
    # Imported judge decisions must remain distinguishable from new model calls.
    existing_judges = prepare.read_jsonl(run_dir/'judge_attempts.jsonl')
    source_sha = prepare.digest(path)
    saved_ids = {a['id'] for a in existing_judges}
    for r in records:
        if parsers.grade_response(r)['status']=='pending_judge' and r['id'] not in saved_ids:
            j = r['grading'].get('local_qwen3_8b_judge')
            if j is None:
                raise RuntimeError(f'Missing saved judge output: {r["id"]}')
            prepare.append(run_dir/'judge_attempts.jsonl',{'id':r['id'],'attempt':j['attempt_count'],
                'raw':j['final_raw_output'],'saved_final_output':True,
                'input_sha256':judge_input_digest(r,c),'source_sha256':source_sha})
    return saved_results


def judge_prompt(row,c):
    choices,_ = parsers.choice_info(row)
    options = 'There are several options: \n' + ''.join(f'{k}. {v}\n' for k,v in choices.items())
    return c['judge']['prompt_template'].format(row['question'],options,row['model_response']['raw'].split('</think>')[-1].strip())


def judge_input_digest(row,c):
    return hashlib.sha256(judge_prompt(row,c).encode()).hexdigest()


def saved_judges(records, attempts, c):
    lookup = {r['id']:r for r in records}
    last = {}
    for a in attempts:
        if a['id'] not in lookup or a['input_sha256'] != judge_input_digest(lookup[a['id']],c):
            raise RuntimeError('Saved judge input no longer matches this response/configuration.')
        if a['id'] in last and a['attempt'] <= last[a['id']]['attempt']:
            raise RuntimeError('Duplicate/non-increasing judge attempt.')
        last[a['id']] = a
    return last


def summaries(records,c):
    check_records(records,c)
    if any(r['grading']['final']['status'] != 'graded' for r in records):
        raise RuntimeError('Incomplete grading; no final accuracy.')
    correct = sum(r['grading']['final']['hit'] for r in records)
    subjects = {}
    for subject in c['dataset']['configs']:
        items = [r for r in records if r['subject']==subject]
        n = sum(r['grading']['final']['hit'] for r in items)
        subjects[subject] = {'total':30,'correct':n,'incorrect':30-n,'accuracy':n/30}
    macro = sum(v['accuracy'] for v in subjects.values())/30
    if not math.isclose(macro,correct/900,abs_tol=1e-12):
        raise RuntimeError('Macro/micro arithmetic mismatch.')
    return {'total':900,'graded':900,'correct':correct,'incorrect':900-correct,
        'accuracy':macro,'accuracy_percent':macro*100,'aggregation':'unweighted mean of 30 subject accuracies',
        'by_subject':subjects,'method_counts':dict(Counter(r['grading']['final']['method'] for r in records)),
        'length_limit_incorrect':sum(r['model_response']['finish_reason']=='length' for r in records),
        'judge_count':sum(r['grading']['final']['method']=='local_qwen3_8b_judge_parser' for r in records),
        'judge_states':dict(Counter(r['grading']['final'].get('judge_parser_decision',{}).get('status') for r in records if 'judge_parser_decision' in r['grading']['final'])),
        'random_fallback':0,'exact_official_report_reproduction':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'src/config.json'))
    p.add_argument('--run-name',default='baseline')
    p.add_argument('--checkpoint',help='Local merged BF16 Qwen3-VL checkpoint; use pinned baseline when omitted.')
    p.add_argument('--model-cache'); p.add_argument('--dataset-cache')
    p.add_argument('--output-root',default=str(ROOT/'outputs'))
    p.add_argument('--selfcheck',action='store_true')
    p.add_argument('--import-results',help='Score preserved baseline_results.json responses without model inference.')
    p.add_argument('--_stage',choices=['infer','judge'],help=argparse.SUPPRESS)
    args = p.parse_args()
    c = prepare.config(args.config)
    parsers.selfcheck()
    if args.selfcheck:
        return
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_name):
        p.error('run-name must be a simple directory name.')
    if args.import_results and args.checkpoint:
        p.error('Saved results replay cannot be combined with a different checkpoint.')
    run_dir = Path(args.output_root).expanduser().resolve()/args.run_name
    run_dir.mkdir(parents=True,exist_ok=True)
    lock_handle = run_lock(run_dir/'.run.lock') if not args._stage else None
    if args._stage:
        manifest = prepare.read(run_dir/'manifest.json')
        if args._stage=='infer':
            infer(args,c,run_dir,manifest)
        else:
            import judge
            judge.run(c,run_dir,manifest)
        return
    paths = prepare.cache_paths(c,args.model_cache,args.dataset_cache)
    identity = model_identity(args,c)
    code = {str(f.relative_to(ROOT)):prepare.digest(f) for f in sorted((ROOT/'src').glob('*')) if f.is_file()}
    signature = hashlib.sha256(json.dumps({'configuration':c,'model_identity':identity,'code':code},sort_keys=True).encode()).hexdigest()
    manifest_path = run_dir/'manifest.json'
    if manifest_path.exists():
        manifest = prepare.read(manifest_path)
        if manifest['fingerprint'] != signature or manifest['mode'] != ('saved_response_scoring' if args.import_results else 'inference'):
            raise RuntimeError('Existing run has different code/config/checkpoint/mode; choose a new run-name.')
        manifest['cache_paths'] = paths
        prepare.write(manifest_path,manifest)
    else:
        try:
            commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError,FileNotFoundError):
            commit = None
        manifest = {'created_at':prepare.now(),'fingerprint':signature,'configuration':c,'code_sha256':code,
            'git_commit':commit,'model_identity':identity,'cache_paths':paths,
            'mode':'saved_response_scoring' if args.import_results else 'inference',
            'environment':{'python':sys.version,'platform':platform.platform()} if args.import_results else prepare.runtime_check(c)}
        if args.import_results:
            manifest['imported_source_sha256'] = prepare.digest(args.import_results)
        prepare.write(manifest_path,manifest)
    if args.import_results:
        if prepare.digest(args.import_results)!=manifest['imported_source_sha256']:
            raise RuntimeError('Saved results changed since this run began.')
        saved_results = import_saved_results(Path(args.import_results),run_dir,c)
    else:
        command = [sys.executable,str(Path(__file__).resolve()),'--config',str(Path(args.config).resolve()),
            '--run-name',args.run_name,'--output-root',str(run_dir.parent), '--model-cache',paths['model'], '--dataset-cache',paths['dataset']]
        if args.checkpoint:
            command += ['--checkpoint',str(Path(args.checkpoint).expanduser().resolve())]
        # Process exit releases CUDA allocations before the larger judge is loaded.
        subprocess.run(command+['--_stage','infer'],check=True)
    records = prepare.read_jsonl(run_dir/'predictions.jsonl')
    check_records(records,c)
    last = saved_judges(records,prepare.read_jsonl(run_dir/'judge_attempts.jsonl'),c)
    pending = []
    for r in records:
        r['grading'] = {'final':parsers.grade_response(r)}
        if r['grading']['final']['status']=='pending_judge':
            a = last.get(r['id'])
            if a and (a.get('saved_final_output') or parsers.grade_judge(r,a['raw'])['judge_parser_decision']['decision_parsed'] or a['attempt']>=c['judge']['max_attempts']):
                r['grading']['final'] = parsers.grade_judge(r,a['raw'])
            else:
                pending.append(r['id'])
    if pending:
        if args.import_results:
            raise RuntimeError('Saved results replay is missing terminal judge outputs; no model is called.')
        prepare.write(run_dir/'pending_judge.json',pending)
        subprocess.run(command+['--_stage','judge'],check=True)
        last = saved_judges(records,prepare.read_jsonl(run_dir/'judge_attempts.jsonl'),c)
        for r in records:
            if r['id'] in pending:
                r['grading']['final'] = parsers.grade_judge(r,last[r['id']]['raw'])
    summary = summaries(records,c)
    if args.import_results:
        original = {r['id']:r['grading']['final'] for r in saved_results['records']}
        def normalized_grade(grade):
            result = {key:grade[key] for key in ('hit','extracted_answer','grading_gold','method')}
            # Ignore naming-only version suffixes in imported grading metadata.
            result['method'] = re.sub(r'_v[0-9]+$', '', result['method'])
            return result
        differences = [r['id'] for r in records if normalized_grade(r['grading']['final'])!=normalized_grade(original[r['id']])]
        if differences:
            raise RuntimeError(f'Grading differs from saved results: {differences}')
        summary['saved_grade_mismatches'] = 0
        summary['model_calls'] = summary['judge_calls'] = 0
    prepare.write(run_dir/'results.json',{'schema_version':'mmmu-frozen-evaluation-v1','created_at':prepare.now(),
        'manifest':manifest,'summary':summary,'records':records})
    prepare.write(run_dir/'summary.json',summary)
    table = ['| Subject | Data Num | Correct | Acc |','|---|---:|---:|---:|']
    table += [f'| {s} | 30 | {v["correct"]} | {v["accuracy"]*100:.2f}% |' for s,v in summary['by_subject'].items()]
    table += [f'| Overall (macro avg) | 900 | {summary["correct"]} | {summary["accuracy_percent"]:.2f}% |']
    (run_dir/'scores.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='by_subject'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
