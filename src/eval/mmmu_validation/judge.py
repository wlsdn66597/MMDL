"""Local Qwen3-8B, official matching prompt, frozen judge-output parser."""
import hashlib
import random
import time

import prepare
import parsers


def run(c, run_dir, manifest):
    prepare.runtime_check(c)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from evaluate import judge_prompt, judge_input_digest, saved_judges
    records = prepare.read_jsonl(run_dir/'predictions.jsonl')
    attempts_path = run_dir/'judge_attempts.jsonl'
    last = saved_judges(records,prepare.read_jsonl(attempts_path),c)
    targets = set(prepare.read(run_dir/'pending_judge.json'))
    spec = c['judge']
    model = prepare.snapshot(c,manifest['cache_paths']['model'],judge=True)
    tokenizer = AutoTokenizer.from_pretrained(model,local_files_only=True)
    rt = spec['runtime']
    llm = LLM(model=str(model),dtype='bfloat16',quantization=None,seed=spec['sampling']['seed'],
        **{k:rt[k] for k in ('tensor_parallel_size','max_model_len','gpu_memory_utilization','max_num_seqs','max_num_batched_tokens','enforce_eager')})
    sampling = SamplingParams(**spec['sampling'],max_tokens=spec['max_new_tokens'])
    for row in records:
        if row['id'] not in targets:
            continue
        prompt = judge_prompt(row,c)
        messages = [{'role':'user','content':prompt}]
        token_ids = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=spec['enable_thinking'])
        rendered = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=spec['enable_thinking'])
        if len(token_ids)+spec['max_new_tokens']>rt['max_model_len']:
            raise RuntimeError(f'Judge context exceeded: {row["id"]}; input is never truncated.')
        previous = last.get(row['id'])
        start = previous['attempt']+1 if previous else 1
        rng = random.Random(spec['sampling']['seed']+int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16))
        for attempt in range(start,spec['max_attempts']+1):
            begin = time.perf_counter()
            response = llm.generate([{'prompt_token_ids':token_ids}],sampling,use_tqdm=False)[0].outputs[0]
            decision = parsers.grade_judge(row,response.text)
            prepare.append(attempts_path,{'id':row['id'],'attempt':attempt,'raw':response.text,
                'input_prompt':prompt,'rendered_chat':rendered,'input_tokens':len(token_ids),
                'input_sha256':judge_input_digest(row,c),'output_tokens':len(response.token_ids),
                'finish_reason':response.finish_reason,'elapsed_seconds':time.perf_counter()-begin,
                'decision':decision['judge_parser_decision'],'saved_final_output':False})
            if decision['judge_parser_decision']['decision_parsed']:
                break  # Includes Z, which terminates as incorrect immediately.
            if attempt<spec['max_attempts']:
                time.sleep(rng.random()*spec['retry_wait_seconds']*2)
        print(f'Judge {row["id"]}: {decision["reason"]} attempt={attempt}',flush=True)
