"""Replay frozen MMMU grades and prepare/run an isolated final-answer scoring repair.

Never changes student responses, reference labels, frozen evaluator, or old results.
MC scoring stays frozen. Open answers cannot select synthetic A/B by substring or
unit letter. Exact final-answer matches are accepted; remaining open responses go
to the SAME pinned Qwen3-8B and matching prompt, using a unique terminal answer
when available. No semantic-review scores or new numeric tolerance are used.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import random
import sys
import time
import unicodedata

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = CODE_ROOT / 'baseline' / 'src'
SPEC = importlib.util.spec_from_file_location('frozen_mmmu_parsers', SOURCE / 'parsers.py')
frozen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen)
VERSION = 'open-final-answer-repair-v1'
MODEL_PATHS = {'baseline':'historical_baseline', **{f'epoch_{e}':f'epoch_{e}_seed3407' for e in (2,3,4)}}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()] if Path(path).exists() else []

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def digest(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode('utf-8')).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',',':'))

def judge_prompt(row, config, answer):
    choices, _ = frozen.choice_info(row)
    options = 'There are several options: \n' + ''.join(f'{k}. {v}\n' for k,v in choices.items())
    return config['judge']['prompt_template'].format(row['question'], options, answer)

def aliases(value):
    if isinstance(value,str) and value.startswith('['):
        try:
            parsed=ast.literal_eval(value)
            if isinstance(parsed,list) and all(isinstance(x,str) for x in parsed):return parsed
        except (ValueError,SyntaxError):pass
    return [str(value)]

def normalized(value):
    text=unicodedata.normalize('NFKC',str(value)).strip()
    for _ in range(4):
        before=text
        for marker in ('**','__','`','$'):
            if text.startswith(marker) and text.endswith(marker) and len(text)>2*len(marker):
                text=text[len(marker):-len(marker)].strip()
        text=re.sub(r'^\\(?:boxed|mathrm|text)\{([^{}]*)\}$',r'\1',text).strip()
        if text==before:break
    return re.sub(r'\s+',' ',text).casefold()

def terminal_open_answer(raw):
    """Reuse strict extraction; allow an explicit final label within the last line.

    Multiple conflicting final fields remain ambiguous; never select one merely
    because it agrees with the reference. This helper never sees the reference.
    """
    field=frozen.extract_final(raw,'open',{})
    if field['status']=='extracted':return field['payload'],'terminal_final_field'
    if field['status']=='absent':
        lines=[frozen.normalize_line(x) for x in str(raw).split('</think>')[-1].splitlines() if x.strip()]
        if lines:
            m=re.search(r'\bFinal\s+answer\s*:\s*(.+)$',lines[-1],re.I)
            if m:return frozen.normalize_line(m.group(1)),'terminal_inline_final_field'
    return None,field['reason']

def proposed_grade(row, original_replayed):
    if row['question_type']!='open':
        return {**copy.deepcopy(original_replayed),'repair_route':'unchanged_multiple_choice'},None
    if row['model_response']['finish_reason']=='length':
        return {**copy.deepcopy(original_replayed),'repair_route':'unchanged_length_policy'},None
    answer,route=terminal_open_answer(row['model_response']['raw'])
    if answer is not None and normalized(answer) in {normalized(x) for x in aliases(row['reference_answer'])}:
        return {'status':'graded','hit':1,'correct':True,'extracted_answer':answer,
                'grading_gold':row['reference_answer'],'method':'open_exact_final_answer',
                'repair_route':route},None
    payload=answer if answer is not None else str(row['model_response']['raw']).split('</think>')[-1].strip()
    return {'status':'pending_judge','hit':None,'correct':None,'method':'pending_same_qwen3_8b',
            'repair_route':route,'answer_payload':answer},payload

def grade_new_judge(row, raw):
    choices,_=frozen.choice_info(row)
    decision=frozen.parse_judge_output(raw,choices,frozen.can_infer)
    # For open questions A represents the reference, including a reference like C.
    # The original choice_info quirk incorrectly used literal reference letters.
    gold='A' if row['question_type']=='open' else row['reference_answer']
    hit=int(decision['valid_option_extracted'] and decision['answer']==gold)
    return {'status':'graded','hit':hit,'correct':bool(hit),'grading_gold':gold,
            'extracted_answer':decision['answer'],'method':'same_qwen3_8b_repaired_input',
            'judge_parser_decision':decision}

def load_inputs(evidence):
    runs={}
    for name,folder in MODEL_PATHS.items():
        root=evidence/folder
        manifest=read(root/'manifest.json')
        records=read(root/'results.json')['records']
        if len(records)!=900 or len({r['id'] for r in records})!=900:raise ValueError('900 unique IDs required')
        if digest((SOURCE/'parsers.py').read_bytes())!=manifest['code_sha256']['src/parsers.py']:
            raise ValueError('Frozen parser bytes changed')
        attempts=jsonl(root/'judge_attempts.jsonl')
        runs[name]={'manifest':manifest,'records':records,'attempts':attempts,
                    'source_results_sha256':digest((root/'results.json').read_bytes())}
    base=runs['baseline']
    for name,run in runs.items():
        for k in ('prompts','scoring','judge'):
            if run['manifest']['configuration'][k]!=base['manifest']['configuration'][k]:raise ValueError('Protocol mismatch '+name+'/'+k)
        for b,r in zip(base['records'],run['records']):
            for k in ('id','question','options','reference_answer','question_type','input_prompt'):
                if b[k]!=r[k]:raise ValueError('Input mismatch '+name+'/'+k)
    return runs

def prepare(runs,output):
    config=runs['baseline']['manifest']['configuration']
    judge_identity=digest(canonical(config['judge']))
    replayed={};cache={};checks=[]
    for name,run in runs.items():
        lookup={r['id']:r for r in run['records']};last={}
        for a in run['attempts']:
            row=lookup[a['id']]
            prompt=judge_prompt(row,config,row['model_response']['raw'].split('</think>')[-1].strip())
            if digest(prompt)!=a['input_sha256']:raise ValueError('Saved judge input mismatch '+name+'/'+a['id'])
            last[a['id']]=a
            # A cache is valid only for exactly the same prompt and judge config.
            key=a['input_sha256'];cache.setdefault(key,[]).append({'raw':a['raw'],'origin':name+'/'+a['id'],'source':'saved_frozen_judge'})
        grades={}
        for row in run['records']:
            g=frozen.grade_response(row)
            if g['status']=='pending_judge':g=frozen.grade_judge(row,last[row['id']]['raw'])
            old=row['grading']['final']
            if any(g.get(k)!=old.get(k) for k in ('hit','method','extracted_answer')):
                raise ValueError('Frozen replay failed '+name+'/'+row['id'])
            grades[row['id']]=g
        replayed[name]=grades
        checks.append({'model':name,'n':900,'original_correct':sum(g['hit'] for g in grades.values()),
                       'source_results_sha256':run['source_results_sha256'],
                       'mc_correct':sum(grades[r['id']]['hit'] for r in run['records'] if r['question_type']=='multiple-choice'),
                       'open_correct':sum(grades[r['id']]['hit'] for r in run['records'] if r['question_type']=='open'),
                       'saved_judge_attempts_checked':len(run['attempts'])})
    attempts=jsonl(output/'new_judge_attempts.jsonl')
    for a in attempts:
        if a['judge_identity']!=judge_identity or digest(a['input_prompt'])!=a['input_sha256']:
            raise ValueError('New judge cache identity mismatch')
        cache.setdefault(a['input_sha256'],[]).append({**a,'source':'new_same_qwen3_8b'})
    records=[];jobs={}
    for name,run in runs.items():
        for row in run['records']:
            grade,payload=proposed_grade(row,replayed[name][row['id']])
            entry={'model':name,'id':row['id'],'subject':row['subject'],'question_type':row['question_type'],
                   'reference_answer':row['reference_answer'],'response_sha256':digest(row['model_response']['raw']),
                   'original_hit':row['grading']['final']['hit'],'original_method':row['grading']['final']['method'],
                   'proposed':grade}
            if payload is not None:
                prompt=judge_prompt(row,config,payload);key=digest(prompt)
                entry['judge_input_sha256']=key
                cached=cache.get(key,[])
                if cached:
                    results=[grade_new_judge(row,c['raw']) for c in cached]
                    parsed=[(c,g) for c,g in zip(cached,results) if g['judge_parser_decision']['decision_parsed']]
                    if len({g['extracted_answer'] for _,g in parsed})>1:raise ValueError('Conflicting cached judge verdicts '+key)
                    chosen_cache,chosen_grade=parsed[-1] if parsed else (cached[-1],results[-1])
                    decision=chosen_grade['judge_parser_decision']
                    new_attempt_count=sum(c['source']=='new_same_qwen3_8b' for c in cached)
                    if decision['decision_parsed'] or new_attempt_count>=config['judge']['max_attempts'] or cached[-1]['source']=='saved_frozen_judge':
                        entry['proposed']={**grade,**chosen_grade,'judge_reuse':chosen_cache['source']}
                if entry['proposed']['status']=='pending_judge':
                    jobs.setdefault(key,{'input_sha256':key,'input_prompt':prompt,'judge_identity':judge_identity,
                                         'origins':[],'row_for_grading':row,'prior_attempts':sum(c['source']=='new_same_qwen3_8b' for c in cached)})['origins'].append(name+'/'+row['id'])
            records.append(entry)
    summary={}
    for name in MODEL_PATHS:
        group=[r for r in records if r['model']==name];pending=sum(r['proposed']['status']!='graded' for r in group)
        known=sum(r['proposed']['hit'] or 0 for r in group)
        summary[name]={'total':900,'original_correct':sum(r['original_hit'] for r in group),
                       'original_accuracy_percent':sum(r['original_hit'] for r in group)/9,
                       'resolved':900-pending,'pending':pending,'resolved_correct':known,
                       'corrected_accuracy_percent':known/9 if not pending else None,
                       'proposed_method_counts':dict(Counter(r['proposed']['method'] for r in group))}
    mc_counts={c['model']:c['mc_correct'] for c in checks}
    open_n=sum(r['question_type']=='open' for r in runs['baseline']['records'])
    state={'version':VERSION,'status':'complete' if not jobs else 'awaiting_same_qwen3_8b',
           'total_responses':3600,'unique_judge_jobs_pending':len(jobs),'models':summary,
           'changes':{'multiple_choice':'unchanged frozen final-field parser + saved Qwen3-8B decisions',
                      'open':'exact final answer or same Qwen3-8B on explicit final answer; never synthetic-letter/substring selection',
                      'numeric_tolerance':'no new tolerance introduced; fractional/rounded/text equivalence remains Qwen3-8B responsibility',
                      'gold_labels':'unchanged','length_policy':'all_incorrect unchanged','semantic_review_scores':'not read or used'},
           'frozen_replay':{'passed':True,'rows':3600,'checks':checks},
           'open_only_regrade_bound':{'mc_questions':900-open_n,'open_questions':open_n,
                'baseline_min_correct':mc_counts['baseline'],
                'epoch_max_correct_if_all_open_correct':{m:n+open_n for m,n in mc_counts.items() if m!='baseline'},
                'baseline_rank_cannot_reverse_if_mc_unchanged':all(mc_counts['baseline']>n+open_n for m,n in mc_counts.items() if m!='baseline')},
           'judge_identity':judge_identity,'judge_config':config['judge']}
    write(output/'summary.json',state);write(output/'regrade_records.json',records)
    public_jobs=[{k:v for k,v in j.items() if k!='row_for_grading'} for j in jobs.values()]
    write(output/'pending_judge_jobs.json',public_jobs)
    write(output/'resolved_changes.json',[r for r in records if r['proposed']['status']=='graded' and r['original_hit']!=r['proposed']['hit']])
    write(output/'provenance.json',{'scoring_version':VERSION,'code_sha256':digest(Path(__file__).read_bytes()),
          'frozen_parser_sha256':digest((SOURCE/'parsers.py').read_bytes()),'same_prompt_and_scoring_across_four_models':True,
          'originals_preserved':True,'judge_identity':judge_identity,'new_judge_calls_recorded':len(attempts)})
    report=['MMMU 900 재채점 준비 및 기존 규칙 재현','',
            '기존 기준의 정답률은 본 비교 결과로 보존합니다. 수정 규칙은 네 모델 전체에 동일하게 적용한 보조 비교입니다.',
            '실제 생성 지시는 result:가 아니라 Final answer: LETTER(객관식) / Final answer: ANSWER(주관식)였습니다.',
            '원래 3600응답의 파서와 저장된 Qwen3-8B 판정을 재실행하여 기존 hit/method/extracted_answer를 모두 재현했습니다.','',
            '모델 | 기존 전체 정답/900 | 객관식 정답/847 | 주관식 정답/53 | 새 Judge 대기 응답',
            *[f"{c['model']} | {c['original_correct']} ({c['original_correct']/9:.2f}%) | {c['mc_correct']} | {c['open_correct']} | {summary[c['model']]['pending']}" for c in checks],'',
            '수정 범위:',
            '- 객관식: 기존 최종 선지 파서와 Qwen3-8B 판정을 유지. 설명이 틀려도 최종 정답을 고르면 정답입니다.',
            '- 주관식: 수치 답의 단위 A 또는 정답 문자열 일부가 합성 선택지 A로 인식되는 경로를 제거합니다.',
            '- 유일한 최종 답이 있으면 그 답을 같은 Qwen3-8B에 전달합니다. 모호한 응답은 전체 원문을 유지합니다.',
            '- 주관식 정답 자체가 A/B/C 등의 문자일 때 합성 선택지와 혼동하지 않습니다.',
            '- 정답표, 길이 상한 오답 처리, 모델 생성 응답, Qwen3-8B 모델·revision·프롬프트·샘플링을 보존합니다.',
            '- 새 숫자 오차 허용값을 도입하지 않습니다. 분수·반올림·텍스트 동등성은 기존 Qwen3-8B의 판정 대상입니다.',
            '- 이전 E/R/A 내용 판정은 이 재채점에 사용하지 않습니다.','',
            f"현재 상태: {state['status']}. 대기 {sum(s['pending'] for s in summary.values())}응답, 동일 입력 중복을 제거하면 Qwen3-8B {len(jobs)}건입니다.",
            '새 Judge 실행 전에는 수정 정답률을 제시하지 않습니다. 새로운 Judge 판정에도 오류가 남을 수 있으므로 실행 후 알려진 오류 사례를 다시 확인해야 합니다.','',
            f"주관식 채점만 바꾸는 한 baseline 우위는 역전될 수 없습니다. baseline 객관식만 {mc_counts['baseline']}개 정답이고, 2epoch는 주관식 {open_n}개를 모두 맞혀도 {mc_counts['epoch_2']+open_n}개입니다.",
            '이는 수정된 정확도 수치가 아니라 객관식 점수를 유지한다는 조건에서의 수학적 상한 비교입니다. 2/3/4epoch 사이의 순서는 아직 확정하지 않습니다.','',
            '실행: 기존 Linux 환경에서 tools/regrade_final_answers.py --execute. 별도 번들의 코드/결과 폴더 구조를 보존하면 기본 경로로 실행할 수 있습니다.']
    write_text='\n'.join(report)+'\n';(output/'comparison_note.txt').write_text(write_text,encoding='utf-8')
    return state,jobs,config,records

def execute_jobs(jobs,config,output,model_cache):
    if not jobs:return
    sys.path.insert(0,str(SOURCE))
    import prepare as baseline_prepare
    baseline_prepare.runtime_check(config)
    import os
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ.pop('VLLM_BATCH_INVARIANT',None)
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    spec=config['judge'];runtime=spec['runtime']
    snapshot=baseline_prepare.snapshot(config,model_cache,judge=True)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    llm=LLM(model=str(snapshot),dtype='bfloat16',quantization=None,seed=spec['sampling']['seed'],
            **{k:runtime[k] for k in ('tensor_parallel_size','max_model_len','gpu_memory_utilization','max_num_seqs','max_num_batched_tokens','enforce_eager')})
    sampling=SamplingParams(**spec['sampling'],max_tokens=spec['max_new_tokens'])
    for j in jobs.values():
        rng=random.Random(spec['sampling']['seed']+int(hashlib.sha256(j['row_for_grading']['id'].encode()).hexdigest()[:8],16))
        token_ids=tokenizer.apply_chat_template([{'role':'user','content':j['input_prompt']}],tokenize=True,
                add_generation_prompt=True,enable_thinking=spec['enable_thinking'])
        if len(token_ids)+spec['max_new_tokens']>runtime['max_model_len']:raise ValueError('Judge context limit; no truncation')
        for attempt in range(j['prior_attempts']+1,spec['max_attempts']+1):
            start=time.perf_counter()
            response=llm.generate([{'prompt_token_ids':token_ids}],sampling,use_tqdm=False)[0].outputs[0]
            grade=grade_new_judge(j['row_for_grading'],response.text)
            event={k:v for k,v in j.items() if k not in ('row_for_grading','prior_attempts')}
            event.update(attempt=attempt,raw=response.text,input_tokens=len(token_ids),output_tokens=len(response.token_ids),
                         finish_reason=response.finish_reason,elapsed_seconds=time.perf_counter()-start,
                         decision=grade['judge_parser_decision'])
            baseline_prepare.append(output/'new_judge_attempts.jsonl',event)
            print('Judge '+j['input_sha256'][:12]+' '+str(grade['extracted_answer']),flush=True)
            if grade['judge_parser_decision']['decision_parsed']:break
            if attempt<spec['max_attempts']:time.sleep(rng.random()*spec['retry_wait_seconds']*2)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=REPO_ROOT/'results/mmmu900_epochs_20261031/evidence')
    parser.add_argument('--output',type=Path,default=REPO_ROOT/'results/mmmu900_epochs_20261031/regrade_final_answers_20261103')
    parser.add_argument('--execute',action='store_true',help='Run pending jobs using the original Linux/CUDA environment and pinned Qwen3-8B')
    parser.add_argument('--model-cache',default='/home/rlatjswo/mmdl-cache/huggingface/hub')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.output.resolve()==args.evidence.resolve() or args.evidence.resolve() in args.output.resolve().parents:
        raise ValueError('Output must not be inside frozen evidence')
    runs=load_inputs(args.evidence);state,jobs,config,records=prepare(runs,args.output)
    if args.execute:
        execute_jobs(jobs,config,args.output,args.model_cache)
        state,jobs,config,records=prepare(runs,args.output)
    for name,run in runs.items():
        if digest((args.evidence/MODEL_PATHS[name]/'results.json').read_bytes())!=run['source_results_sha256']:
            raise ValueError('Original results changed during run')
    print(json.dumps({k:state[k] for k in ('status','total_responses','unique_judge_jobs_pending','models')},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
