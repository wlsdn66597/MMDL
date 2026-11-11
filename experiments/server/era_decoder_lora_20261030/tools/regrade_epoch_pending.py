"""Resolve two saved-response parser ambiguities using the frozen local judge."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from era_lora.common import load_config,write_json,sha256_file
from era_lora.evaluate import validate_source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    cfg=load_config(ROOT/'configs/era_4epoch_20261030.json')
    source,protocol,configs,hashes=validate_source(cfg);c=configs[3407]
    sys.path.insert(0,str(source/'src'))
    import prepare,parsers,judge
    run=ROOT/'runs/era_4epoch_20261030'
    mm_path=run/'diagnostics/epoch_4/mmmu/predictions.jsonl'
    train_path=run/'diagnostics/epoch_3/train/generations.jsonl'
    mm=[r for r in prepare.read_jsonl(mm_path) if parsers.grade_response(r)['status']=='pending_judge']
    pending=list(mm);origins={r['id']:{'epoch':4,'domain':'mmmu'} for r in mm}
    for r in prepare.read_jsonl(train_path):
        if r['condition']!='eval_resolution':continue
        opts=dict(re.findall(r'^([A-Z])\. (.*)$',r['input_prompt'],re.M))
        converted={'id':r['id'],'question_type':'multiple-choice','question':r['input_prompt'].split('\nOptions:\n')[0].removeprefix('Question: '),
            'options':opts,'reference_answer':r['answer_letter'],
            'model_response':{'raw':r['response'],'finish_reason':r['finish_reason']}}
        if parsers.grade_response(converted)['status']=='pending_judge':
            pending.append(converted);origins[r['id']]={'epoch':3,'domain':'train','condition':'eval_resolution'}
    assert len(pending)==2
    output=run/'reasoning_audit_judge'
    if not a.execute:print([r['id'] for r in pending]);return
    if output.exists():raise RuntimeError('Refuse to overwrite judge run')
    output.mkdir()
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0';os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ.pop('VLLM_BATCH_INVARIANT',None)
    manifest={'cache_paths':prepare.cache_paths(c,cfg['evaluation']['model_cache'],cfg['evaluation']['dataset_cache']),
        'baseline_source_hashes':hashes,'input_files_sha256':{str(mm_path):sha256_file(mm_path),str(train_path):sha256_file(train_path)},
        'student_inference':False,'training':False,'judge_config':c['judge'],'origins':origins}
    write_json(output/'manifest.json',manifest)
    for r in pending:prepare.append(output/'predictions.jsonl',r)
    write_json(output/'pending_judge.json',[r['id'] for r in pending])
    judge.run(c,output,manifest)
    last={r['id']:r for r in prepare.read_jsonl(output/'judge_attempts.jsonl')}
    grades=[{'id':r['id'],**origins[r['id']],'grade':parsers.grade_judge(r,last[r['id']]['raw']),
        'judge_raw':last[r['id']]['raw']} for r in pending]
    for path,h in manifest['input_files_sha256'].items():assert sha256_file(path)==h
    write_json(output/'completed.json',{'status':'complete','records':grades,'original_predictions_unchanged':True})
    print(json.dumps(grades,indent=2),flush=True)


if __name__=='__main__':main()
