"""Small format-adoption diagnostic using the unchanged baseline inference function.

Only row selection and the required total-count check are adapted. This is not
the 900-item comparison campaign, and parser-unresolved answers are not judged.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from era_lora.common import load_config, resolve_path, write_json, sha256_file
from era_lora.evaluate import validate_source, checkpoint_preflight, training_provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    ids = json.loads(Path(args.ids).read_text(encoding='utf-8-sig'))
    source, protocol, configs, hashes = validate_source(cfg)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Require distinct preselected IDs')
    frozen = json.loads((protocol / 'evaluation_items.json').read_text())
    if not set(ids) <= {r['id'] for r in frozen}:
        raise ValueError('Unknown diagnostic IDs')
    output = Path(args.output).resolve()
    plan = {'mode':'subset_format_diagnostic', 'ids':ids, 'seed':3407,
            'selection':'First frozen item from each of eight prespecified subjects; no outcome selection',
            'prompt_change':False, 'generation_config_change':False,
            'not_full_benchmark':True, 'judge_called':False, 'source_hashes':hashes}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    if output.exists():
        raise ValueError('Use a fresh diagnostic output directory')
    checkpoint = resolve_path(cfg, cfg['evaluation']['checkpoints']['era'])
    plan['checkpoint_hashes'] = checkpoint_preflight(checkpoint)
    plan['training_provenance'] = training_provenance(cfg,'era',checkpoint)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ.pop('VLLM_BATCH_INVARIANT', None)
    sys.path.insert(0, str(source / 'src'))
    spec = importlib.util.spec_from_file_location('frozen_diagnostic_evaluator',source/'src/evaluate.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    c = configs[3407]
    paths = evaluator.prepare.cache_paths(c,cfg['evaluation']['model_cache'],cfg['evaluation']['dataset_cache'])
    original_load = evaluator.prepare.load_data
    def subset_load(*a, **kw):
        datasets, rows = original_load(*a, **kw)
        by_id = {r['id']:r for r in rows}
        return datasets, [by_id[i] for i in ids]
    original_check = evaluator.check_records
    def subset_check(records, config, partial=False):
        original_check(records,config,partial=True)
        if any(r['id'] not in ids for r in records):
            raise ValueError('Unexpected item in subset')
    evaluator.prepare.load_data = subset_load
    evaluator.check_records = subset_check
    output.mkdir(parents=True)
    write_json(output/'manifest.json',plan)
    evaluator.infer(SimpleNamespace(checkpoint=str(checkpoint)),c,output,{'cache_paths':paths})
    records = evaluator.prepare.read_jsonl(output/'predictions.jsonl')
    if [r['id'] for r in records] != ids:
        raise ValueError('Incomplete diagnostic')
    old = json.loads((Path(cfg['evaluation']['baseline_runs']['3407'])/'results.json').read_text())
    baseline = {r['id']:r for r in old['records']}
    rows=[]
    for r in records:
        previous=baseline[r['id']]
        if r['input_prompt'] != previous['input_prompt'] or r['rendered_chat'] != previous['rendered_chat']:
            raise ValueError('Diagnostic input differs from baseline')
        def structure(text):
            return bool(re.search(r'(?ms)^Evidence:\s*\S.*?^Rule:\s*\S.*?^Application:\s*\S',text))
        grade=evaluator.parsers.grade_response(r)
        rows.append({'id':r['id'],'era_structure':structure(r['model_response']['raw']),
                     'baseline_era_structure':structure(previous['model_response']['raw']),
                     'finish_reason':r['model_response']['finish_reason'],
                     'output_tokens':r['model_response']['output_tokens'],
                     'parser_grade':grade,'response':r['model_response']['raw'],
                     'baseline_response':previous['model_response']['raw']})
    report={**plan,'items':len(rows),'era_structure_count':sum(r['era_structure'] for r in rows),
            'baseline_era_structure_count':sum(r['baseline_era_structure'] for r in rows),
            'length_count':sum(r['finish_reason']=='length' for r in rows),
            'parser_correct':sum(r['parser_grade'].get('hit')==1 for r in rows),
            'parser_unresolved':sum(r['parser_grade'].get('status')=='pending_judge' for r in rows),
            'rows':rows,'limitations':'Eight-item format diagnostic only; no general accuracy or ERA semantic correctness claim.'}
    write_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_hashes','training_provenance','checkpoint_hashes')},indent=2))


if __name__=='__main__':
    main()
