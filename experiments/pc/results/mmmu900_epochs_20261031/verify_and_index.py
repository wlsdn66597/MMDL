"""CPU-only independent artifact verification and full 900-question index."""
from pathlib import Path
import json, hashlib, re, statistics, collections, itertools

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def save(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def stable(x):
    if isinstance(x, dict): return {k: stable(v) for k,v in x.items() if k not in {'created_at','completed_at','started_at','timestamp','elapsed_seconds'}}
    if isinstance(x, list): return [stable(v) for v in x]
    return x

manifest = read(EVIDENCE/'download_manifest.json')
for name,digest in manifest.items(): assert sha(EVIDENCE/name)==digest, name
plan = read(EVIDENCE/'plan.json')
receipt = read(EVIDENCE/'pilot/verification.json')
assert sha(EVIDENCE/'plan.json') == receipt['original_plan_sha256']
assert sha(EVIDENCE/'scope_change_20261102.json') == receipt['scope_change_sha256']
assert read(EVIDENCE/'pilot_completed.json')['baseline_replay_and_all_three_repeats_passed']
assert not any(EVIDENCE.glob('epoch_*_seed42*'))
models=['baseline','epoch_2','epoch_3','epoch_4']
paths={'baseline':'historical_baseline', **{f'epoch_{e}':f'epoch_{e}_seed3407' for e in (2,3,4)}}
loaded={m:read(EVIDENCE/p/'results.json') for m,p in paths.items()}
ids=[r['id'] for r in loaded['baseline']['records']]
assert len(ids)==len(set(ids))==900
inputs=['id','subject','dataset_config','dataset_index','question_type','question','options','reference_answer','image_fields','original_image_sizes_wh','input_prompt','rendered_chat']
checks={}
for model,p in paths.items():
    result=loaded[model]; records=result['records']; summary=read(EVIDENCE/p/'summary.json')
    assert result['manifest']==read(EVIDENCE/p/'manifest.json')
    assert result['manifest']['configuration']==loaded['baseline']['manifest']['configuration']
    assert result['manifest']['code_sha256']==loaded['baseline']['manifest']['code_sha256']
    assert result['manifest']['mode']=='inference'
    assert [r['id'] for r in records]==ids
    assert result['summary']==summary
    assert sum(r['grading']['final']['hit'] for r in records)==summary['correct']
    for a,b in zip(loaded['baseline']['records'],records):
        assert all(a.get(k)==b.get(k) for k in inputs),a['id']
        for k in ('input_tokens','image_grid_thw','resized_image_sizes_wh'):
            assert a.get(k)==b.get(k)
        assert a['model_response']['input_tokens']==b['model_response']['input_tokens']
        g=b['grading']['final']; response=b['model_response']
        assert g['status']=='graded' and g['hit'] in (0,1)
        assert bool(g['hit'])==g['correct']
        assert not g.get('random_fallback',False)
        assert response['finish_reason'] in ('stop','length')
        if response['finish_reason']=='length': assert g['hit']==0 and g['token_limit_forced_incorrect']
    repeat_path='baseline_seed3407_replay' if model=='baseline' else p+'_repeat'
    repeated=read(EVIDENCE/repeat_path/'results.json')
    assert stable(records)==stable(repeated['records']),model
    assert summary==repeated['summary']
    for suffix in ('predictions.jsonl','judge_attempts.jsonl'):
        first=[json.loads(s) for s in (EVIDENCE/p/suffix).read_text(encoding='utf8').splitlines() if s.strip()]
        second=[json.loads(s) for s in (EVIDENCE/repeat_path/suffix).read_text(encoding='utf8').splitlines() if s.strip()]
        assert stable(first)==stable(second),(model,suffix)
    for filename,h in receipt['artifacts'][model].items(): assert sha(EVIDENCE/p/filename)==h
    if model=='baseline':
        for filename,h in plan['baseline_runs']['3407']['hashes'].items(): assert sha(EVIDENCE/p/filename)==h
    else:
        launch=read(EVIDENCE/p/'launcher.json'); expected=plan['exports'][model[-1]]['hashes']
        assert launch['checkpoint_sha256']==expected
        for f,h in result['manifest']['model_identity']['weights_sha256'].items(): assert expected[f]==h
        assert launch['source_hashes']==plan['source_hashes']
    checks[model]={'unique_questions':900,'correct':summary['correct'],'final_grades_complete':True,'repeat_records_predictions_judge_identical_excluding_timing':True,'same_inputs_and_configuration':True,'preserved_hashes_verified':True}

inventory=read(EVIDENCE/'pilot/all900_response_inventory.json')
assert [r['id'] for r in inventory]==ids
for i,row in enumerate(inventory):
    for model in models: assert row['models'][model]['record']==loaded[model]['records'][i]
stats=read(EVIDENCE/'pilot/summary.json')
# Recompute the frozen format convention without importing any model/evaluator.
def fmt(r):
    raw=r['model_response']['raw']
    legacy=bool(re.search(r'(?ms)^Evidence:\s*\S.*?^Rule:\s*\S.*?^Application:\s*\S',raw))
    hs=list(re.finditer(r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?(Evidence|Rule|Application)(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?',raw))
    ordered=[h[1].lower() for h in hs]==['evidence','rule','application']
    f=re.search(r'(?im)^[ \t]*(?:\*\*)?Final answer[ \t]*:[ \t]*(?:\*\*)?([^\n]+)[ \t]*\s*\Z',raw)
    terminal=bool(f and re.search(r'\w',f[1]) and (not hs or f.start()>hs[-1].end()))
    nonempty=ordered and all(re.search(r'\w',raw[h.end():end]) for h,end in zip(hs,[hs[1].start(),hs[2].start(),f.start() if f else len(raw)]))
    return {'legacy_strict_headings':legacy,'ordered_unique_headers':ordered,'nonempty_sections':bool(nonempty),'terminal_final_answer':terminal,'complete_era_format':bool(ordered and nonempty and terminal and r['model_response']['finish_reason']=='stop')}
index=[]; additional={}
for row in inventory:
    short={'id':row['id'],'subject':row['subject'],'models':{}}
    for model,x in row['models'].items():
        assert fmt(x['record'])==x['format']
        short['models'][model]={'correct':x['record']['grading']['final']['hit'],'format':x['format']['complete_era_format'],'legacy':x['format']['legacy_strict_headings'],'tokens':x['record']['model_response']['output_tokens'],'length':x['record']['model_response']['finish_reason']=='length','repeat':x['repetition']['has_line_repeated_3_times'],'answer':x['record']['grading']['final']['extracted_answer']}
    index.append(short)
for m in models:
    rs=[r['models'][m] for r in index]
    assert sum(r['correct'] for r in rs)==stats['metrics'][m]['correct']
    assert sum(r['format'] for r in rs)==stats['metrics'][m]['complete_era_format_count']
    assert sum(r['legacy'] for r in rs)==stats['metrics'][m]['legacy_strict_headings_count']
    additional[m]={'by_question_type':{},'correct_but_incomplete_format':sum(x['correct'] and not x['format'] for x in rs),'complete_format_but_wrong':sum(x['format'] and not x['correct'] for x in rs)}
    for t in ('multiple-choice','open'):
        group=[r for r in loaded[m]['records'] if r['question_type']==t]
        additional[m]['by_question_type'][t]={'total':len(group),'correct':sum(r['grading']['final']['hit'] for r in group)}
pairs={}
for left,right in itertools.combinations(models,2):
    rows=collections.Counter((x['models'][left]['correct'],x['models'][right]['correct']) for x in index)
    regressions=[x for x in index if x['models'][left]['correct'] and not x['models'][right]['correct']]
    pairs[left+'_to_'+right]={'wrong_to_right':rows[0,1],'right_to_wrong':rows[1,0],'both_correct':rows[1,1],'both_wrong':rows[0,0],'regressions_complete_format':sum(x['models'][right]['format'] for x in regressions),'regressions_length':sum(x['models'][right]['length'] for x in regressions)}
    pairs[left+'_to_'+right]['by_subject']={}
    for subject in sorted({x['subject'] for x in index}):
        counts=collections.Counter((x['models'][left]['correct'],x['models'][right]['correct']) for x in index if x['subject']==subject)
        pairs[left+'_to_'+right]['by_subject'][subject]={'wrong_to_right':counts[0,1],'right_to_wrong':counts[1,0],'both_correct':counts[1,1],'both_wrong':counts[0,0]}
save(ROOT/'local_verification.json',{'passed':True,'archive_sha256':sha(ROOT/'pilot_evidence_verified.zip'),'manifest_files_verified':len(manifest),'runs':checks,'model_weights_downloaded':False,'cancelled_seeds':[42,123,2026,9876],'note':'Checkpoint hash provenance verified against frozen launch/plan; weights were not downloaded. Remote process absence/cancelled directories were checked separately.'})
save(ROOT/'analysis_metrics.json',{'metrics':stats['metrics'],'paired':pairs,'additional':additional})
save(ROOT/'all900_index.json',index)
print(json.dumps({'passed':True,'files':len(manifest),'correct':{m:checks[m]['correct'] for m in models},'paired':pairs},indent=2))
