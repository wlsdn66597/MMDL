"""Verify the seven MMMU runs against downloaded evidence; produce per-item paired comparisons."""
from pathlib import Path
import ast, collections, hashlib, itertools, json, math, re
R=Path(__file__).resolve().parent
P=R.parent/'mmmu900_epochs_20261031'
D=R/'evidence'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
def stable(x):
    if isinstance(x,dict): return {k:stable(v) for k,v in x.items() if k not in ('created_at','completed_at','started_at','timestamp','elapsed_seconds')}
    if isinstance(x,list): return [stable(v) for v in x]
    return x
# Same strict format definition previously used in the baseline/ERA independent audit.
tree=ast.parse((P/'verify_and_index.py').read_text(encoding='utf8'))
fm=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='fmt')
exec(compile(ast.Module(body=[fm],type_ignores=[]),'format_metric','exec'))
plan=read(D/'plan.json'); completion=read(D/'completed.json');old=P/'evidence'
assert sha(D/'plan.json')==sha(R/'plan.json')
assert plan['evaluation_seeds']==[3407] and completion['seeds']==[3407]
assert read(D/'training_pair_verification.json')['passed']
assert completion['all_repeats_passed'] and completion['primary_runs']==completion['repeat_runs']==3
attempt=next((D/'resume_attempts').iterdir());assert read(attempt/'exit.json')['code']==0 and read(attempt/'completed.json')==completion
assert sha(old/'pilot/verification.json')==plan['reference_replay_receipt_sha256']
for folder in (D,old):
    for k,h in read(folder/'download_manifest.json').items(): assert sha(folder/k)==h,k
inv=read(D/'all900_response_inventory.json');sample=read(D/'prespecified30_responses.json')
ids=[x['id'] for x in inv];assert len(set(ids))==len(ids)==900
assert len(sample)==30 and set(x['id'] for x in sample)==set(plan['semantic_sample']['ids'])
assert len({x['subject'] for x in sample})==30
assert sample==[x for x in inv if x['id'] in set(plan['semantic_sample']['ids'])]
models=list(inv[0]['models']);stats=read(D/'summary.json')['metrics']
paths={'baseline':old/'historical_baseline',**{f'era_epoch_{e}':old/f'epoch_{e}_seed3407' for e in (2,3,4)},**{f'plain_epoch_{e}':D/f'plain_epoch_{e}_seed3407' for e in (2,3,4)}}
fields=('id','subject','dataset_config','dataset_index','question_type','question','options','reference_answer','image_fields','original_image_sizes_wh','input_prompt','rendered_chat')
base=read(paths['baseline']/'results.json')
idx=[{'id':x['id'],'subject':x['subject'],'models':{}} for x in inv]
for name in models:
    path=paths[name];run=read(path/'results.json');repdir=old/'baseline_seed3407_replay' if name=='baseline' else Path(str(path)+'_repeat')
    repeat=read(repdir/'results.json');summary=read(path/'summary.json')
    assert run['summary']==summary==repeat['summary'] and summary['graded']==summary['total']==900
    assert [v['id'] for v in run['records']]==ids
    assert run['manifest']==read(path/'manifest.json')
    assert run['manifest']['configuration']==base['manifest']['configuration'] and run['manifest']['code_sha256']==base['manifest']['code_sha256']
    assert run['manifest']['mode']=='inference' and run['manifest']['configuration']['sampling']['seed']==3407
    assert all(run['manifest'][k]==repeat['manifest'][k] for k in ('configuration','environment','model_identity','code_sha256'))
    assert stable(run['records'])==stable(repeat['records'])
    pred=lines(path/'predictions.jsonl');assert [v['id'] for v in pred]==ids
    for fn in ('predictions.jsonl','judge_attempts.jsonl'):
        assert stable(lines(path/fn))==stable(lines(repdir/fn)),(name,fn)
    if name.startswith('plain'):
        e=name[-1];proof=read(D/f'export_epoch_{e}.json');receipt=read(D/f'epoch_{e}_repeatability.json')
        assert receipt['passed'] and not receipt['differences'] and receipt['records_compared']==900
        assert proof['optimizer_step']==13*int(e)
        assert read(path/'launcher.json')['checkpoint_sha256']==proof['hashes']
        assert all(proof['hashes'][k]==h for k,h in run['manifest']['model_identity']['weights_sha256'].items())
        for dd,key in [(path,'first_artifacts_sha256'),(repdir,'second_artifacts_sha256')]:
            for k,h in receipt[key].items():assert sha(dd/k)==h
        expected=completion['artifacts'][name]
    else: expected=plan['reference_hashes'][name]
    for k,h in expected.items():assert sha(path/k)==h,(name,k)
    for i,v in enumerate(run['records']):
        x=inv[i]['models'][name]; ref=base['records'][i]; resp=v['model_response']; grade=v['grading']['final']
        assert x['record']==v and resp==pred[i]['model_response']
        assert all(v.get(k)==ref.get(k)==pred[i].get(k) for k in fields)
        assert resp['input_tokens']==ref['model_response']['input_tokens']
        assert grade['status']=='graded' and grade['hit'] in (0,1) and bool(grade['hit'])==grade['correct']
        if resp['finish_reason']=='length':assert grade['hit']==0 and grade['token_limit_forced_incorrect']
        assert fmt(v)==x['format']
        idx[i]['models'][name]=dict(correct=grade['hit'],format=x['format']['complete_era_format'],legacy=x['format']['legacy_strict_headings'],tokens=resp['output_tokens'],length=resp['finish_reason']=='length',repeat=x['repetition']['has_line_repeated_3_times'],answer=grade['extracted_answer'])
    ms=[x['models'][name] for x in idx];assert sum(x['correct'] for x in ms)==summary['correct']==stats[name]['correct']
    assert sum(x['format'] for x in ms)==stats[name]['complete_era_format_count']
    assert sum(x['legacy'] for x in ms)==stats[name]['legacy_strict_headings_count']
    assert sum(x['length'] for x in ms)==stats[name]['length_count']
def pairs(rows,l,r):
    counts=collections.Counter((x['models'][l]['correct'],x['models'][r]['correct']) for x in rows)
    return dict(both_correct=counts[1,1],left_only=counts[1,0],right_only=counts[0,1],both_wrong=counts[0,0])
edges=[('baseline',m) for m in models if m!='baseline']+[(f'plain_epoch_{e}',f'era_epoch_{e}') for e in (2,3,4)]
pairs_out={};discord={}
for l,r in edges:
    k=l+'__'+r;v=pairs(idx,l,r);n=v['left_only']+v['right_only']
    v['mcnemar_exact_two_sided']=min(1,2*sum(math.comb(n,i) for i in range(min(v['right_only'],v['left_only'])+1))/(2**n)) if n else 1
    v['by_subject']={s:pairs([x for x in idx if x['subject']==s],l,r) for s in sorted({x['subject'] for x in idx})};pairs_out[k]=v
    discord[k]=[{'id':x['id'],'subject':x['subject'],'left_correct':x['models'][l]['correct'],'right_correct':x['models'][r]['correct']} for x in idx if x['models'][l]['correct']!=x['models'][r]['correct']]
save(R/'all900_index.json',idx);save(R/'paired_results.json',pairs_out);save(R/'discordant_items.json',discord)
save(R/'verification.json',{'verified':True,'seed':3407,'unique_items':900,'models':models,'format_and_scores_recomputed':True,'all_seven_artifact_sets_and_repeats_verified':True,'equal_prompts_inputs_configuration_and_input_token_counts':True,'length_policy_and_final_grades_verified':True,'verified_downloaded_files':len(read(D/'download_manifest.json')),'official_correct':{m:stats[m]['correct'] for m in models},'prespecified_ids':[x['id'] for x in sample]})
print('VERIFIED', {m:stats[m]['correct'] for m in models})

