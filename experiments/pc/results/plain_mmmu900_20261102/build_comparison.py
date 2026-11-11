from pathlib import Path
import json, re, collections
R=Path(__file__).resolve().parent;P=R.parent/'mmmu900_epochs_20261031'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def js(x):return json.dumps(x,ensure_ascii=False).replace('</','<\\/').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
inv=read(R/'evidence/all900_response_inventory.json');sample=read(R/'evidence/prespecified30_responses.json');idx=read(R/'all900_index.json');pairs=read(R/'paired_results.json');stats=read(R/'evidence/summary.json')['metrics'];old=read(P/'semantic_review.json');notes=read(R/'review_notes.json');anns={}
for a in old['annotations']:
    a=dict(a);m=a['model'];m=m if m=='baseline' else 'era_'+m;a['model']=m
    if a['id']=='validation_Energy_and_Power_3' and m=='era_epoch_3':
        a['original_scores']=dict(a['scores']);a['scores']=dict(a['scores'],A=1)
        a['rationale']='앞부분은 정압·정체압이 혼재한다. 말미에서 110−101.325=8.675 kPa와 원형 출구 면적으로 약 5.25를 구했다. 상류 속도항 생략이 남아 이전 A=0을 부분 정확으로 수정했다.'
    anns[a['id'],m]=a
for i,row in enumerate(sample):
    for n,e in enumerate((2,3,4)):
        m=f'plain_epoch_{e}';v=row['models'][m]['record'];raw=v['model_response']['raw']
        code,note=notes[i][n].split(' ',1);scores={k:None if val=='?' else int(val) for k,val in zip('ERA',code)}
        intervals=[(0,len(raw))] if len(raw)<=3600 else [(0,1300),(len(raw)//2,len(raw)//2+400),(len(raw)-850,len(raw))]
        anns[row['id'],m]=dict(id=row['id'],subject=row['subject'],model=m,official_correct=v['grading']['final']['hit'],scores=scores,rationale=note,coverage='full_response' if len(intervals)==1 else 'excerpts',raw_response_chars=len(raw),image_reviewed=True,quotes=[dict(offset=start,exact_quote=raw[start:end]) for start,end in intervals])
cases=[]
for i,e,stage,note,pquote,equote in read(R/'reviewed_pairs_notes.json'):
    row=sample[i];ms=row['models'];pk=f'plain_epoch_{e}';ek=f'era_epoch_{e}'
    raw=lambda m:ms[m]['record']['model_response']['raw'];hit=lambda m:ms[m]['record']['grading']['final']['hit']
    assert hit(pk)!=hit(ek) and pquote in raw(pk) and equote in raw(ek),(row['id'],e)
    cases.append(dict(id=row['id'],epoch=e,plain_correct=hit(pk),era_correct=hit(ek),stage=stage,rationale=note,plain_quote=pquote,era_quote=equote))
assert {(x['id'],x['epoch']) for x in cases}=={(x['id'],e) for x in sample for e in (2,3,4) if x['models'][f'plain_epoch_{e}']['record']['grading']['final']['hit']!=x['models'][f'era_epoch_{e}']['record']['grading']['final']['hit']}
models=['baseline']+[f'{arm}_epoch_{e}' for e in (2,3,4) for arm in ('plain','era')]
labels={m:'Baseline' if m=='baseline' else ('일반 ' if m.startswith('plain') else 'ERA ')+m[-1] for m in models};summary={}
for m in models:
    group=[a for a in anns.values() if a['model']==m];assert len(group)==30
    summary[m]={'correct':sum(a['official_correct'] for a in group),'all2':sum(list(a['scores'].values()).count(2)==3 for a in group),'correct_with_zero':sum(a['official_correct'] and 0 in a['scores'].values() for a in group),'excerpt':sum(a['coverage']!='full_response' for a in group),'components':{k:dict(collections.Counter(str(a['scores'][k]) for a in group)) for k in 'ERA'}}
save(R/'semantic_review.json',dict(scope='사전 선정 30문항. 기존 120응답 검토와 새 일반 풀이 90응답. 긴 응답은 표시한 구간만 검토. assistant 정성검토.',annotations=list(anns.values()),summary=summary,discordant_prespecified_pairs=cases))
images=read(P/'evidence/pilot/image_manifest.json');(R/'explorer_data').mkdir(exist_ok=True)
for short,row in zip(idx,inv):
    base=row['models']['baseline']['record'];item={'id':row['id'],'subject':row['subject'],'question':base['question'],'options':base['options'],'gold':base['reference_answer'],'prompt':base['input_prompt'],'images':['../'+P.name+'/evidence/'+f['path'] for f in images if f['id']==row['id']],'models':{}}
    short['reviewed']=bool(item['images'])
    for m in models:
        v=row['models'][m];item['models'][m]={**short['models'][m],'raw':v['record']['model_response']['raw'],'grading':v['record']['grading']['final'],'semantic':anns.get((row['id'],m))}
    (R/'explorer_data'/f"{row['id']}.js").write_text('window.showItem('+js(item)+');',encoding='utf8')
(R/'explorer_index.js').write_text('const ITEMS='+js(idx)+';',encoding='utf8')
page=(P/'explorer.html').read_text(encoding='utf8')
page=re.sub(r'<header>.*?</header>','<header><h1>같은 MMMU 900문항에서 일반 풀이와 ERA 비교</h1><p>Baseline 및 두 LoRA의 2·3·4epoch · seed 3407</p></header>',page,flags=re.S)
page=re.sub(r'(<select id="model">).*?(</select>)',lambda g:g[1]+''.join('<option value="'+m+'">'+labels[m]+'</option>' for m in models[1:]+['baseline'])+g[2],page)
page=page.replace('<option value="all">전체</option>','<option value="all">전체</option><option value="split">같은 epoch 일반·ERA 정오 갈림</option><option value="plainwins">일반만 정답</option><option value="erawins">ERA만 정답</option>')
page=re.sub(r'const labels=\{.*?\};','const labels='+js(labels)+';',page,count=1)
page=page.replace("case'regress':","case'split':case'plainwins':case'erawins':{let e=$('model').value.slice(-1),p=x.models['plain_epoch_'+e],r=x.models['era_epoch_'+e];return !!p&&!!r&&(p.correct!==r.correct)&&($('filter').value==='split'||($('filter').value==='plainwins'?p.correct:r.correct))}case'regress':")
page=page.replace('사전 선정30문항의120응답(긴12응답은 발췌)','사전 선정 30문항의 210응답(기존 120개 재사용, 새 일반 풀이 90개, 긴 응답은 발췌)').replace('local_verification.json','verification.json')
(R/'explorer.html').write_text(page,encoding='utf8')
print('explorer rows',len(idx),'semantic',len(anns),'discordant paired reviews',len(cases))

