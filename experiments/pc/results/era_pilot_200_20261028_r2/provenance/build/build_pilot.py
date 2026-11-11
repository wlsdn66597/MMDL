from pathlib import Path
import sys,json,csv,copy,hashlib,math,string,re,ast,importlib.util,shutil,itertools,html,zipfile
from collections import Counter,defaultdict
ROOT=Path('C:/project/MMDL');TMP=ROOT/'tmp/era_pilot_20261027'
BASE=ROOT/'results/ERA_검토_200문항/era_astra_review_20261021';SNAP=BASE/'reviewed_snapshot'
OUT=ROOT/'results/era_pilot_200_20261028';OUT.mkdir(exist_ok=True)
VERSION='era-pilot-v1.5-20261028';DATE='2026-10-28'
sys.path.insert(0,str(ROOT/'scripts/visual_groups'))
sys.path.insert(0,str(ROOT/'.runtime/era_review_deps'))
sys.path.insert(0,str(ROOT/'.runtime/tokenizers'))
from visual_groups import fingerprint,assign,validated_decisions,is_candidate,distance,PROFILE
from tokenizers import Tokenizer
import tokenizers
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def save(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jsonl(name,data):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
def writecsv(name,data,fields=None):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)
def copyto(src,dst):
 p=OUT/dst;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
old=rows(SNAP/'paired_train.draft.jsonl');oldby={r['sample_id']:r for r in old};assert len(oldby)==200
sourcehash=sha(SNAP/'paired_train.draft.jsonl');assert sourcehash==read(BASE/'snapshot_manifest.json')['paired_file_sha256']
human=list(csv.DictReader((BASE/'team_review_20261024/checklist_전체합본.csv').open(encoding='utf-8-sig')))
hby={r['sample_id']:r for r in human};astra={r['sample_id']:r for r in rows(BASE/'astra_review_results.jsonl')}
assert all(r['record_sha256']==digest(oldby[r['sample_id']]) for r in human)
spec=importlib.util.spec_from_file_location('selector',ROOT/'reports/era_selection_kit_20261015/select_questions.py');selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
index={};source_sets={}
for o in 'ABCD':
 src,fp=selector.load_sources(ROOT/'reports/era_dataset_examples_20261014',o);source_sets[o]=(src,fp)
 index[o]={r['source_id']:(n,r) for n,_,batch in selector.grouped(src) for r in batch}
tree=ast.parse((ROOT/'code/evaluation/mmdl-visual-token-analysis/src/evaluate.py').read_text(encoding='utf-8'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prompt_for');ns=dict(ast=ast,math=math,string=string)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'baseline_prompt','exec'),ns)
config=read(ROOT/'results/era_D_50_20261016/provenance/baseline_config.json')
def prompt(r):return ns['prompt_for'](dict(question=r['question'],hint=r['hint'] or None,options=r['options'],question_type='multiple-choice'),config)

replace={
 'A_chartqa_g00073':'A_chartqa_g00105','B-chartqa-036':'B-chartqa-106',
 'C_10067':'C_10618','C_10224':'C_10620','C_10229':'C_10628','C_10313':'C_10636',
 'C_1035':'C_10669','C_10426':'C_10679','C_10527':'C_10680','C_10507':'C_10693',
 'D_ScienceQA_10161':'D_ScienceQA_1062','D_ScienceQA_10448':'D_ScienceQA_10625'}
reasons={
 'A_chartqa_g00073':'3B: highest dip의 최저 골짜기/최대 하락 해석이 해소되지 않아 원문 수정 없이 교체.',
 'B-chartqa-036':'3B: 연령구간5와전체계열7의 해석범위 이견이 남아 보수적으로 교체. 원래 B=5가 확실한 오답이라고 판단한 것은 아님.',
 'D_ScienceQA_10448':'3B: 학생에게 제공되는 Glyptodon 1천만년 전 hint의 사실성 문제가 해소되지 않아 원문 수정 없이 교체. 발가락 정답 자체는 타당.',
 'C_10507':'2B: C_10502와 같은 세계지도·경선-해양 교차 질문이고 보기만 바뀌므로 중복 교체.'}
policy=read(TMP/'adversarial_policy_review.json')
for g in policy['repeated_groups']:
 for sid in g['replace']:reasons.setdefault(sid,'2B: '+g['reason']+' 유지 대표: '+', '.join(g['keep']))
edits={r['sample_id']:r for r in read(TMP/'concentration_edits.json')['edits']}
newcontent=read(TMP/'new_pair_content.json');assert set(newcontent)==set(replace.values())
candidates={r['source_id']:r for r in read(TMP/'candidates.json')}
changed=set(edits)|set(newcontent)
generation={'editor':'Codex assistant','model_id':None,'model_revision':None,'settings':None,'human_editor':False,'method':'Same assistant authors both variants from the original image and shared semantic units; peer AI reviews the pair.','version':VERSION,'date':DATE}
def apply_content(r,s):
 units=[u for u in s['units'] if u[0]!='conclusion'];order=[i for i in s['order'] if s['units'][i][0]!='conclusion']
 # Non-conclusion entries precede conclusion in provided edits.
 r['response_era']='\n'.join(cat.title()+': '+' '.join(u[1] for u in units if u[0]==cat) for cat in ['evidence','rule','application'])+'\nFinal answer: '+r['answer_letter']
 r['response_general']=' '.join(s['units'][i][2] for i in order)+'\nFinal answer: '+r['answer_letter']
 r['common_content']={cat:' '.join(u[1] for u in units if u[0]==cat) for cat in ['evidence','rule','application']}
 r['common_content']['conclusion']={'value':r['answer_value'],'letter':r['answer_letter']}
 counts=Counter();r['semantic_units']=[]
 for cat,e,g in units:
  counts[cat]+=1;r['semantic_units'].append({'id':cat[0].upper()+str(counts[cat]),'category':cat,'description':e,'plain_span':g,'era_span':e})
 r['evidence_verification_note']=s['evidence_note'];r['calculation_verification_note']=s['calculation_note']
 r['generation_method']=generation['method'];r['model_id']=None;r['model_revision']=None;r['generation_settings']=None
 r['prompt_version']=VERSION;r['code_version']=VERSION;r['human_edits']=False;r['pilot_generation']=generation
 r['pair_semantic_status']='AI_rechecked_shared_semantic_units_human_pending'
 r['rule_reference']={'type':'original_image_and_shared_operational_rule','scope':r['common_content']['rule'],'source_reference':'source_records.jsonl'}
active=[];newsource=[]
for o in old:
 if o['sample_id'] in replace:continue
 r=copy.deepcopy(o)
 if r['sample_id'] in edits:apply_content(r,edits[r['sample_id']])
 active.append(r)
for sid,s in newcontent.items():
 cand=candidates[s['source_id']];owner=s['owner'];n,src=index[owner][s['source_id']]
 r={k:None for k in old[0]};r.update(sample_id=sid,owner=owner,reviewer={'A':'B','B':'A','C':'D','D':'C'}[owner],source_dataset=src['dataset'],source_split='train',source_id=src['source_id'],image_path=f'images/{owner}/{Path(cand["local_image"]).name}',image_sha256=cand['image_sha256'],image_group_no=n,selection_version=selector.VERSION,source_fingerprint=source_sets[owner][1],original_question=src['question'],original_answer=src['original_answer'],original_options=src['original_options'],question=src['question'],hint=src['hint'],original_image_path=src['image_path'],submitted_image_path=None,source_reference='source_records.jsonl',skill_tags=s['skill_tags'],options_generated=owner in 'AB',title_ko=sid,question_ko=None,batch_version=VERSION,integration_version=VERSION)
 r['options']=s['options'] if owner in 'AB' else src['original_options'];r['answer_letter']=s['answer_letter'] if owner in 'AB' else string.ascii_uppercase[src['original_answer']];r['answer_value']=r['options'][string.ascii_uppercase.index(r['answer_letter'])]
 r['distractor_rationales']=s.get('distractor_rationales')
 template=next(x for x in old if x['owner']==owner)
 for k in ['source_revision','source_url','source_license']:r[k]=template[k]
 r['input_prompt']=prompt(r);apply_content(r,s)
 r['asset_provenance']=cand['asset_provenance'];r['source_csv_status']='image_verified; table not used' if owner in 'AB' else 'not_applicable'
 if sid=='C_10680':r['source_disagreement_note']='Reference solution incorrectly calls individual options incorrect but its final explanation, original answer and image support July/August. Reference solution is not student input.'
 copyto(cand['local_image'],r['image_path']);active.append(r)
active.sort(key=lambda r:(r['owner'],r['image_group_no']));aby={r['sample_id']:r for r in active}
assert len(active)==len(aby)==200 and Counter(r['owner'] for r in active)==dict.fromkeys('ABCD',50)

# Preserve original artifacts and all historical group members, including retired items.
for o in old:copyto(SNAP/o['image_path'],o['image_path'])
for name in ['paired_train.draft.jsonl','source_records.jsonl','visual_group_registry.json']:copyto(SNAP/name,'provenance/original_'+name)
for name in ['astra_review_results.jsonl','astra_blind_results.jsonl','REVIEW_PROTOCOL.md','TEAM_RULES_v1.4.md']:copyto(BASE/name,'provenance/'+name)
copyto(BASE/'team_review_20261024/checklist_전체합본.csv','provenance/checklist_전체합본_원본.csv')
for name in ['adversarial_policy_review.json','concentration_edits.json','new_pair_review.json','new_pair_content.json','candidates.json']:copyto(TMP/name,'provenance/'+name)
for name in ['build_pilot.py','prepare_candidates.py','scan_new_images.py']:copyto(TMP/name,'provenance/build/'+name)

previous=read(SNAP/'visual_group_registry.json');images=copy.deepcopy(previous['images'])+read(TMP/'new_fingerprints.json');iby={r['sample_id']:r for r in images}
newdecisions=[]
for p in read(TMP/'new_image_pairs.json'):
 a,b=iby[p['left']],iby[p['right']];same=a['pixel_sha256']==b['pixel_sha256']
 if same:reason='고정 원본 ChartQA10157/10159 파일·픽셀이 동일. 실제 원본 대조. 최대값선택과 범례식별은 별개과제로 유지.'
 elif 'C_10618' in [p['left'],p['right']]:reason='원본 대조: 신규노랑3/4입자·20mL가 기존녹색2/5·25mL 또는노랑다른개수·50mL와 다름.'
 else:reason='원본 지도 대조: 바탕은 같지만 표시된 주 이름과 표시위치 조합이 다름. 관찰한 라벨목록은 image_identity_observations.json 참조.'
 newdecisions.append({'left':p['left'],'right':p['right'],'left_sha256':a['file_sha256'],'right_sha256':b['file_sha256'],'verdict':'same' if same else 'different','reviewer':'Codex AI','reviewed_at':DATE,'reason':reason})
decisions=validated_decisions(images,previous['decisions']+newdecisions);groups=assign(images,decisions,previous)
priorids={r['sample_id']:r['visual_group_id'] for r in previous['images']}
assert all(iby[sid]['visual_group_id']==gid for sid,gid in priorids.items())
regid='VGR_'+digest({'members':[(r['sample_id'],r['file_sha256'],r['visual_group_id']) for r in images],'decisions':decisions})[:24]
decision_map={tuple(sorted((d['left_pixel_sha256'],d['right_pixel_sha256']))):d['verdict'] for d in decisions}
allpairs=[]
for a,b in itertools.combinations(images,2):
 if a['pixel_sha256']==b['pixel_sha256'] or is_candidate(a,b):
  status='same' if a['visual_group_id']==b['visual_group_id'] else decision_map.get(tuple(sorted((a['pixel_sha256'],b['pixel_sha256']))),'pending')
  allpairs.append({'left':a['sample_id'],'right':b['sample_id'],'status':status,**distance(a,b)})
assert not any(p['status']=='pending' for p in allpairs)
for r in images:
 r['active_in_pilot']=r['sample_id'] in aby
 r['image_path']=(aby.get(r['sample_id']) or oldby[r['sample_id']])['image_path']
for g in groups:g['active_member_ids']=[s for s in g['member_ids'] if s in aby]
save('visual_group_registry.json',dict(schema_version='visual-groups-v1',registry_id=regid,previous_registry_id=previous['registry_id'],hash_profile=PROFILE,images=images,groups=groups,decisions=decisions,candidates=allpairs,unresolved_pairs=0,scope='All previous 200 registry members plus 12 replacements; active membership is separate. AI identity review, not human approval.'))
labels={
 'C_1':['Arizona','Oklahoma','Louisiana','West Virginia'],'C_10123':['South Dakota','Maine','Illinois','Kansas'],'C_10208':['Washington','South Dakota','California','Maine'],'C_10366':['Missouri','Arizona','Louisiana','North Carolina'],'C_10546':['Montana','Oregon','New Mexico','Kansas'],
 'D_ScienceQA_10008':['Kansas','Kentucky','New Mexico','New Jersey'],'D_ScienceQA_10233':['New York','Kentucky','North Carolina','Arkansas'],'D_ScienceQA_10239':['Nebraska','Indiana','New Jersey','Texas'],'D_ScienceQA_10328':['Montana','New Jersey','West Virginia','Alabama'],'D_ScienceQA_1042':['Tennessee','Louisiana','Florida','West Virginia'],'D_ScienceQA_10472':['Arizona','Delaware','Rhode Island','Vermont'],'D_ScienceQA_10518':['Kentucky','Arkansas','Alabama','Florida'],
 'C_10636':['Idaho','California','Mississippi','Georgia'],'C_10679':['Michigan','Maryland','New Jersey','Vermont'],'C_10693':['Colorado','Ohio','New Jersey','Massachusetts'],'D_ScienceQA_10625':['Colorado','North Carolina','Louisiana','Alabama']}
assert len({tuple(sorted(v)) for v in labels.values()})==len(labels)
save('provenance/image_identity_observations.json',{'date':DATE,'reviewer':'Codex AI','basis':'Original new images and two contact sheets covering all 12 old maps visually inspected; particle models viewed at original resolution.','map_labels':labels,'new_pairs':57,'different_pairs':56,'identical_pair':['A_chartqa_g00105','B-chartqa-106']})

# Replay the existing selector with old records, exclusions, and new candidates.
skip={'10615':'원본hint에서 동물이름이 비어 있음(The skin of the  is ...). 3B 원문수정없이 제외.','10624':"원본hint에서 동물이름이 비어 있음(The 's mouth ...). 3B 원문수정없이 제외.",'10632':'지도는 Kentucky영역만 표시하며 수도위치/이름을 제공하지 않아 수도정답의 시각근거가 없음.','10645':'사진과hint만으로 제시된 계절별기후·나무종수 묶음을 검증할 시각근거가 충분하지 않음.','10651':'사진에 속이름이 표시되지 않아 학명문자열비교에 시각근거가 없음.','10674':'snowboarding사진은 실험조건/측정결과를 보여주지 않아 방법비교의 근거가 되지 않음.','10683':'지도에서 수도명/수도위치가 제공되지 않아 수도정답의 시각근거가 없음.','10622':'풍경사진이 매년12~3월의날씨반복을 보여주지 않아 기후판단의시각근거가 없음.'}
selection_report={}
for owner in 'ABCD':
 src,fp=source_sets[owner];history=selector.read_log(ROOT/f'results/era_abcd_prereview_20261021/selection/{owner}/image_log.csv')
 rejected_source={oldby[sid]['source_id']:sid for sid in replace if oldby[sid]['owner']==owner}
 for rec in history:
  if rec['원본문항ID'] in rejected_source:
   sid=rejected_source[rec['원본문항ID']];rec.update(상태='제외',제외범위='이미지',메모=rec['메모']+' | '+DATE+' '+reasons[sid]+' 새문항 '+replace[sid])
 picked=[r for r in active if r['owner']==owner and r['sample_id'] in newcontent];new_by_source={r['source_id']:r for r in picked}
 maximum=max(r['image_group_no'] for r in picked)
 minimum=min(c['image_group_no'] for c in candidates.values() if c['owner']==owner)
 for n,path,batch in selector.grouped(src):
  if not minimum<=n<=maximum or selector.assigned_owner(batch[0]['dataset'],n)!=owner:continue
  selected_item=next((rr for rr in batch if rr['source_id'] in new_by_source),None)
  rr=selected_item or batch[0];rec=selector.new_log(owner,n,rr,fp)
  if selected_item:rec.update(사용모델='Codex assistant; exact serving ID not recorded',제작방법=generation['method'],상태='작성완료',메모='AI 내용검수 완료; 사람 재확인 대기. '+new_by_source[rr['source_id']]['sample_id'])
  else:
   assert rr['source_id'] in skip,rr
   rec.update(상태='제외',제외범위='문항',메모=skip[rr['source_id']])
  history.append(rec)
 log,selected,skipped=selector.select(owner,src,fp,50,history)
 assert {r['source_id'] for r in selected}=={r['source_id'] for r in active if r['owner']==owner}
 writecsv(f'selection/{owner}/image_log.csv',log,selector.HEADERS);jsonl(f'selection/{owner}/selected_questions.jsonl',selected)
 selection_report[owner]={'active':50,'history_rows':len(log),'source_fingerprint':fp,'replay_matches':True}
save('selection_validation.json',selection_report)

# Pair/source invariants and exact response token lengths.
tokenizer=Tokenizer.from_file(str(ROOT/'.runtime/qwen_tokenizer/tokenizer.json'))
def count(s):return len(tokenizer.encode(s,add_special_tokens=False).ids)
def body(s):return re.sub(r'(?m)^(?:Evidence|Rule|Application):\s*','',re.sub(r'\n?Final answer: [A-Z]\s*$','',s)).strip()
input_fields=['question','hint','options','answer_letter','answer_value','input_prompt','image_sha256']
pool=[r for r in active if r['owner']=='C' and r['sample_id'] in oldby and r['sample_id'] not in hby and astra[r['sample_id']]['verdict']=='pass']
extra=sorted(pool,key=lambda r:hashlib.sha256(('20261028|C_expansion|'+r['sample_id']).encode()).hexdigest())[:math.ceil(len(pool)*.2)]
extra_ids={r['sample_id'] for r in extra}
human_needed=set(changed)|extra_ids
def option_matches(value,option):
 if str(value)==str(option):return True
 v=str(value)
 if v.startswith('[') and v.endswith(']'):
  return ' and '.join(x.strip().strip("'\"") for x in v[1:-1].split(','))==option
 m=re.fullmatch(r'([0-9.]+) \(([0-9.]+)%\)',option)
 return bool(m and math.isclose(float(v),float(m[1])) and math.isclose(float(v)*100,float(m[2]),abs_tol=1e-9))
sources=[];checklist=[];tokens=[];reviewlog=[]
special_signatures={'C_10003':'world|prime_meridian|continent_intersection','C_10502':'world|prime_meridian|ocean_intersection','D_ScienceQA_10315':'world|equator|continent_intersection','A_chartqa_g00105':'cellphone_chart|blue_series|maximum_situation','B-chartqa-106':'cellphone_chart|blue_series|legend_meaning'}
for r in active:
 sid=r['sample_id'];n,src=index[r['owner']][r['source_id']]
 assert r['question']==r['original_question']==src['question'] and r['hint']==src['hint']
 assert r['original_answer']==src['original_answer'] and r['original_options']==src['original_options']
 assert r['image_group_no']==n and selector.assigned_owner(src['dataset'],n)==r['owner']
 assert r['input_prompt']==prompt(r);assert option_matches(r['answer_value'],r['options'][string.ascii_uppercase.index(r['answer_letter'])]),sid
 if r['owner'] in 'CD':assert r['options']==src['original_options'] and r['answer_letter']==string.ascii_uppercase[src['original_answer']]
 else:
  assert len(r['options'])==4
  if sid in newcontent:assert str(r['original_answer'])==str(r['answer_value'])
  else:assert r['original_answer']==oldby[sid]['original_answer'] and r['answer_value']==oldby[sid]['answer_value']
 assert len(set(r['options']))==len(r['options']) and r['source_split'] in ['train','train_human']
 assert sha(OUT/r['image_path'])==r['image_sha256']
 if sid in oldby:
  assert all(r[k]==oldby[sid][k] for k in input_fields)
  if sid not in edits:assert all(r[k]==oldby[sid][k] for k in ['response_general','response_era','common_content','semantic_units'])
 for key in ['response_general','response_era']:
  assert r[key].endswith('Final answer: '+r['answer_letter']) and len(re.findall(r'(?m)^Final answer:',r[key]))==1
 assert all(r['response_era'].count(k+':')==1 for k in ['Evidence','Rule','Application'])
 assert not re.search(r'(?m)^(Evidence|Rule|Application):',r['response_general'])
 if sid in changed:
  for u in r['semantic_units']:assert u['plain_span'] in r['response_general'] and u['era_span'] in r['response_era']
 r['visual_group_id']=iby[sid]['visual_group_id'];r['visual_group_registry_id']=regid;r['visual_group_registry']='visual_group_registry.json';r['image_pixel_sha256']=iby[sid]['pixel_sha256']
 r['pilot_version']=VERSION;r['pilot_split']='train';r['quality_status']='pilot_candidate';r['final_training_release']=False
 r['review_checks']={'source_fields_checked':True,'input_pair_checked':True,'format_checked':True,'independent_human_review':'pending_recheck' if sid in changed else 'pending_additional_sample' if sid in extra_ids else hby.get(sid,{}).get('status','not_selected'),'final_acceptance':False}
 r['review_note']='새 기준 적용 파일럿 준비본. 이전 기록은 provenance에 보존. 변경본 사람 재확인 및 릴리스 검증 대기.'
 r['reviewer_status']=r['review_checks']['independent_human_review']
 r['overlap_status']={'internal':'resolved_with_two_multitask_image_groups','mmmu_validation':'screened_pinned_reference','separate_dev200':'not_run_missing_manifest','separate_eval300':'not_run_missing_manifest','pretraining_contamination':'not_assessed'}
 r['task_signature']=special_signatures.get(sid,'task_'+digest({'group':r['visual_group_id'],'question':r['question'],'options':r['options']})[:20])
 r['previous_record_sha256']=digest(oldby[sid]) if sid in oldby else None
 oldh=hby.get(sid);r['prior_human_record']=copy.deepcopy(oldh) if oldh else None
 r['prior_astra_review']=copy.deepcopy(astra.get(sid))
 r['ai_review_scope']='new_content_peer_review' if sid in newcontent else 'concentration_type_image_and_pair_recheck' if sid in edits else 'prior_Astra_content_review_plus_current_policy_and_integrity_checks'
 r['changed_content_requires_new_review']=sid in changed
 bg=sid in oldby and any(i.get('category')=='image_necessity' for i in astra[sid].get('issues',[]))
 r['background_knowledge_compatible']=bool(bg or 'map_direction' in (r['skill_tags'] or []))
 gb,eb=count(body(r['response_general'])),count(body(r['response_era']));ratio=(max(gb,eb)-min(gb,eb))/min(gb,eb)
 r['token_review']={'tokenizer':'Qwen/Qwen3-VL-4B-Instruct','revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17','tokenizer_sha256':sha(ROOT/'.runtime/qwen_tokenizer/tokenizer.json'),'general_total':count(r['response_general']),'era_total':count(r['response_era']),'general_body':gb,'era_body':eb,'body_difference_ratio':ratio,'over_15_percent_review':ratio>.15,'measurement':'Response strings only; actual processor/EOS/loss mask not verified.'}
 r['token_review']['tokenizers_version']=tokenizers.__version__
 r['length_check']=r['token_review'];tokens.append(dict(sample_id=sid,**r['token_review']))
 sources.append(dict(sample_id=sid,**src,source_revision=r['source_revision'],image_group_no=n,source_fingerprint=r['source_fingerprint']))
 reviewlog.append({'sample_id':sid,'new_record_sha256':digest(r),'previous_record_sha256':r['previous_record_sha256'],'content_changed':sid in changed,'ai_review_scope':r['ai_review_scope'],'human_recheck':'pending' if sid in changed else 'pending_additional_sample' if sid in extra_ids else 'not_relabelled','date':DATE})

for o in old:
 sid=o['sample_id'];action='replace' if sid in replace else 'revise' if sid in edits else 'keep';target=replace.get(sid,sid);r=aby[target];h=hby.get(sid,{})
 reason=reasons[sid] if sid in replace else edits[sid]['reason'] if sid in edits else '새 기준에서 실질 오류 없음. 기존 내용 유지.'
 if action=='keep' and h.get('action')=='replace':
  reason='1A: 배경지식으로도 풀린다는 이유만으로 제외하지 않음. 이미지 근거·정답이 타당한 기존 내용 유지.' if 'astra_error' in h.get('review_reason','') else '2B: 기존 쌍의 대표로 유지. 상대 검토자의 이전 replace 의견은 원본 이력에 보존.'
 checklist.append({'sample_id':sid,'owner':o['owner'],'previous_assessment':h.get('human_assessment',''),'previous_action':h.get('action',''),'previous_notes':h.get('notes',''),'policy_action':action,'policy_reason':reason,'pilot_sample_id':target,'answer_letter':r['answer_letter'],'visual_group_id':r['visual_group_id'],'old_record_sha256':digest(o),'pilot_record_sha256':digest(r),'human_status_original':h.get('status','not_selected'),'current_human_recheck':'pending' if target in changed else 'pending_additional_sample' if target in extra_ids else 'original_record_preserved','policy_version':VERSION})
writecsv('checklist.csv',checklist);jsonl('policy_decisions.jsonl',checklist)
jsonl('paired_train.pilot.jsonl',active);jsonl('source_records.jsonl',sources);jsonl('review_log.jsonl',reviewlog);save('token_report.json',tokens)
jsonl('astra_blind_input_19.jsonl',[{k:r[k] for k in ['sample_id','image_path','question','hint','options']} for r in active if r['sample_id'] in changed])
jsonl('astra_reference_19.jsonl',[r for r in active if r['sample_id'] in changed])
for variant,name in [('general','train_plain.pilot.jsonl'),('era','train_era.pilot.jsonl')]:
 target=[]
 for r in active:target.append({'sample_id':r['sample_id'],'visual_group_id':r['visual_group_id'],'visual_group_registry_id':regid,'quality_status':'pilot_candidate','messages':[{'role':'user','content':[{'type':'image','path':r['image_path']},{'type':'text','text':r['input_prompt']}]},{'role':'assistant','content':r['response_'+variant]}]})
 jsonl(name,target)
plain=rows(OUT/'train_plain.pilot.jsonl');era=rows(OUT/'train_era.pilot.jsonl')
assert all(a['sample_id']==b['sample_id'] and a['messages'][0]==b['messages'][0] for a,b in zip(plain,era))
assert all(set(m['messages'][0]['content'][1])=={'type','text'} for m in plain)

# Recheck the complete active set against the pinned MMMU validation reference.
refs=read(ROOT/'reports/era_abcd_review_20261020/mmmu_reference_fingerprints.json');reference_path=ROOT/'results/mmmu_first_baseline_20261009/baseline_seed3407/results.json'
old_overlap=read(ROOT/'reports/era_abcd_review_20261020/mmmu_overlap_full.json');assert sha(reference_path)==old_overlap['reference_sha256']
refquestions=read(reference_path)['records'];norm=lambda s:' '.join(re.findall(r'\w+|[%+−-]',str(s).lower()))
mm=[];qt=[]
for r in active:
 for b in refs:
  a=iby[r['sample_id']]
  if a['pixel_sha256']==b['pixel_sha256'] or is_candidate(a,b):mm.append({'sample_id':r['sample_id'],'reference_id':b['id'],**distance(a,b)})
 for q in refquestions:
  if norm(r['question'])==norm(q['question']):qt.append([r['sample_id'],q['id']])
assert not mm and not qt
save('overlap_report.json',{'reference_dataset':old_overlap['dataset'],'reference_sha256':sha(reference_path),'reference_questions':len(refquestions),'reference_images':len(refs),'active_items_checked':200,'image_candidates':mm,'exact_question_matches':qt,'separate_dev200':'not_run_missing_manifest','separate_eval300':'not_run_missing_manifest','limitation':'This fingerprint and exact-question screen does not prove absence of semantic overlap or pretraining contamination.'})

# New human work is explicit and never populated with invented reviewer identities.
pool=[r for r in active if r['owner']=='C' and r['sample_id'] in oldby and r['sample_id'] not in hby and astra[r['sample_id']]['verdict']=='pass']
extra=sorted(pool,key=lambda r:hashlib.sha256(('20261028|C_expansion|'+r['sample_id']).encode()).hexdigest())[:math.ceil(len(pool)*.2)]
human_needed=set(changed)|{r['sample_id'] for r in extra}
reviewqueue=[]
for r in active:
 sid=r['sample_id']
 if sid not in human_needed:continue
 why=[]
 if sid in newcontent:why.append('신규 교체 문항 원문·이미지·두풀이 확인')
 if sid in edits:why.append('농도 필수 규칙 수정본 재확인')
 if sid in {x['sample_id'] for x in extra}:why.append('C 실질 오류에 따른 추가20% 표본')
 reviewqueue.append({'sample_id':sid,'owner':r['owner'],'reviewer':r['reviewer'],'record_sha256':digest(r),'reason':'; '.join(why),'human_assessment':'','human_answer':'','action':'','notes':'','reviewer_name':'','reviewed_at':'','status':'pending'})
writecsv('사람_재확인.csv',reviewqueue)
save('human_review_scope.json',{'old_recorded_comments':len(human),'old_status_counts':dict(Counter(r['status'] for r in human)),'new_or_edited_content':len(changed),'C_additional_pool':len(pool),'C_additional_count':len(extra),'C_additional_sample_ids':[r['sample_id'] for r in extra],'selection_rule':'SHA256(20261028|C_expansion|sample_id) ascending, ceil(20%); no resampling','human_queue_count':len(reviewqueue),'old_127_expansion':'Image-necessity-only basis superseded; re-evaluate rather than mark completed.','conditional_expansion':'If a human confirms the same substantive concentration-rule error in D, the v1.4 two-author trigger expands remaining unreviewed pass items.','administrative_records':'Original pending states and date-only timestamps are retained; owners must supply real completion/confirmation, not invented times.'})
writecsv('image_groups.csv',[{'sample_id':r['sample_id'],'owner':r['owner'],'image_group_no':r['image_group_no'],'visual_group_id':r['visual_group_id'],'task_signature':r['task_signature'],'split':'train','image_path':r['image_path'],'image_sha256':r['image_sha256']} for r in active])
activegroups=defaultdict(list)
for r in active:activegroups[r['visual_group_id']].append(r['sample_id'])
multis={k:v for k,v in activegroups.items() if len(v)>1};assert sorted(map(len,multis.values()))==[2,3]
save('multitask_image_groups.json',{'groups':[{'visual_group_id':k,'member_ids':v,'task_signatures':[aby[s]['task_signature'] for s in v],'reason':'동일 이미지에서 서로 다른 판단 대상/축/연산을 요구하여 새2B에 따라 함께 유지.','split':'train'} for k,v in multis.items()]})
assert all(len({aby[s]['task_signature'] for s in v})==len(v) for v in multis.values())

pending=['변경·신규19문항의 Astra 재검수(현재 Codex 동료 AI 검토 완료)','변경·신규 문항 및 추가 표본의 사람 확인','기존 미완료/날짜 기록의 담당자 확인','확정 dev200/eval300과 문항·이미지그룹 중복 검사','실제 학습 processor/chat-template 길이·EOS·loss mask 검증','최종 릴리스 승인']
report={'version':VERSION,'date':DATE,'source_snapshot_sha256':sourcehash,'items':200,'responses':400,'owners':dict(Counter(r['owner'] for r in active)),'datasets':dict(Counter(r['source_dataset'] for r in active)),'all_train':True,'decisions':dict(Counter(r['policy_action'] for r in checklist)),'replacement_map':replace,'unchanged_content':len(active)-len(changed),'revised_existing':len(edits),'new_items':len(newcontent),'active_visual_groups':len(activegroups),'multitask_groups':multis,'registry_members_including_retired':len(images),'registry_history_preserved':True,'new_similarity_candidates':57,'unresolved_image_pairs':0,'source_preservation_checks':200,'input_pair_equality_checks':200,'image_hash_checks':200,'selection_replay':selection_report,'token_over_512':[t['sample_id'] for t in tokens if max(t['general_total'],t['era_total'])>512],'token_body_over_15pct':[t['sample_id'] for t in tokens if t['over_15_percent_review']],'human_recheck_count':len(reviewqueue),'final_training_release':False,'pending':pending,'checks_not_claimed':['New Astra-model-specific review was not run; new content reviewed by a peer Codex AI.','No new human review is claimed.','Actual model processor training and SFT were not run.']}
save('validation_report.json',report);save('release_status.json',{'status':'pilot_prepared_pending_validation','final_training_release':False,'pending':pending})

# Self-contained review interface: only local images, no external script or source notes in model input.
esc=lambda x:html.escape(str(x))
css='body{font-family:Malgun Gothic,Arial,sans-serif;background:#f4f7fa;color:#18344b;max-width:1300px;margin:30px auto;padding:0 22px}header,article{background:white;border:1px solid #d7e2e9;border-radius:10px;padding:24px;margin:18px 0}h1{font-size:28px}h2{font-size:21px}p,li{line-height:1.7}.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{max-width:100%;max-height:650px;object-fit:contain}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.7 Consolas,Malgun Gothic,monospace}input,select{padding:10px;margin:5px}small{color:#52677a}a{color:#176b7a}.changed{border-left:5px solid #176b7a}@media(max-width:800px){.cols{grid-template-columns:1fr}}@media print{header input,header select{display:none}article{break-before:page;border:0}.cols{display:block}}'
parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>ERA 파일럿 200문항 검수</title><style>'+css+'</style><body><header><h1>ERA 파일럿 200문항 준비본</h1><p>기준 v1.5 · 2026-10-28 · ChartQA 100 + ScienceQA 100 · 모두 학습용</p><p>원본 200개 중 유지 181 · 풀이 수정 7 · 교체 12. 새 이미지 그룹 중복 검사는 완료했으며, 사람 재확인과 별도 dev/eval 대조 등 릴리스 조건은 남아 있습니다.</p><p><a href="노션_공유_검토기준.md">노션 공유용 기준</a> · <a href="checklist.csv">기존 200개 판정표</a> · <a href="사람_재확인.csv">사람 재확인 목록</a></p><input id="q" placeholder="문항 ID 또는 문제 검색"><select id="kind"><option value="all">전체 200개</option><option value="changed">수정·신규 19개</option><option value="human">사람 재확인 대상</option></select><span id="count"></span></header>']
inverse={v:k for k,v in replace.items()}
for r in active:
 sid=r['sample_id'];label='신규 교체' if sid in newcontent else '풀이 수정' if sid in edits else '내용 유지'
 parts.append(f'<article data-changed="{int(sid in changed)}" data-human="{int(sid in human_needed)}" class="{"changed" if sid in changed else ""}"><h2>{esc(sid)} · {label}</h2><p>담당 {r["owner"]} / 검토 {r["reviewer"]} · 원본 번호 {r["image_group_no"]} · {esc(r["visual_group_id"])}</p>')
 if sid in inverse:parts.append(f'<p>교체 이전: {esc(inverse[sid])} · {esc(reasons[inverse[sid]])}</p>')
 if sid in edits:parts.append('<p>'+esc(edits[sid]['reason'])+'</p>')
 parts.append(f'<div class="cols"><div><img src="{esc(r["image_path"])}" alt="{esc(sid)} 원본 이미지"></div><div><h3>원본 문제</h3><pre>{esc(r["question"])}</pre>')
 if r['hint']:parts.append('<h3>원본 힌트</h3><pre>'+esc(r['hint'])+'</pre>')
 parts.append('<ol type="A">'+''.join('<li>'+esc(o)+'</li>' for o in r['options'])+'</ol></div></div><details><summary>정답과 두 풀이 보기</summary><p>정답 '+r['answer_letter']+' · '+esc(r['answer_value'])+'</p><div class="cols"><section><h3>일반 풀이</h3><pre>'+esc(r['response_general'])+'</pre></section><section><h3>ERA 풀이</h3><pre>'+esc(r['response_era'])+'</pre></section></div><p>'+esc(r['evidence_verification_note'])+'</p></details><small>검토 범위: '+esc(r['ai_review_scope'])+'<br>새 레코드 해시: '+digest(r)+'</small></article>')
parts.append('<script>const q=document.getElementById("q"),k=document.getElementById("kind");function f(){let n=0;document.querySelectorAll("article").forEach(a=>{let ok=a.textContent.toLowerCase().includes(q.value.toLowerCase())&&(k.value==="all"||a.dataset[k.value]==="1");a.hidden=!ok;if(ok)n++});document.getElementById("count").textContent=n+"개 표시"}q.oninput=f;k.onchange=f;f();</script></body></html>')
(OUT/'검수_200문항.html').write_text(''.join(parts),encoding='utf-8')
summary=f'''# ERA 파일럿 200문항 준비본

2026-10-28 / v1.5 / 학습 실행 전 검증 대기

- ChartQA 100 + ScienceQA 100, A/B/C/D 각 50개, 전부 train. 고유 문항 200개, 두 조건 응답 400개.
- 기존 200개 판정: 유지 181, 필수 풀이 수정 7, 교체 12. 기존 스냅샷은 변경하지 않았다.
- 활성 시각 그룹 197개. 세계지도 3문항과 휴대전화 차트 2문항은 서로 다른 과제로 함께 사용한다. 이전 등록 원장은 제외 문항까지 212개 구성원으로 보존했다.
- 질문·hint·원본 보기·원본 정답·이미지 바이트 및 일반/ERA 입력 동일성을 검증했다. 수정·신규 19문항은 공통 의미 단위와 두 풀이를 맞췄다.
- 기존 Astra 200개 검토를 재사용하고 새 내용은 Codex 동료 AI가 재검토했다. 새 Astra 전수 실행이나 새 사람 검수 완료를 뜻하지 않는다.

## 사용 파일

`노션_공유_검토기준.md`를 노션에 붙여 넣거나 Markdown으로 가져온다. 원격 노션에 직접 업로드하지 않았다.

`검수_200문항.html`에서 원본 이미지·문제·두 풀이를 확인한다. `checklist.csv`는 기존 200개 각각의 판정과 교체 연결이다.

`paired_train.pilot.jsonl`은 관리·검수용 전체 레코드다. 학습 변환에는 `train_plain.pilot.jsonl` 또는 `train_era.pilot.jsonl`을 사용한다. 두 파일은 같은 200문항의 동일 입력을 공유한다. 학생 입력에는 messages만 사용하고 정답 메타데이터·검토 메모를 추가하지 않는다. 이 파일은 모델별 학습 실행 설정까지 검증된 릴리스가 아니다.

이미지 경로는 이 폴더 기준이다. `visual_group_registry.json`은 기존 200 + 신규 12개 이력을 보존하며, 실제 활성 200개는 paired 파일과 `image_groups.csv`다. `selection/`에는 담당별 선별 이력과 공통 코드 재현 결과가 있다.

## 남은 확인

`사람_재확인.csv`의 {len(reviewqueue)}개를 확인한다(수정·신규 19개와 C 추가 표본의 합집합). 이전 73개 코멘트의 미완료 상태 및 날짜만 있는 기록은 원작성자가 확인한다. 실제 사람 검수 이름·시각을 추정해 채우지 않았다.

변경·신규 19개는 Codex 동료 AI의 내용 검토를 받았다. 기존 프로토콜이 지정한 Astra 재검수는 별도 미완료다. 현재 연결된 Astra 도구는 조회 전용으로 새 실행을 제공하지 않는다. `astra_blind_input_19.jsonl`은 정답·기존 풀이가 없는 독립 풀이 입력이며, 결과를 저장한 뒤 `astra_reference_19.jsonl`과 대조한다. 두 파일을 처음부터 함께 제공하지 않는다.

이미지 필수성만을 이유로 한 127개 전수 확대 근거는 재평가했다. D의 농도 규칙 오류까지 사람이 확정하면 기존 프로토콜의 두 작성자 동일 오류 조건에 따라 미확인 pass 전체 확대가 다시 필요하다.

기존 고정 MMMU validation {len(refquestions)}문항/{len(refs)}이미지와 정확 질문·픽셀·지각 해시 검사는 후보 0개다. 별도 dev 200 / eval 300 명세가 없어 해당 검사는 미실행이다. 지각 해시 후보 0은 모든 의미 중복이나 사전학습 오염 부재를 증명하지 않는다.

실제 모델 processor/chat-template의 길이·EOS·loss mask 확인과 릴리스 승인은 남아 있다. `final_training_release=false`를 유지한다. SFT 학습은 실행하지 않았다.

출처·원본 revision·이용 조건은 각 레코드와 provenance에 보존했다. 배포 시 출처별 이용 조건을 유지한다.
'''
(OUT/'README.md').write_text(summary,encoding='utf-8')
assert sha(SNAP/'paired_train.draft.jsonl')==sourcehash
save('FILE_SHA256.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='FILE_SHA256.json'})
print(json.dumps(report,ensure_ascii=False,indent=2))
