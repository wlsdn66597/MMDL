"""Extend the established first-five builder and preserve its completed records."""
import json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
OLD=R.parent/'era_D_first5_20261015'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
new=read(R/'author_content_new.json')
content={**read(OLD/'author_content.json'),**new}
(R/'author_content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
qs=[json.loads(x) for x in (R/'candidate_queue.jsonl').read_text(encoding='utf-8').splitlines()]
special={
 '10323':'1022와 동일한 Dominica 강조 위치·동일 질문의 근접 중복. 바탕의 다른 국가 색 일부는 다르므로 정확한 해시 중복이라고 주장하지 않음.',
 '1048':'10315와 동일한 세계 지도·좌표선·대륙 라벨 구성의 시각적 중복. 동일 이미지당 1문항 취지에 따라 보수적으로 제외. 파일 해시는 다름.',
 '10525':'이미지는 원하는 품목 표만 포함하며 hint가 지칭하는 실제 두 도시락 사진이 없음. 실제 보유 품목·교환 가능 여부를 이미지로 확인할 수 없어 불완전 자료로 제외.',
 '10585':'hint에 batteries cool이라고 명시되어 있어 냉각 방향을 텍스트만으로 유추 가능. 표의 수치는 정답 방향을 결정하는 데 필수적이지 않음.',
 '10202':'hint가 plant kingdom 소속을 명시하므로 식물 세포의 핵 유무는 분류 지식만으로 판별 가능.',
 '1023':'hint가 예시 fire salamander와 경고색 원리를 설명하며 보기는 동물 이름. 후보 이미지를 읽지 않고도 명명된 동물의 지식으로 답할 수 있어 이미지 의존성 부족.',
 '10270':'hint의 예시는 flying fox이고 보기는 flamingo와 gibbon. 명명된 동물의 비행 적응 지식만으로 결정 가능.',
 '10506':'보기 kangaroo와 red salamander의 이름 및 포유류 분류 지식으로 답할 수 있어 이미지 정보가 필요하지 않음.'}
ex={}
for q in qs:
 sid=q['source_id']
 if q['image_group_no']>196 or sid in new:continue
 text=q['question']
 if sid in special:reason=special[sid]
 elif 'food chain' in text:reason='질문 자체가 producer/consumer의 단계 또는 역할을 지정. 해당 용어의 정의와 보기만으로 답할 수 있으므로 화살표를 읽을 필요가 없음.'
 elif 'weather or the climate' in text:reason='hint 본문에 시간 범위와 기상·기후 정의가 있어 텍스트만으로 정답 결정 가능.'
 elif 'capital of' in text:reason='질문에서 주 이름을 이미 지정. 수도 지식과 보기로 답할 수 있어 위치 이미지를 사용할 필요가 없음.'
 elif any(x in text for x in ['scientific name','common name','same genus','same species']):reason='hint의 학명·일반명 문자열과 보기 대조만으로 정답 결정 가능. 이미지가 새 결정 정보를 제공하지 않음.'
 elif 'ecosystem' in text:reason='hint에 사막·초원·산림 등 생물군계가 명시되어 해당 기후·토양 성질 지식으로 보기 판별 가능. 정답을 위한 이미지 의존성 부족.'
 elif any(x in text for x in ['experiment','test show']):reason='hint에 조작·통제 조건과 측정 대상이 서술되어 실험 질문·변수·검사 가능 범위를 텍스트만으로 판별 가능.'
 else:raise ValueError(sid)
 ex[sid]=reason
assert len(ex)==39
(R/'new_exclusions.json').write_text(json.dumps(ex,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
shutil.copytree(OLD/'provenance',R/'provenance',dirs_exist_ok=True)
for sid in list(content)[:5]:shutil.copy2(OLD/'images'/f'{sid}.png',R/'images'/f'{sid}.png')
# Keep extraction's full acquisition manifest as an audit record; final manifest only selected images.
if not (R/'acquisition_image_manifest.json').exists():shutil.copy2(R/'image_manifest.json',R/'acquisition_image_manifest.json')
manifest={x['source_id']:x for x in read(OLD/'image_manifest.json')+read(R/'acquisition_image_manifest.json')}
(R/'image_manifest.json').write_text(json.dumps([manifest[sid] for sid in content],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
oldcode=(OLD/'build_dataset.py').read_text(encoding='utf-8')
core=oldcode[oldcode.index("config=read(ROOT/'provenance/baseline_config.json')"):oldcode.index("if (ROOT/'revisions/before_length_review/token_report.json').exists():")]
core=core.replace('paired=[]; plain=[]; era=[]; common=[]; review=[]',"paired=rows(OLD/'paired_train.draft.jsonl'); plain=rows(OLD/'train_plain.draft.jsonl'); era=rows(OLD/'train_era.draft.jsonl'); common=rows(OLD/'common_content.jsonl'); review=rows(OLD/'review_log.jsonl')")
core=core.replace('image_inspected=True,reviewer_type',"image_inspected=sid in ['10323','1048','10525','10585'],reviewer_type")
core=core.replace('for q in selected[:5]:','for q in selected:\n    if q[\'source_id\'] not in new_content: continue')
core=core.replace('D-first5-paired-v1.3-20261015','D-completion-paired-v1.3-20261016').replace('D-first5-build-v1','D-completion-build-v1')
core=core.replace("No numerical calculation is required; visual comparison or categorical rule application was checked by the authoring model.","Author-checked image values and comparisons; explicit arithmetic checks are recorded in arithmetic_checks.json where applicable.")
header='''import csv, hashlib, html, json, shutil, string, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=ROOT.parent/'era_D_first5_20261015'
REV='2cbf8318e07b9ece895bb2ae605e71e38d623264'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
def jsonl(p,rs):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\\n' for x in rs),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
content=read(ROOT/'author_content.json')
new_content=read(ROOT/'author_content_new.json')
exclusions=read(ROOT/'new_exclusions.json')
history=OLD/'image_log.csv'
kit=PROJECT/'reports/era_selection_kit_20261015/select_questions.py'
for round_no in range(1,20):
    with history.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;log=list(reader)
    for row in log:
        sid=row['원본문항ID']
        if sid in exclusions:row.update({'상태':'제외','제외범위':'문항','메모':exclusions[sid]})
        elif sid in content:row.update({'상태':'작성완료','사용모델':'Codex / GPT-6 system identity; exact serving model ID unknown',
            '제작방법':'모델 시각 작성·자체검토 + 코드 직렬화·검증; 사람 검수 전',
            '메모':'draft; C 검수 대기; 일반·ERA 동일 공통 내용. '+content[sid]['note']})
    updated=ROOT/f'history_round_{round_no}.csv'
    with updated.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(log)
    active={r['원본문항ID'] for r in log if r['상태']!='제외'}
    if active==set(content):
        shutil.copy2(updated,ROOT/'image_log.csv');break
    out=ROOT/f'selection_round_{round_no}'
    if not out.exists():subprocess.run([sys.executable,str(kit),'select','--owner','D','--data',str(PROJECT/'reports/era_dataset_examples_20261014'),'--history',str(updated),'--out',str(out)],check=True)
    history=out/'image_log.csv'
else:raise RuntimeError('selection did not converge')
qmap={x['source_id']:x for x in rows(OLD/'first5_selected_questions.jsonl')+rows(ROOT/'candidate_queue.jsonl')}
selected=sorted([qmap[sid] for sid in content],key=lambda x:x['image_group_no'])
jsonl(ROOT/'selected_questions.jsonl',selected)
'''
tail='''
assert len(paired)==50 and paired[:5]==rows(OLD/'paired_train.draft.jsonl')
jsonl(ROOT/'new45_paired_train.draft.jsonl',paired[5:])
write(ROOT/'run_info.json',dict(target_total=50,completed_drafts=50,existing_preserved=5,new_questions=45,all_train=True,
    inspected_or_text_screened_candidates=98,excluded_count=48,latest_history='image_log.csv',last_image_group=196,
    selection_version=selected[0]['selection_version'],source_fingerprint=selected[0]['source_fingerprint'],
    original_first5_records_unchanged=True,independent_human_review=False,quality_status='draft'))
write(ROOT/'generation_provenance.json',dict(prompt_version='D-completion-paired-v1.3-20261016',
    model_id=None,model_revision=None,generation_settings=None,human_edits=False,
    model_identity_note=paired[-1]['model_identity_note'],same_author_for_both_conditions=True,
    method='Codex visual inspection and common-content authoring; explicit natural paraphrases, with code helpers for repeated domain patterns; Python serialization and validation.',
    authoring_inputs=['provenance/team_spec_v1.3.txt','authoring_instructions.txt','selected_questions.jsonl','images/'],
    source_metadata_not_in_student_input=['lecture','solution','answer','common_content'],
    files_sha256={n:sha(ROOT/n) for n in ['author_new.py','author_content_new.json','build_dataset.py']},
    gpu_used=False,remote_cpu_work=['pinned cached-parquet extraction with source-field equality checks'],
    token_measurement='Local tokenizers runtime and public pinned tokenizer.json; first-five counts cross-checked against prior Transformers measurements',
    rejected_action='Draft SCP upload to practice PC was rejected by automatic approval review; no draft upload performed, local token measurement used instead.',
    original_first5_provenance='provenance/first5_generation_provenance.json'))
shutil.copy2(OLD/'generation_provenance.json',ROOT/'provenance/first5_generation_provenance.json')
with (ROOT/'dataset_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
    names=['sample_id','source_id','owner','image_group_no','image_path','question','hint','options_json','answer_letter','answer_value','response_general','response_era','quality_status','reviewer_status','source_disagreement_note']
    w=csv.DictWriter(f,fieldnames=names);w.writeheader()
    for p in paired:w.writerow({k:(json.dumps(p['options'],ensure_ascii=False) if k=='options_json' else p.get(k)) for k in names})
esc=html.escape
cards=[]
for p in paired:
    cards.append(f'<article id="{p["source_id"]}"><h2>{p["source_id"]} · 그룹 {p["image_group_no"]} · 정답 {p["answer_letter"]}</h2><div class="grid"><img src="{p["image_path"]}" alt="Original ScienceQA image {p["source_id"]}"><pre>{esc(p["input_prompt"])}</pre></div><div class="grid"><section><h3>일반 풀이</h3><pre>{esc(p["response_general"])}</pre></section><section><h3>ERA 풀이</h3><pre>{esc(p["response_era"])}</pre></section></div><details><summary>공통 내용·검수 메모</summary><pre>{esc(json.dumps(p["common_content"],ensure_ascii=False,indent=2))}</pre><p>{esc(p["source_disagreement_note"] or "원본 정답 불일치 발견 없음; C 검토 대기")}</p></details></article>')
page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>D ScienceQA 50문항 초안</title><style>body{font:16px/1.65 system-ui;background:#f2f5f8;color:#172638;margin:0}main{max-width:1180px;margin:auto;padding:24px}article{background:white;border:1px solid #d5dee8;border-radius:12px;padding:24px;margin:24px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}img{max-width:100%;max-height:560px;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.7 system-ui}section{padding:16px;background:#f3f7fb}nav a{display:inline-block;margin:4px 12px 4px 0}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style><main><h1>D · ScienceQA 50문항</h1><p>기존 5 + 추가 45 · 전부 학습용 · 일반 50 / ERA 50 · draft / C 교차 검수 대기</p><p>원본 질문·hint·보기 순서 유지. 공유 CSV의 작성완료는 초안 작성 완료를 뜻하며 최종 accepted가 아닙니다.</p><nav>'+''.join(f'<a href="#{p["source_id"]}">{p["source_id"]}</a>' for p in paired)+'</nav>'+''.join(cards)+'</main></html>'
(ROOT/'review.html').write_text(page,encoding='utf-8')
print(json.dumps({'unique_questions':50,'responses':100,'excluded_history_rows':48,'first5_preserved':True}))
'''
(R/'build_dataset.py').write_text(header+core+tail,encoding='utf-8')
v=(OLD/'validate_dataset.py').read_text(encoding='utf-8')
v=v.replace('==5','==50').replace("'unique_questions':5","'unique_questions':50").replace("'responses':10","'responses':100").replace("'questions':5","'questions':50")
v=v.replace(", 'Notion publication destination'",'').replace("'Notion publication destination',",'')
(R/'validate_dataset.py').write_text(v,encoding='utf-8')
t=(OLD/'measure_tokens.py').read_text(encoding='utf-8').replace("'counts':counts}","'paired_file_sha256':hashlib.sha256((ROOT/'paired_train.draft.jsonl').read_bytes()).hexdigest(), 'counts':counts}")
t=t.replace('from transformers import AutoTokenizer',"import sys\nsys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.runtime/tokenizers'))\nfrom tokenizers import Tokenizer\nimport tokenizers")
t=t.replace("MODEL = Path('/home/rlatjswo/mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots')/REV", "MODEL = ROOT.parents[1]/'.runtime/qwen_tokenizer'")
t=t.replace('tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)',"assert hashlib.sha256((MODEL/'tokenizer.json').read_bytes()).hexdigest()=='a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7'\ntok = Tokenizer.from_file(str(MODEL/'tokenizer.json'))")
t=t.replace('len(tok.encode(response,add_special_tokens=False))','len(tok.encode(response,add_special_tokens=False).ids)').replace('len(tok.encode(body,add_special_tokens=False))','len(tok.encode(body,add_special_tokens=False).ids)')
t=t.replace("'add_special_tokens':False", "'runtime':'tokenizers '+tokenizers.__version__, 'execution_location':'local', 'add_special_tokens':False")
t=t.replace("(ROOT/'token_report.json').write_text", "old=json.loads((ROOT.parent/'era_D_first5_20261015/token_report.json').read_text(encoding='utf-8'))\nassert counts[:5]==old['counts'], 'Local tokenizer counts differ from prior Transformers result'\nreport['first5_counts_equal_prior_transformers']=True\n(ROOT/'token_report.json').write_text")
t=t.replace('print(json.dumps(report),flush=True)',"print(json.dumps({'count':len(counts),'first5_counts_match':True,'length_flags':[x for x in counts if x['review_length_difference']],'over512':[x for x in counts if x['review_over_512']]}),flush=True)")
(R/'measure_tokens.py').write_text(t,encoding='utf-8')
print('Prepared builder, validator, tokenizer script and',len(ex),'new exclusions')
