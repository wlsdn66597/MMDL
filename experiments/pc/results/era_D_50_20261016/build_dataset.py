import csv, hashlib, html, json, shutil, string, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
OLD=ROOT.parent/'era_D_first5_20261015'
REV='2cbf8318e07b9ece895bb2ae605e71e38d623264'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jsonl(p,rs):p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs),encoding='utf-8')
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
config=read(ROOT/'provenance/baseline_config.json')
manifest={r['source_id']:r for r in read(ROOT/'image_manifest.json')}
paired=rows(OLD/'paired_train.draft.jsonl'); plain=rows(OLD/'train_plain.draft.jsonl'); era=rows(OLD/'train_era.draft.jsonl'); common=rows(OLD/'common_content.jsonl'); review=rows(OLD/'review_log.jsonl')
for sid,reason in exclusions.items():
    review.append(dict(source_id=sid,owner='D',stage='selection_self_review',status='reject',scope='question',reason=reason,
        image_inspected=sid in ['10323','1048','10525','10585'],reviewer_type='same authoring model',independent_human_review=False))
for q in selected:
    if q['source_id'] not in new_content: continue
    sid=q['source_id']; c=content[sid]; im=manifest[sid]; letter=string.ascii_uppercase[q['original_answer']]
    prompt=config['prompts']['multiple-choice'].format(optional_hint=config['prompts']['optional_hint'].format(hint=q['hint']),
        question=q['question'],lettered_options=''.join(f'{string.ascii_uppercase[i]}. {o}\n' for i,o in enumerate(q['original_options']))).rstrip()
    prompt+='\n'+config['prompts']['suffixes']['multiple-choice']
    general=c['general']+'\nFinal answer: '+letter
    structured='Evidence: '+c['evidence']+'\nRule: '+c['rule']+'\nApplication: '+c['application']+'\nFinal answer: '+letter
    units=[dict(id=f'U{i+1}',meaning=u[0],general_excerpt=u[1],era_excerpt=u[2]) for i,u in enumerate(c['units'])]
    units.append(dict(id=f'U{len(units)+1}',meaning='Identical final-answer letter.',general_excerpt='Final answer: '+letter,era_excerpt='Final answer: '+letter))
    sample='D_ScienceQA_'+sid
    p=dict(sample_id=sample,owner='D',reviewer='C',reviewer_status='pending',
        source_dataset='ScienceQA',source_split='train',source_id=sid,source_revision=REV,
        source_url=f'https://github.com/lupantech/ScienceQA/blob/{REV}/data/scienceqa/problems.json',
        source_license='CC-BY-NC-SA-4.0 (official LICENSE-DATA; retained despite differing mirror card)',
        image_path=im['path'],original_image_path=q['image_path'],image_sha256=im['sha256'],image_pixel_sha256=im['pixel_sha256'],
        image_group_no=q['image_group_no'],image_provenance=im,selection_version=q['selection_version'],source_fingerprint=q['source_fingerprint'],
        original_question=q['question'],original_answer=q['original_answer'],original_options=q['original_options'],
        question=q['question'],hint=q['hint'],options=q['original_options'],options_generated=False,distractor_rationales=None,
        answer_letter=letter,answer_value=q['original_options'][q['original_answer']],input_prompt=prompt,
        common_content=dict(evidence=c['evidence'],rule=c['rule'],application=c['application'],conclusion=dict(letter=letter,value=q['original_options'][q['original_answer']]),verification=c['note']),
        semantic_units=units,response_general=general,response_era=structured,
        generation_method='Codex multimodal authoring from the same image, locked source and common content; Python serialization and structural validation',
        model_id=None,model_revision=None,generation_settings=None,
        model_identity_note='System identifies the author as GPT-6/Codex; exact serving model ID, revision and sampling settings are not exposed. No guessed ID is recorded.',
        prompt_version='D-completion-paired-v1.3-20261016',code_version='D-completion-build-v1; '+q['selection_version'],human_edits=False,
        skill_tags=c['tags'],vic_tag=None,evidence_verification_note=c['note'],rule_reference=dict(note=c['rule_note'],source_id=sid,fields=['lecture','solution']),
        calculation_verification_note='Author-checked image values and comparisons; explicit arithmetic checks are recorded in arithmetic_checks.json where applicable.',
        source_disagreement_note=c['source_disagreement'],quality_status='draft',
        review_checks=dict(image_inspected=True,source_answer_matches=True,semantic_units_self_reviewed=True,independent_human_review=False),
        review_note='작성 모델 자체 검토 완료. C 담당 교차 검수 및 통합 중복 검수 전. 최종 accepted 아님.',overlap_status='See validation_report.json; final overlap clearance pending',
        student_prompt_contract='Existing baseline prompt_for; no system message or ERA instruction. Relative image paths resolve against this package directory.')
    paired.append(p)
    user={'role':'user','content':[{'type':'image','path':im['path']},{'type':'text','text':prompt}]}
    plain.append(dict(sample_id=sample,messages=[user,{'role':'assistant','content':general}]))
    era.append(dict(sample_id=sample,messages=[user,{'role':'assistant','content':structured}]))
    common.append(dict(sample_id=sample,common_content=p['common_content'],semantic_units=units))
    review.append(dict(sample_id=sample,source_id=sid,owner='D',stage='author_self_review',status='draft',
        reviewer_type='same authoring model',independent_human_review=False,note=c['note'],meaning_equivalence='All listed semantic units occur in both responses; author model review only.'))
jsonl(ROOT/'paired_train.draft.jsonl',paired)
jsonl(ROOT/'train_plain.draft.jsonl',plain)
jsonl(ROOT/'train_era.draft.jsonl',era)
jsonl(ROOT/'common_content.jsonl',common)
jsonl(ROOT/'review_log.jsonl',review)

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
