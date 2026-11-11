"""Build only five paired drafts; retain the full fifty-candidate selection history."""
import csv, hashlib, html, json, shutil, string, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
REV='2cbf8318e07b9ece895bb2ae605e71e38d623264'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jsonl(p,rs): p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
content=read(ROOT/'author_content.json')
exclusions={
 '100':'질문 자체에 chalk가 명시되어 있으며 물질 상태는 명명된 물질에 대한 지식만으로 판단 가능. 그림은 추가 결정 정보를 제공하지 않음.',
 '10023':'hint에 조작 변수(각도)와 측정 변수(착지 거리)가 모두 명시됨. catapult 사진에 정답을 바꾸는 추가 정보가 없음.',
 '10025':'hint에 잎 종류와 섭취 면적 측정이 모두 명시됨. 민달팽이 사진 없이 실험 질문을 판별 가능.',
 '10032':'hint에 Cyanea capillata가 명시되고 원본 보기와 문자열 대조로 정답을 결정 가능. 해파리 사진 불필요.',
 '10044':'이미지에는 예시인 사마귀만 있으며 정답 보기의 동물들은 없음. hint가 녹색 위장을 이미 설명하고 보기 이름으로 판단 가능. 원문 The  is의 누락도 보존하며 임의 수정하지 않음.',
 '10069':'hint에 Camellia japonica가 명시되어 첫 단어를 보기와 대조하면 됨. 동백 사진 불필요.',
 '10084':'이미지는 보기 동물 아닌 hint의 예시 sea otter. fruit bat와 tortoise의 명칭과 분류 지식으로 답할 수 있어 선택 이미지가 불필요.',
 '10088':'질문이 Alaska를 이미 지정하고 그림에 수도 표기나 Juneau 라벨이 없음. 수도에 대한 사실 지식으로만 정답을 결정하므로 그림 불필요.',
 '10105':'fronds라는 잎 명칭을 묻는 정의 문제. 실제 도식에 frond라는 표기가 없고 원문 질문과 지식으로 답할 수 있어 이미지 불필요.'
}
with (ROOT/'selection_v1/image_log.csv').open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f); fields=reader.fieldnames; log=list(reader)
for row in log:
    sid=row['원본문항ID']
    if sid in exclusions:
        row.update({'상태':'제외','제외범위':'문항','메모':exclusions[sid]})
history=ROOT/'selection_v1/image_log.reviewed.csv'
with history.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(log)
latest=ROOT/'selection_v2'
if not latest.exists():
    subprocess.run([sys.executable,str(PROJECT/'reports/era_selection_kit_20261015/select_questions.py'),
        'select','--owner','D','--data',str(PROJECT/'reports/era_dataset_examples_20261014'),
        '--history',str(history),'--out',str(latest)],check=True)
selected=[json.loads(x) for x in (latest/'selected_questions.jsonl').read_text(encoding='utf-8').splitlines()]
assert [x['source_id'] for x in selected[:5]]==list(content)
config=read(ROOT/'provenance/baseline_config.json')
manifest={r['source_id']:r for r in read(ROOT/'image_manifest.json')}
paired=[]; plain=[]; era=[]; common=[]; review=[]
for sid,reason in exclusions.items():
    review.append(dict(source_id=sid,owner='D',stage='selection_self_review',status='reject',scope='question',reason=reason,
        image_inspected=True,reviewer_type='same authoring model',independent_human_review=False))
for q in selected[:5]:
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
        prompt_version='D-first5-paired-v1.3-20261015',code_version='D-first5-build-v1; '+q['selection_version'],human_edits=False,
        skill_tags=c['tags'],vic_tag=None,evidence_verification_note=c['note'],rule_reference=dict(note=c['rule_note'],source_id=sid,fields=['lecture','solution']),
        calculation_verification_note='No numerical calculation is required; visual comparison or categorical rule application was checked by the authoring model.',
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
if (ROOT/'revisions/before_length_review/token_report.json').exists():
    with (ROOT/'review_log.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(stage='length_and_semantic_review',status='draft',source_ids=['10008','10076','10111'],
            note='최초 길이 차이 15% 초과 3문항의 ERA 반복 표현을 축약. 관찰·규칙·조건·적용·정답 의미 단위는 유지. 일반 풀이를 길이 맞추기 위해 늘리지 않음. 수정 전 기록은 revisions/before_length_review에 보존.',
            author='same authoring model',human_edits=False),ensure_ascii=False)+'\n')
jsonl(ROOT/'first5_selected_questions.jsonl',selected[:5])
shutil.copy2(latest/'selected_questions.jsonl',ROOT/'selected_questions.jsonl')
with (latest/'image_log.csv').open(encoding='utf-8-sig',newline='') as f: log=list(csv.DictReader(f))
for row in log:
    sid=row['원본문항ID']
    if sid in content:
        row.update({'사용모델':'Codex / GPT-6 계열(system identity); 정확한 serving ID·revision 미노출',
            '제작방법':'모델 작성·자체검토 + Python 구조검증 (사람 검수 전)', '상태':'작성완료',
            '메모':'draft; 일반·ERA 동시 제작; prompt D-first5-paired-v1.3-20261015; 설정 미노출; 사람 수정 없음; C 검수 대기. '+content[sid]['note']})
with (ROOT/'image_log.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(log)
run=read(latest/'run_info.json')
run.update(completed_drafts=5,remaining_candidates_unreviewed=45,excluded_count=len(exclusions),
    first5_ids=list(content),inspected_candidate_count=14,latest_history='image_log.csv',
    selection_run_record='selection_v2/run_info.json',image_inspection_scope='first 14 original candidates only; replacements metadata selected, not inspected',
    image_file_existence='checked for 14 inspected candidates only',image_content_and_mmmu_overlap='see validation_report.json',
    dataset_partition='all five drafts are train; no internal dev/evaluation split')
write(ROOT/'run_info.json',run)
write(ROOT/'generation_provenance.json',dict(prompt_version='D-first5-paired-v1.3-20261015',
    specification='provenance/team_spec_v1.3.txt',authoring_prompt='authoring_instructions.txt',
    model_id=None,model_revision=None,generation_settings=None,human_edits=False,
    disclosure=paired[0]['model_identity_note'],code_sha256={n:sha(ROOT/n) for n in ['build_dataset.py','author_content.json','validate_dataset.py','measure_tokens.py']},
    same_author_and_sources_for_both_conditions=True,independent_generation_or_review_model=None,
    practice_pc='rlatjswo@aix408-40 (verified Tailscale IP 100.109.154.30)',remote_workdir='/home/rlatjswo/era_D_first5_20261015',
    gpu_used=False,remote_actions=['extract embedded source images with pyarrow','CPU tokenizer measurement']))
# Notion-ready records are prepared locally; no publication or messaging is performed.
with (ROOT/'notion_import.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['문항ID','담당','교차검수자','그룹번호','이미지','정답','상태','모델','제작방법','검수메모']);w.writeheader()
    for p in paired: w.writerow(dict(zip(w.fieldnames,[p['source_id'],'D','C',p['image_group_no'],p['image_path'],p['answer_letter'],'draft',p['model_identity_note'],p['generation_method'],p['review_note']])))
esc=html.escape
cards=[]
for p in paired:
    cards.append(f'''<article id="{p['source_id']}"><h2>{p['source_id']} · 그룹 {p['image_group_no']} · {p['answer_letter']}</h2>
<div class="question"><img src="{p['image_path']}" alt="ScienceQA {p['source_id']} original image"><pre>{esc(p['input_prompt'])}</pre></div>
<div class="pair"><section><h3>일반 풀이</h3><pre>{esc(p['response_general'])}</pre></section><section><h3>ERA 풀이</h3><pre>{esc(p['response_era'])}</pre></section></div>
<details><summary>공통 내용표·의미 단위·검수 메모</summary><pre>{esc(json.dumps(p['common_content'],ensure_ascii=False,indent=2))}</pre><ol>{''.join('<li>'+esc(u['meaning'])+'</li>' for u in p['semantic_units'])}</ol><p>{esc(p['evidence_verification_note'])}</p></details></article>''')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>D · ScienceQA 첫 5문항</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#f2f5f8;color:#172638;margin:0}main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:30px}article{background:white;border:1px solid #d5dee8;border-radius:14px;padding:24px;margin:25px 0}.question,.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}.question{align-items:start}.question img{max-width:100%;max-height:480px;object-fit:contain;justify-self:center}.pair section{background:#f3f7fb;padding:18px;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.7 system-ui,sans-serif}details{margin-top:20px}nav a{margin-right:20px}a{color:#12549a}@media(max-width:750px){.question,.pair{grid-template-columns:1fr}main{padding:16px}article{padding:16px}}</style><main>
<h1>D · ScienceQA 최초 5문항</h1><p>2026-10-15 · 고유 5문항 / 일반 5개 + ERA 5개 · 전부 학습용 · <strong>draft / C 교차 검수 전</strong></p>
<p>원문 질문·hint·보기 순서를 유지했습니다. 공통 후보 50개 중 첫 14개를 확인하고 이미지 불필요 문항 9개를 제외했습니다. 모델 자체 검토와 코드 검증을 사람 검수와 구분합니다.</p>
<nav>'''+''.join(f'<a href="#{sid}">{sid}</a>' for sid in content)+'</nav>'+''.join(cards)+'</main></html>'
(ROOT/'review.html').write_text(page,encoding='utf-8')
print(json.dumps({'draft_ids':list(content),'draft_responses':10,'excluded':len(exclusions),'remaining_unreviewed':45},ensure_ascii=False))
