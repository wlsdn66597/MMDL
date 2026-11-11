import csv
import hashlib
import html
import json
import math
import shutil
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/era_abcd_prereview_20261021'
OUT=ROOT/'results/era_astra_review_20261021'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def objsha(o):return hashlib.sha256(json.dumps(o,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jsonl(name,rs):(OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
def csvsave(path,rs):
    path.parent.mkdir(parents=True,exist_ok=True)
    headers=['sample_id','owner','reviewer','record_sha256','review_reason','astra_verdict','astra_issues','human_assessment','human_answer','action','notes','family_decision','reviewer_name','reviewed_at','status','revision_recheck']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(rs)
records=rows(BASE/'paired_train.draft.jsonl');byid={r['sample_id']:r for r in records}
batch_manifest=read(OUT/'batch_manifest.json')
results=[];blind=[];provenance=[]
for batch in batch_manifest:
    folder=OUT/'astra'/f'batch_{batch["batch"]}'
    bs=rows(folder/'blind_results.jsonl');rs=rows(folder/'review_results.jsonl')
    assert len(bs)==len(rs)==len(batch['ids'])
    assert {r['sample_id'] for r in rs}=={r['sample_id'] for r in bs}==set(batch['ids'])
    assert all(r['image_inspected'] is True for r in bs)
    assert len({r['sample_id'] for r in rs})==len(rs)
    for r in rs:
        assert r['verdict'] in {'pass','error','uncertain'}
        assert r['record_sha256']==objsha(byid[r['sample_id']])
        assert r['reference_answer_letter']==byid[r['sample_id']]['answer_letter']
        assert set(r['checked_dimensions'])=={'visual_evidence','rule_application','answer_unique','pair_equivalence','image_necessity','format'}
        assert set(r['checked_dimensions'].values())<={'pass','error','uncertain'}
    manifest=read(folder/'review_manifest.json')
    assert manifest['blind_results_sha256']==sha(folder/'blind_results.jsonl')
    bm={r['sample_id']:r for r in bs}
    assert all(r['blind_answer_letter']==bm[r['sample_id']]['answer_letter'] for r in rs)
    results+=rs;blind+=bs
    provenance.append(dict(batch=batch['batch'],rows=len(rs),model_selection='gpt-6-astra',blind_sha256=sha(folder/'blind_results.jsonl'),review_sha256=sha(folder/'review_results.jsonl'),manifest=read(folder/'review_manifest.json')))
assert len(results)==len({r['sample_id'] for r in results})==200
reviews={r['sample_id']:r for r in results};blinds={r['sample_id']:r for r in blind}
jsonl('astra_review_results.jsonl',[reviews[r['sample_id']] for r in records])
jsonl('astra_blind_results.jsonl',[blinds[r['sample_id']] for r in records])
save('astra_run_provenance.json',provenance)
repeated=read(BASE/'repeated_visual_families.json')['groups']
repeated_ids={sid for g in repeated for sid in g['member_ids']}
family_by_id={sid:g for g in repeated for sid in g['member_ids']}
reasons={r['sample_id']:[] for r in records}
for r in records:
    sid=r['sample_id'];v=reviews[sid];b=blinds[sid]
    if v['verdict']!='pass':reasons[sid].append('astra_'+v['verdict'])
    if b.get('answer_letter')!=r['answer_letter']:reasons[sid].append('blind_reference_disagreement')
    if any(x!='pass' for x in v['checked_dimensions'].values()) and v['verdict']=='pass':reasons[sid].append('dimension_requires_review')
    if sid in repeated_ids:reasons[sid].append('repeated_visual_family')
    if sid=='B-chartqa-006':reasons[sid].append('length_exception')
    if sid in {'C_10598','C_10611'}:reasons[sid].append('new_replacement')
sampling=[]
for owner in 'ABCD':
    pool=[r['sample_id'] for r in records if r['owner']==owner and not reasons[r['sample_id']]]
    ranked=sorted(pool,key=lambda sid:hashlib.sha256(('20261021|'+sid).encode()).hexdigest())
    n=math.ceil(.2*len(pool));chosen=ranked[:n]
    for sid in chosen:reasons[sid].append('random_sample_20pct')
    sampling.append(dict(owner=owner,pool_size=len(pool),sample_size=n,selected_ids=chosen,pool_ids=ranked))
save('sampling_manifest.json',dict(seed='20261021',algorithm='ascending SHA256(seed|sample_id)',rate=.2,strata=sampling))
tasks=[];statuses=[]
for r in records:
    sid=r['sample_id'];v=reviews[sid]
    statuses.append(dict(sample_id=sid,record_sha256=objsha(r),astra_verdict=v['verdict'],requires_human_review=bool(reasons[sid]),review_reasons=reasons[sid],human_status='pending' if reasons[sid] else 'not_selected',quality_status='draft',final_acceptance=False,protocol='v1.4'))
    if reasons[sid]:
        issues=' | '.join(i.get('detail_ko',str(i)) for i in v['issues'])
        tasks.append(dict(sample_id=sid,owner=r['owner'],reviewer=r['reviewer'],record_sha256=objsha(r),review_reason=';'.join(reasons[sid]),astra_verdict=v['verdict'],astra_issues=issues,human_assessment='',human_answer='',action='',notes='',family_decision='',reviewer_name='',reviewed_at='',status='pending',revision_recheck=''))
jsonl('review_status.jsonl',statuses);jsonl('human_review_tasks.jsonl',tasks);csvsave(OUT/'human_review_tasks.csv',tasks)
for reviewer in 'ABCD':csvsave(OUT/f'reviewer_{reviewer}'/'checklist.csv',[r for r in tasks if r['reviewer']==reviewer])
# Frozen record hashes reference this unchanged snapshot, never a modified response.
snap=OUT/'reviewed_snapshot';snap.mkdir(exist_ok=True)
for name in ['paired_train.draft.jsonl','source_records.jsonl','visual_group_registry.json','repeated_visual_families.json','token_report.json','validation_report.json','mmmu_overlap_report.json']:
    shutil.copy2(BASE/name,snap/name)
shutil.copytree(BASE/'images',snap/'images',dirs_exist_ok=True)
save('snapshot_manifest.json',dict(source_version='era-abcd-prereview-v1-20261021',paired_file_sha256=sha(snap/'paired_train.draft.jsonl'),dataset_edits_during_review=0,scope='Snapshot metadata retains prior v1.3 human-review plan; current process/status is REVIEW_PROTOCOL.md and review_status.jsonl.'))
counts=dict(Counter(r['verdict'] for r in results))
summary=dict(stage='astra_complete_human_queue_pending',model='gpt-6-astra',model_reviewed=200,verdicts=counts,human_required=len(tasks),mandatory=sum(bool(reasons[r['sample_id']]) and 'random_sample_20pct' not in reasons[r['sample_id']] for r in records),random_sample=sum(len(s['selected_ids']) for s in sampling),human_by_reviewer=dict(Counter(r['reviewer'] for r in tasks)),human_completed=0,not_selected_for_human=200-len(tasks),dataset_responses_changed=0,final_training_release=False,remaining=['human adjudication and sample checks','confirmed corrections/replacements and re-review','separate dev200/eval300 overlap','processor EOS loss-mask preflight','release approval'],browser_visual_check='NOT_RUN_local_file_browser_policy')
save('review_summary.json',summary)
def esc(v):return html.escape(str(v))
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ERA Astra 전수 검수와 사람 확인</title><style>body{font:16px/1.65 system-ui,'Malgun Gothic',sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;background:#f3f6fa;color:#16263c}header,article{background:white;border:1px solid #d9e0ea;border-radius:12px;padding:22px;margin:18px 0}img{max-width:100%;max-height:650px;display:block;margin:16px auto}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;font:14px/1.6 system-ui;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}select,input{padding:8px;font:inherit;max-width:95%}.tag{padding:4px 8px;border-radius:6px;background:#edf2fa}.error{border-left:5px solid #c84242}.uncertain{border-left:5px solid #d39224}small{color:#586879}@media(max-width:760px){.cols{grid-template-columns:1fr}}</style><header><h1>Astra 200문항 검수 결과와 사람 확인 목록</h1>''']
parts.append(f'<p>Astra: 통과 {counts.get("pass",0)} · 오류 후보 {counts.get("error",0)} · 불확실 {counts.get("uncertain",0)}. 사람 확인 대상 <b>{len(tasks)}문항</b>, 실제 확인 0건.</p>')
parts.append('<p><a href="HUMAN_GUIDE.md">팀원용 작성 안내와 다음 단계</a> · <a href="README.md">분석 결과 요약</a></p><p>이번 오류 후보 17개는 모두 이미지 필요성 문제입니다. 원본 질문·보기·힌트·정답은 고정하며, 아래 모델 제안 중 원문 변경은 현재 절차에서 실행하지 않습니다. 해소되지 않으면 교체·보류합니다.</p>')
parts.append('''<p>먼저 이미지와 문제로 답을 판단한 뒤 아래 Astra 판정을 펼치세요. 모델 오류 판정은 사람의 확정 판정이 아닙니다. <b>결과는 CSV에 별도 작성</b>합니다.</p><p><a href="REVIEW_PROTOCOL.md">새 절차</a> · <a href="human_review_tasks.csv">전체 사람 확인 CSV</a> · <a href="review_summary.json">수량 요약</a></p><p>담당자별 CSV: <a href="reviewer_A/checklist.csv">A</a> · <a href="reviewer_B/checklist.csv">B</a> · <a href="reviewer_C/checklist.csv">C</a> · <a href="reviewer_D/checklist.csv">D</a></p><label>검수 담당 <select id="reviewer"><option value="">전체</option><option>A</option><option>B</option><option>C</option><option>D</option></select></label> <label>범위 <select id="scope"><option value="human">사람 확인 대상</option><option value="all">전체 200문항</option><option value="error">오류 후보</option><option value="uncertain">불확실</option><option value="sample">무작위 표본</option></select></label> <input id="query" placeholder="문항 ID 또는 질문"><span id="count"></span></header>''')
for r in records:
    sid=r['sample_id'];v=reviews[sid];b=blinds[sid];reason=reasons[sid]
    parts.append(f'<article id="{esc(sid)}" class="{v["verdict"]}" data-reviewer="{r["reviewer"]}" data-verdict="{v["verdict"]}" data-human="{int(bool(reason))}" data-sample="{int("random_sample_20pct" in reason)}" data-search="{esc(sid+" "+r["question"])}"><h2>{esc(sid)}</h2><span class="tag">작성 {r["owner"]} → 검수 {r["reviewer"]}</span><p>{esc("; ".join(reason) if reason else "사람 확인 미선정 · 모델 검수만 완료")}</p><pre>{esc(r["input_prompt"])}</pre><a href="reviewed_snapshot/{esc(r["image_path"])}"><img loading="lazy" src="reviewed_snapshot/{esc(r["image_path"])}" alt="{esc(sid)} 원본"></a><details><summary>Astra 판정과 기존 풀이 열기</summary><p><b>Astra: {esc(v["verdict"])} · 최초 독립 답 {esc(b.get("answer_letter"))} · 원본 정답 {r["answer_letter"]}</b></p><p>{esc(v.get("evidence_ko",""))}</p>')
    if sid in family_by_id:
        family=family_by_id[sid]
        links=' · '.join(f'<a href="#{esc(s)}" onclick="document.querySelector(\'#scope\').value=\'all\';document.querySelector(\'#reviewer\').value=\'\';document.querySelector(\'#query\').value=\'\';filter();">{esc(s)}</a>' for s in family['member_ids'])
        parts.append(f'<p><b>같은 이미지 계열:</b> {links}</p><pre>{esc(json.dumps(family,ensure_ascii=False,indent=2))}</pre>')
    for issue in v['issues']:parts.append(f'<p><b>{esc(issue.get("category","issue"))}</b>: {esc(issue.get("detail_ko",""))}<br>모델 원문 제안(실행 승인 아님): {esc(issue.get("suggested_action_ko",""))}</p>')
    parts.append(f'<div class="cols"><section><h3>일반</h3><pre>{esc(r["response_general"])}</pre></section><section><h3>ERA</h3><pre>{esc(r["response_era"])}</pre></section></div><details><summary>공통 내용과 의미 단위</summary><pre>{esc(json.dumps(dict(common_content=r["common_content"],semantic_units=r["semantic_units"]),ensure_ascii=False,indent=2))}</pre></details><small>record_sha256: {objsha(r)}</small></details></article>')
parts.append('''<script>const cards=[...document.querySelectorAll('article')],who=document.querySelector('#reviewer'),scope=document.querySelector('#scope'),q=document.querySelector('#query');function filter(){let n=0;for(const c of cards){let s=scope.value,ok=(!who.value||c.dataset.reviewer===who.value)&&c.dataset.search.toLowerCase().includes(q.value.toLowerCase())&&(s==='all'||s==='human'&&c.dataset.human==='1'||s==='sample'&&c.dataset.sample==='1'||c.dataset.verdict===s);c.hidden=!ok;if(ok)n++;}document.querySelector('#count').textContent=` ${n}문항`;}who.onchange=scope.onchange=q.oninput=filter;filter();</script></html>''')
(OUT/'review.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
