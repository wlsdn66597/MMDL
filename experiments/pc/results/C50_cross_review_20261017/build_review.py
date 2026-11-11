"""Package the completed AI review; never modify source_C or the original archive."""
import csv, hashlib, html, json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name, value): (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def esc(v): return html.escape(str(v), quote=True)
C = [json.loads(x) for x in (OUT/'source_C/paired_train.draft.jsonl').read_text(encoding='utf-8').splitlines()]
auto = read(OUT/'automated_audit.json')
checks = {r['sample_id']:r for r in auto['checks']}
blind = {r[0]:r for r in read(OUT/'blind_observations.json')['rows']}
refs = {
 'kelp': {'title':'Monterey Bay Aquarium — Kelp bass', 'url':'https://www.montereybayaquarium.org/animals-the-ocean/animals-a-to-z/kelp-bass','finding':'먹이는 작은 물고기·오징어·갑각류로 기재되어 있다. 제공 그림의 kelp → kelp bass를 실제 식성의 일반 사실로 가르치는 것은 부적절하다.'},
 'ocean': {'title':'NOAA NCEI — Indian Ocean limits','url':'https://www.ncei.noaa.gov/archive/archive-management-system/OAS/bin/prd/jquery/seaname/details/72','finding':'인도양의 동쪽 경계는 인도네시아·호주·태즈메이니아까지 이어진다. 전체 인도양을 동경 60–100도로 한정한 설명은 맞지 않는다.'},
 'fsm': {'title':'FSM Government — Statement to the 56th UN General Assembly','url':'https://www.fsmgov.org/fsmun/56unga.html','finding':'미크로네시아연방에는 낮은 산호 환초뿐 아니라 높은 화산섬도 포함된다.'},
 'heat': {'title':'OpenStax — Thermal Energy and Work','url':'https://openstax.org/books/physics/pages/12-2-first-law-of-thermodynamics-thermal-energy-and-work','finding':'온도 상승은 열전달뿐 아니라 일에 의해서도 일어날 수 있다. 이 문제의 열전달 모형과 순에너지 유입이라는 조건을 명시해야 한다.'},
 'rhetoric': {'title':'Purdue OWL — Rhetorical Strategies','url':'https://owl.purdue.edu/owl/general_writing/academic_writing/establishing_arguments/rhetorical_strategies.html','finding':'Logos는 논리와 추론을 포함하며 반드시 수치·통계를 요구하지 않는다. Pathos는 공포·욕망에만 한정되지 않고 여러 호소가 함께 나타날 수 있다.'},
 'tonga': {'title':'Tonga Embassy — About Tonga','url':'https://www.tongaembassycn.gov.to/en/14-data-articles-english?start=23','finding':'통가의 북서쪽에 피지, 서쪽에 바누아투가 있다. 지도에서 강조된 섬무리의 상대 위치와 일치한다.'},
 'vanuatu': {'title':'Vanuatu Government — About Vanuatu','url':'https://www.gov.vu/index.php/about/about-vanuatu','finding':'바누아투는 호주 동쪽, 뉴칼레도니아 북동쪽, 피지 서쪽에 위치한다.'},
 'kiribati': {'title':'Australian DFAT — Kiribati country brief','url':'https://www.dfat.gov.au/geo/kiribati/kiribati-country-brief','finding':'키리바시는 적도를 걸쳐 넓게 분포한 길버트·피닉스·라인 제도를 포함한다.'},
}
save('references.json',refs)

# These are visual adjudications after viewing all 60 candidate pairs, not a dHash-only verdict.
groups = [
 ('G01','세계지도',['C_10003','C_10502','C_10507'],['D_ScienceQA_10315']),
 ('G02','통가',['C_10057','C_10527'],[]),
 ('G03','파푸아뉴기니',['C_1035','C_10589'],[]),
 ('G04','유럽',['C_10072'],['D_ScienceQA_10161']),
 ('G05','호주',['C_10229'],['D_ScienceQA_10378']),
 ('G06','자메이카',['C_10067'],['D_ScienceQA_1058']),
 ('G07','키리바시',['C_10224'],['D_ScienceQA_1055']),
 ('G08','마셜제도',['C_10313'],['D_ScienceQA_10262']),
 ('G09','매사추세츠 식민지',['C_10426'],['D_ScienceQA_1038']),
]
group_records=[dict(group_id=g,subject=t,C=c,D=d,verdict='시각적 동일/준중복',action='이미지당 1문항 원칙에 따라 유지 문항을 협의하고 나머지를 대체. 자동 삭제하거나 담당자를 재배정하지 않음.') for g,t,c,d in groups]
save('duplicate_groups.json',group_records)
by_id={sid:g for g in group_records for sid in g['C']}
metrics=read(OUT/'near_image_metrics.json')
for i,m in enumerate(metrics):
    g=next((g for g in group_records if m['left'] in g['C']+g['D'] and m['right'] in g['C']+g['D']),None)
    m.update(visual_verdict='시각적 동일/준중복' if g else '비중복: 같은 양식이나 강조 대상·값·유전자형이 다름',group_id=g['group_id'] if g else None,comparison_panel=f'duplicate_panels/{i//4*4+1:02d}-{min(i//4*4+4,60):02d}.jpg',reviewer='Codex AI visual review')
save('duplicate_pair_review.json',metrics)

issues={
 '1003':dict(status='교체 권고',axis='이미지 필요성',note='질문이 clock라고 이미 밝히고 보기 purring / splashing / ringing 중 시계의 소리는 ringing이다. 그림의 결정적 정보를 읽지 않고도 답이 강하게 결정된다. 공통규칙의 이미지 필수 조건에 부적합할 가능성이 높아 교체를 권고한다.',fix='원문·보기를 바꾸지 말고 C의 다음 적합 홀수 이미지 그룹에서 보충한다. 정답 C 자체는 맞음.',refs=[]),
 '10413':dict(status='보류',axis='원천 도식의 과학적 타당성·규칙',note='그림의 kelp → kelp bass 화살표와 원본 정답 B는 일치한다. 그러나 실제 kelp bass의 식성은 물고기·오징어·갑각류로 소개되어 도식에 의문이 있다. 또한 일반 풀이의 “Producers never receive arrows”는 생산자의 보편적 정의가 아니다. 이 그림의 경로 해석과 실제 생물 지식을 분리해야 한다.',fix='원천 그림의 타당성 확인 또는 문항 교체를 우선 권고. 유지 시 두 풀이 모두 “이 도식에서 kelp로부터 직접 이어진 경로에 따라”라고 한정하고, 생산자를 스스로 유기물을 만드는 생물로 설명한다. 단순 문장 수정으로 원천 도식 문제까지 해결됐다고 처리하지 않는다.',refs=['kelp']),
 '10426':dict(status='수정',axis='이미지 판독·적용',note='강조된 현재의 Massachusetts와 Maine 영역 사이에는 New Hampshire가 있다. 일반 풀이의 “two joined highlighted areas”와 ERA의 “two connected highlighted areas”는 그림과 맞지 않는다. D 정답은 유지한다.',fix='일반 풀이·ERA Application·common_content·semantic_units의 joined/connected를 separated로 고친다. 두 영역이 같은 식민지에 속했다는 사실과 물리적으로 연결됐다는 주장을 구분한다.',refs=[]),
 '10430':dict(status='수정',axis='규칙의 적용 조건',note='4→20°C, 4→22°C 판독과 정답 A는 맞다. 그러나 온도 상승이면 반드시 주변에서 열이 들어왔다거나 역방향 전달이 전혀 없다는 단정은 과도하다. 일이 온도를 높일 수도 있고 유입·유출이 동시에 일어날 수도 있다.',fix='두 풀이에서 이 문제의 단순 열전달 모형을 전제하고 net energy transfer into each greenhouse라고 표현한다. “must have flowed”, “not the other way round” 등의 무조건적 단정을 제거한다.',refs=['heat']),
 '10502':dict(status='수정',axis='지도 판독·불필요한 지리 주장',note='인도양 전체를 약 60°E–100°E에 위치한다고 한정한 문장은 잘못이다. 지도 글자 위치 부근을 바다 전체 범위로 해석한 것으로 보인다. 0°가 북극해를 지난다는 핵심과 정답 A는 맞다.',fix='두 풀이·공통 내용표에서 수치 범위를 삭제하고 인도양은 이 지도에서 0° 동쪽, 아프리카·아시아·호주 사이에 있다고 표현한다.',refs=['ocean']),
 '10507':dict(status='수정',axis='지도 판독·불필요한 지리 주장',note='인도양을 약 60°E–100°E에 한정한 설명이 잘못이다. 0°가 대서양을 지난다는 핵심과 정답 C는 맞다.',fix='C_10502와 동일하게 두 풀이·공통 내용표의 불필요한 경도 범위를 제거한다.',refs=['ocean']),
 '10517':dict(status='수정',axis='Evidence와 Rule 구분',note='Evidence의 “cloth and is not springy”에서 탄성·반발 특성은 정지 사진으로 관찰한 사실이 아니다. 파란색 판독과 정답 A는 맞지만 지식을 관측 근거로 표시했다.',fix='Evidence에는 블루베리·운동복의 외관과 색만 남긴다. 일반적인 천 운동복의 반발 특성은 Rule에 옮겨 조건부로 서술하고 Application에서 적용한다. 일반 풀이에도 같은 내용을 유지한다.',refs=[]),
 '10586':dict(status='수정',axis='대안 배제 논리',note='공정무역을 통한 윤리·신뢰 호소가 주된 appeal이라는 정답 A는 적절하다. 다만 수치·통계가 없으므로 logos가 없고 공포·욕망이 없으므로 pathos가 없다는 식의 배제는 성립하지 않는다. 문제는 primarily를 묻는다.',fix='정의에서 pathos가 공포·욕망에 한정되지 않음을 유지하고, Application은 공정무역 문구가 브랜드의 윤리성을 부각하므로 ethos가 주된 전략이라고 판단한다. 다른 호소의 완전한 부재는 단정하지 않는다.',refs=['rhetoric']),
 '10589':dict(status='수정',axis='배경지식의 정확성',note='FSM을 “tiny atolls”만으로 이루어진 나라처럼 기술했으나 높은 화산섬도 포함된다. New Zealand는 “two main islands”로 표현해야 정확하다. 강조된 PNG 판독과 정답 C는 맞다.',fix='두 풀이·Rule·공통 내용표에서 FSM을 북쪽의 섬나라(높은 섬과 환초 포함), New Zealand를 호주 남동쪽의 두 주요 섬과 주변 섬으로 고친다. 정답에 불필요한 설명을 함께 삭제하는 방법도 가능하다.',refs=['fsm']),
}
optional={
 '10204':'보완 권고: 평균 운동에너지는 평균 속력의 제곱 자체가 아니라 속력 제곱의 평균과 관련된다. 같은 종류의 열평형 기체라는 교과서적 전제에서 B 비교는 맞다. 엄밀한 수치 계산에 평균 속력을 그대로 대입하지 않도록 한다.',
 '10381':'표현 보완 권고: fragile은 쉽게 깨질 수 있다는 성질이다. 떨어뜨리면 반드시 깨진다는 단정 대신 can break easily if dropped로 표현하면 정확하다.',
 '10295':'감각 추론 메모: 사진의 흰 알갱이·식품 종류를 읽는 것과 짠맛 자체를 관찰하는 것은 다르다. 음식에 관한 지식을 Rule로 연결한다. 핵심 답 A와 두 풀이의 핵심 정보는 일치한다.',
}
replacements={
 '10426':{'general_fragment':'The two separated highlighted regions correspond to Massachusetts, which then included the territory that later became Maine.','era_application':'Application: The highlighted Massachusetts and Maine regions are separated by New Hampshire, but both belonged to Massachusetts in the period represented by this map, so the answer is D.'},
 '10430':{'general_fragment':'In the simplified heat-transfer model used by this question, the temperature rise indicates a net gain of energy from the surroundings. Both greenhouses warmed, so the net transfer was into each greenhouse.','era_rule':'Rule: In this question’s simplified heat-transfer model, warming indicates a net gain of energy from the surroundings.','era_application':'Application: Both temperatures rose, so the model indicates net energy transfer from the surroundings into each greenhouse. The answer is A.'},
 '10502':{'both_replace_ocean_clause':'The Indian Ocean lies east of the 0° meridian, between Africa, Asia and Australia.'},
 '10507':{'both_replace_ocean_clause':'The Indian Ocean lies east of the 0° meridian, between Africa, Asia and Australia.'},
 '10517':{'era_evidence':'Evidence: The picture shows a blueberry and a blue tracksuit with white stripes. Both objects are blue.','era_rule':'Rule: Blue describes color. Bouncy objects rebound when dropped; ordinary cloth clothing does not rebound like a ball.','era_application':'Application: The objects share the color blue. The tracksuit would not normally rebound like a ball, so bouncy is not a shared property. The answer is A.','general_fragment':'The blueberry and the tracksuit are both blue. Bouncy means rebounding when dropped, which ordinary cloth clothing does not do like a ball. Their shared property is therefore blue.'},
 '10586':{'era_rule':'Rule: Ethos appeals to character or credibility, logos to reasoning, and pathos to emotion. More than one appeal can occur in an ad.','era_application':'Application: The fair-trade claim mainly presents the brand as ethical and trustworthy, so ethos is the primary appeal and the answer is A.','general_fragment':'The fair-trade wording presents the brand as ethical and trustworthy. That emphasis on character makes ethos the primary appeal, even if the image also evokes feelings.'},
 '10589':{'both_replace_fsm_nz_clause':'The Federated States of Micronesia is an island nation to the north with both high islands and atolls. New Zealand has two main islands and smaller surrounding islands southeast of Australia.'},
}
save('suggested_fixes.json',{'notice':'문장 수정 제안이며 원본 데이터에는 적용하지 않음. 두 응답·공통 내용표·semantic_units를 함께 수정한 후 길이와 의미 일치를 재검사할 것. 최종 답은 유지.','items':replacements})

reviews=[]
for n,p in enumerate(C,1):
    sid=p['sample_id']; src=p['source_id']; c=checks[sid]; issue=issues.get(src); dup=by_id.get(sid)
    content=issue['status'] if issue else '통과'
    overall=content if content in ['교체 권고','보류'] else ('수정' if issue or dup else '통과')
    note=blind[src][2].replace(' 외부 지도 재확인 대상.','')
    row={'index':n,'sample_id':sid,'reviewer':'Codex AI (사람 최종 승인 아님)','answer':p['answer_letter'],'answer_value':p['answer_value'],'source_match':c['source_all_pass'],'image_answer_matches':c['blind_answer_matches'],'image_evidence':note,'blind_exception':src=='1','content_verdict':content,'selection_content_verdict':overall,'pair_semantics':'핵심 관찰·규칙·결론 일치; 공통 오류는 별도 판정','expression':'의미 전달 가능; 수정 사항은 메모 참조' if issue else '대체로 자연스러움','issue':issue,'optional_note':optional.get(src,''),'duplicate_group':dup,'image_path_issue':p['image_path']+' → '+c['resolved_path'],'baseline_prompt_issue':not c['baseline_prompt_exact'],'tokens':c['tokens'],'human_approval':'pending','final_quality_status':'revise' if overall=='수정' else ('hold' if overall=='보류' else ('reject_recommended' if overall=='교체 권고' else 'draft'))}
    reviews.append(row)
(OUT/'review_log.ai.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviews),encoding='utf-8')

# Compatible columns for the author's existing review page; packaging corrections are tracked separately.
cols=['sample_id','검수자','판독(맞음/틀림)','두 응답 정보 일치(예/아니오)','표현(자연스러움/어색)','판정(통과/수정/제외)','메모']
with (OUT/'review_log.ai.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(cols)
    for r in reviews:
        detail=[r['image_evidence']]
        if r['issue']: detail += [r['issue']['axis']+': '+r['issue']['note'],r['issue']['fix']]
        if r['duplicate_group']: detail += ['준중복 '+r['duplicate_group']['group_id']+': '+', '.join(r['duplicate_group']['C']+r['duplicate_group']['D'])+'; 유지·교체 선택 필요']
        if r['optional_note']:detail += [r['optional_note']]
        detail += ['공통: 이미지 상대 경로 수정 필요', '빈 Hint 줄이 baseline과 다름' if r['baseline_prompt_issue'] else 'baseline 프롬프트 일치','AI 검수; 사람 최종 승인 대기. 이 CSV 판정은 내용·선별 기준이며 공통 패키지 수정 및 전체 평가셋 중복 완료를 뜻하지 않음.']
        if r['content_verdict']=='보류':detail.insert(0,'[보류] 기존 CSV에는 보류 값이 없어 수정으로 표시함.')
        verdict={'통과':'통과','수정':'수정','보류':'수정','교체 권고':'제외'}[r['selection_content_verdict']]
        reading='틀림' if r['sample_id'] in ['C_10426','C_10502','C_10507','C_10517'] else '맞음'
        w.writerow([r['sample_id'],r['reviewer'],reading,'예','자연스러움',verdict,' | '.join(detail)])
with (OUT/'review_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['sample_id','정답','정답값','내용판정','중복포함판정','관찰근거','수정·보류사유','중복그룹','선택적보완','입력형식','일반토큰','ERA토큰','사람승인'])
    for r in reviews:w.writerow([r['sample_id'],r['answer'],r['answer_value'],r['content_verdict'],r['selection_content_verdict'],r['image_evidence'],r['issue']['note'] if r['issue'] else '',r['duplicate_group']['group_id'] if r['duplicate_group'] else '',r['optional_note'],'빈 Hint 줄 차이' if r['baseline_prompt_issue'] else '일치',r['tokens']['general_total'],r['tokens']['era_total'],'대기'])

summary={
 'date':'2026-10-17','review_type':'AI-assisted; human final approval pending',
 'content_verdict_counts':dict(Counter(r['content_verdict'] for r in reviews)),
 'including_duplicate_selection_counts':dict(Counter(r['selection_content_verdict'] for r in reviews)),
 'duplicate_C_groups':3,'duplicate_C_affected_rows':7,'duplicate_C_excess_rows':4,
 'duplicate_CD_groups':7,'duplicate_CD_affected_C_rows':9,'duplicate_union_affected_C_rows':len(by_id),
 'visual_duplicate_pairs':sum(m['group_id'] is not None for m in metrics),
 'automated_summary':auto['summary'],
 'limitations':auto['not_covered']+['이미지 유사 후보는 dHash≤4 및 직접 관찰 기준. 모든 변형·계열 누출 부재를 보증하지 않음.','정답 50/50은 주어진 도식 기준. C_10413은 실제 생태 사실과의 충돌로 보류.','C_1은 이전 대화에서 답 노출됨; 나머지 49개는 이번 검수에서 풀이 열람 전 이미지·문제·보기로 답을 기록함.'],
}
save('review_summary.json',summary)

def ref_links(keys):return ' · '.join(f'<a href="{esc(refs[k]["url"])}">{esc(refs[k]["title"])}</a>' for k in keys)
cards=[]
for p,r in zip(C,reviews):
    issue=r['issue']; dup=r['duplicate_group']; flags=[]
    if issue: flags.append(f'<div class="warning"><b>{esc(issue["axis"])}</b><p>{esc(issue["note"])}</p><p><b>조치:</b> {esc(issue["fix"])}</p><p>{ref_links(issue["refs"])}</p></div>')
    if dup: flags.append(f'<p class="warning"><b>준중복 {dup["group_id"]}</b> — {esc(", ".join(dup["C"]+dup["D"]))}. 이미지당 1문항 원칙에 따라 조정.</p>')
    if r['optional_note']:flags.append(f'<p class="optional">{esc(r["optional_note"])}</p>')
    options=''.join(f'<li><b>{esc(k)}.</b> {esc(v)}</li>' for k,v in p['options'].items())
    t=r['tokens']; suggested=replacements.get(p['source_id'])
    fixblock='<details><summary>수정 문장 제안</summary>'+''.join(f'<h4>{esc(k)}</h4><p>{esc(v)}</p>' for k,v in suggested.items())+'</details>' if suggested else ''
    cls='pass' if r['selection_content_verdict']=='통과' else 'flag'
    cards.append(f'''<article id="{esc(p['sample_id'])}" class="item {cls}" data-content="{esc(r['content_verdict'])}" data-dup="{str(bool(dup)).lower()}">
<h2><small>{r['index']:02d}/50</small> {esc(p['sample_id'])} <span class="badge">내용: {esc(r['content_verdict'])}</span>{'<span class="badge">준중복</span>' if dup else ''}</h2>
<div class="question"><img loading="lazy" src="source_C/images/{esc(p['source_id'])}.png" alt="{esc(p['sample_id'])} 원본 이미지"><div><h3>{esc(p['question'])}</h3><p class="hint">{esc(p['hint'])}</p><ul class="options">{options}</ul><p><b>정답 {esc(r['answer'])} — {esc(r['answer_value'])}</b></p><p><b>직접 판독:</b> {esc(r['image_evidence'])}</p></div></div>
{''.join(flags)}
<div class="solutions"><section><h3>일반 풀이</h3><p class="verbatim">{esc(p['response_general'])}</p></section><section><h3>ERA 풀이</h3><p class="verbatim">{esc(p['response_era'])}</p></section></div>
<p class="meta">원본 질문·hint·보기·답·train·배정·선언 이미지 해시 일치 · 두 응답 핵심 정보 일치 · 일반 {t['general_total']} / ERA {t['era_total']} 토큰 · 본문 차이 {t['body_relative_difference']:.1%}</p>
<details><summary>입력·메타데이터 검수</summary><p>이미지 경로: {esc(r['image_path_issue'])}</p><p>{'빈 Hint 줄 누락: baseline과 완전 일치하지 않음.' if r['baseline_prompt_issue'] else '학생 프롬프트 baseline과 완전 일치.'}</p><p>이 쌍의 학생 입력에 원본 답·lecture·solution 혼입은 확인되지 않음. 원본 형식 그대로의 prompt와 원천 질문·보기 구성 대조.</p><pre>{esc(p['common_content']['part1_raw'])}</pre></details>{fixblock}
</article>''')

dup_rows=''.join(f'<tr><td>{g["group_id"]} {esc(g["subject"])}</td><td>{esc(", ".join(g["C"]))}</td><td>{esc(", ".join(g["D"]) or "—")}</td></tr>' for g in group_records)
panel_links=' · '.join(f'<a href="duplicate_panels/{i:02d}-{min(i+3,60):02d}.jpg">{i}–{min(i+3,60)}쌍</a>' for i in range(1,61,4))
references_html=''.join(f'<li><a href="{esc(v["url"])}">{esc(v["title"])}</a> — {esc(v["finding"])}</li>' for v in refs.values())
page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C 50문항 검수 결과</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f1f4f7;color:#17283a;font:16px/1.65 system-ui,"Malgun Gothic",sans-serif}main{max-width:1200px;margin:auto;padding:30px 24px}header,.intro,.item{background:white;padding:28px;border:1px solid #dae1e8;border-radius:14px;margin-bottom:24px}h1{font-size:32px;line-height:1.3}h2{font-size:23px}h3{font-size:18px}small,.meta{font-size:13px;color:#526477}a{color:#12609c}p{margin:.65em 0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:#edf5fb;border-radius:10px;padding:15px}.metric b{font-size:25px;display:block}.question{display:grid;grid-template-columns:minmax(280px,48%) 1fr;gap:24px;align-items:start}.question img{max-width:100%;max-height:570px;object-fit:contain}.solutions{display:grid;grid-template-columns:1fr 1fr;gap:20px}.solutions section{background:#f7f9fb;border:1px solid #e5eaf0;padding:18px;border-radius:8px}.verbatim,.hint{white-space:pre-wrap}.options{list-style:none;padding-left:0}.badge{font-size:12px;background:#edf0f3;border-radius:20px;padding:4px 9px;white-space:nowrap}.warning{border-left:4px solid #c66f13;background:#fff5e9;padding:13px 17px;margin:16px 0}.optional{border-left:4px solid #7d91a7;background:#f0f4f8;padding:12px 16px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce3ea;padding:11px;text-align:left}th{background:#f0f4f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin-top:12px}summary{cursor:pointer;color:#155987}.filters{position:sticky;top:0;padding:12px;background:#142c40;z-index:1;border-radius:8px;margin-bottom:20px;color:white;display:flex;gap:10px;flex-wrap:wrap}input,select{font:inherit;padding:7px;border:0;border-radius:5px}input{flex:1;min-width:180px}.hidden{display:none}.note{color:#5b6875}li{margin:5px 0}@media(max-width:760px){main{padding:12px}.question,.solutions{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}header,.intro,.item{padding:18px}h1{font-size:27px}table{font-size:12px}}@media print{.filters{display:none}.item{break-inside:avoid}.hidden{display:block}}
</style></head><body><main>
<header><p>C 담당 ScienceQA · 2026-10-17 · AI 검수 기록</p><h1>50문항의 정답은 일치합니다.<br>풀이 수정과 이미지 중복 정리가 필요합니다.</h1>
<p>기존 공통규칙 v1.3에 따라 원본 질문·보기·정답, 이미지 판독, 일반/ERA 두 풀이, 계산·규칙·조건, 이미지 필요성, 중복, 입력·응답 형식을 확인했습니다. <b>내용 판정: 통과 41 · 수정 7 · 보류 1 · 교체 권고 1.</b> 중복은 별도 축으로 표시합니다.</p>
<div class="metrics"><div class="metric"><b>50 / 50</b>원본·그림 기준 정답 일치</div><div class="metric"><b>7</b>풀이 수정 문항</div><div class="metric"><b>3 + 7</b>C 내부 / C–D 중복 묶음</div><div class="metric"><b>13</b>중복 관련 C 문항</div></div>
<p class="note">정답 일치는 과학적 사실의 무오류를 뜻하지 않습니다. C_10413은 원본 도식 자체의 타당성 때문에 보류했습니다. 최종 accepted는 작성자 외 사람의 교차 검수와 전체 필수 검증 완료 후에만 가능합니다.</p>
<p><a href="review_log.ai.csv">기존 검수표 형식 CSV</a> · <a href="review_summary.csv">50문항 요약 CSV</a> · <a href="review_log.ai.jsonl">상세 JSONL</a> · <a href="suggested_fixes.json">수정 문장 제안</a> · <a href="automated_audit.json">자동 검증 원자료</a></p></header>
<section class="intro"><h2>먼저 처리할 사항</h2><ol><li><b>C_10413 보류:</b> kelp → kelp bass라는 원천 도식이 실제 식성과 충돌. 원천 확인 또는 교체 권고.</li><li><b>C_1003 교체 권고:</b> clock와 선택지만으로 ringing을 선택할 수 있어 이미지 필요성이 약함.</li><li><b>풀이 7개 수정:</b> C_10426, C_10430, C_10502, C_10507, C_10517, C_10586, C_10589. 아래 문항별 수정 위치와 예문 확인.</li><li><b>이미지 중복 정리:</b> C 내부 7문항의 3묶음은 4문항이 초과한다. C–D 7묶음은 합칠 때 하나씩만 남도록 협의. 같은 바탕 지도라도 강조 국가·숫자가 다른 경우는 동일 이미지로 판정하지 않음.</li></ol>
<p>중복까지 반영한 내용·선별 판정은 통과 32 / 수정 16 / 보류 1 / 교체 권고 1입니다. 이 수는 아래 공통 패키지 문제를 제외한 분류이며, 최종 학습 승인 수가 아닙니다.</p>
<h3>공통 패키지 수정 사항</h3><ul><li><b>50/50 이미지 상대 경로 불일치:</b> JSONL은 train/&lt;id&gt;/image.png를 가리키나 압축 내부는 images/&lt;id&gt;.png. 경로나 파일 구조를 일치시켜야 함.</li><li><b>33/50 학생 프롬프트 차이:</b> 빈 hint의 <code>Hint: \n</code> 행이 생략됨. 나머지 17개는 baseline과 일치. 이미 README에 고지된 차이지만 기존 baseline 고정 규칙 기준으로는 수정 대상.</li><li>100개 응답의 마지막 답 줄·ERA 표제 순서·영어 형식 검사는 통과. 최대 146토큰, 512 초과 0개, 본문 토큰 차이 15% 초과 0쌍. Qwen tokenizer revision과 해시는 자동 검증 원자료에 기록.</li><li>image_log.csv의 작성완료 50개가 제출 ID와 일치(검토 93개 중 제외 43개). 원본 review_log.csv는 검수 결과가 비어 있었음.</li><li>common_content/semantic_units의 [problem]은 실제 문제에 주어진 정보에만 사용. 예: C_10430 열전달 추론 규칙, C_10586 수사학 정의는 배경지식으로 분류해야 함. 응답 수정 시 이 중복 저장 필드들도 함께 동기화할 것.</li></ul></section>
<section class="intro"><h2>이미지 중복 대조</h2><p>파일·RGB 픽셀 해시가 완전히 같은 C 내부 또는 C–D 쌍은 0개였으나, dHash≤4 후보 60쌍을 직접 비교해 14쌍을 시각적 동일/준중복으로 판정했습니다. 확인된 쌍은 픽셀의 99.77% 이상이 정확히 같습니다. 잡음 수준의 차이는 이미지당 1문항 원칙을 충족시키지 않습니다.</p><table><thead><tr><th>묶음</th><th>C 문항</th><th>D 문항</th></tr></thead><tbody>__DUP_ROWS__</tbody></table><p>G01은 C 내부와 C–D 통계에 모두 포함됩니다. C 내부 3묶음과 C–D 7묶음을 합친 서로 다른 묶음은 9개입니다. C의 시각적 고유 이미지는 이 판정 기준으로 46개입니다.</p><details><summary>60쌍 비교 이미지 및 판정 원자료</summary><p>__PANELS__</p><p><a href="duplicate_pair_review.json">60쌍 지표·시각 판정 JSON</a> · <a href="duplicate_groups.json">9개 묶음 JSON</a></p></details></section>
<section class="intro"><h2>검사 범위와 남은 검증</h2><ul><li>ScienceQA train revision <code>2cbf8318e07b9ece895bb2ae605e71e38d623264</code>의 로컬 원본 메타데이터와 질문·hint·보기 순서·정답을 대조. C 홀수 이미지 그룹 배정과 선언 이미지 해시도 일치.</li><li>50개 이미지를 읽고 100개 풀이를 확인. 이번 검수에서 49개는 제출 풀이를 열기 전에 답과 근거를 기록했으며, C_1은 이전 대화에서 답을 본 예외.</li><li>MMMU validation 900문항·982이미지를 독립 재검사. revision <code>98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68</code>. 정규화 질문 일치·Jaccard≥0.8·정확 RGB·dHash≤4 후보 모두 0. 이 검사로 모든 변형·계열 누출이 없음을 보증하지는 않음.</li><li>제출자가 첨부한 MMMU dev 150/test 10,500 무중복 결과는 원자료가 없어 독립 재현하지 못함. A/B 전체 제출물, 팀 별도 개발 200·최종 평가 300 확정 목록과의 검사는 미완료.</li><li>공식 ScienceQA 이미지 전체 아카이브와 바이트 단위 독립 대조는 미실시. 제출 이미지와 메타데이터의 SHA256 일치 및 시각 판독을 확인한 범위임.</li><li>원본 압축·제출 데이터·답은 변경하지 않음. 이 보고서는 AI 검수 보조 결과로, 사람 D의 검수나 최종 accepted를 대신하지 않음.</li></ul><p>기준: <a href="../../reports/era_notion_share_20261015/공통규칙_ABCD.md">공통규칙_ABCD.md</a> (특히 이미지당 1문항, 이미지 필요성, 근거·규칙·적용 정확성, 사람 교차 검수, 필수 평가셋 비교).</p><p>기존 형식 CSV의 “보류”는 지원 값이 없어 “수정”으로 내보내고 메모에 [보류]를 명시했습니다. CSV의 “통과”는 내용·선별 검수상 판정이며, 경로 수정·전체 중복 검증·사람 승인이 완료됐다는 뜻이 아닙니다.</p></section>
<div class="filters"><input id="query" placeholder="문항 ID·내용 검색" aria-label="문항 검색"><select id="filter" aria-label="판정 필터"><option value="all">전체 50문항</option><option value="flag">내용 수정·보류·교체 또는 중복</option><option value="content">내용 수정·보류·교체</option><option value="dup">중복 관련</option><option value="pass">내용·선별 통과</option></select><span id="count">50개</span></div>
__CARDS__
<section class="intro"><h2>추가 사실 확인 출처</h2><ul>__REFERENCES__</ul><p>출처의 해당 지리·생물·개념 설명만 사용했습니다. 문항별 판정은 제출 이미지와 실제 문장을 함께 대조한 검수 판단입니다.</p></section>
</main><script>
const items=[...document.querySelectorAll('.item')],q=document.querySelector('#query'),f=document.querySelector('#filter');function filter(){let n=0;for(const el of items){const mode=f.value;const match=mode==='all'||mode==='flag'&&el.classList.contains('flag')||mode==='content'&&el.dataset.content!=='통과'||mode==='dup'&&el.dataset.dup==='true'||mode==='pass'&&el.classList.contains('pass');const visible=match&&el.textContent.toLowerCase().includes(q.value.trim().toLowerCase());el.classList.toggle('hidden',!visible);n+=visible?1:0}document.querySelector('#count').textContent=n+'개'}q.addEventListener('input',filter);f.addEventListener('change',filter);
</script></body></html>'''
for k,v in {'__DUP_ROWS__':dup_rows,'__PANELS__':panel_links,'__CARDS__':''.join(cards),'__REFERENCES__':references_html}.items():page=page.replace(k,v)
(OUT/'C50_review_report.html').write_text(page,encoding='utf-8')
save('review_artifact_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'C50_review_report.html',OUT/'review_log.ai.csv',OUT/'review_log.ai.jsonl',OUT/'review_summary.csv',OUT/'suggested_fixes.json']})
print(json.dumps(summary,ensure_ascii=False,indent=2))
