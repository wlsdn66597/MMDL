"""Read-only analysis of the frozen run; never revises baseline grades."""
import json, sys, re, hashlib, statistics, collections
from pathlib import Path

BASE = Path('C:/project/MMDL/results/mmmu_first_baseline_20261009')
OUT = BASE / 'analysis'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, 'C:/project/MMDL/tmp/mmmu-remote-20261009/repo/src')
import parsers

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')
source = BASE / 'baseline_seed3407/results.json'
records = read(source)['records']
attempts = collections.defaultdict(list)
for line in (BASE/'baseline_seed3407/judge_attempts.jsonl').read_text(encoding='utf8').splitlines():
    x=json.loads(line); attempts[x['id']].append(x)

notes = {
 'validation_Accounting_1': ('모델 추론 검토', '고정 제조간접비의 단위당 금액을 생산량 증가 후에도 6으로 유지한다고 설명했다. 응답에 사용한 수치대로라면 6×25,000/30,000+2=7이며 정답 B와 일치한다. 고정 총액과 단위당 비용의 구분 오류가 응답에서 확인된다.'),
 'validation_Architecture_and_Engineering_24': ('채점 재검토', '마지막 Final answer는 정답 C이고 계산값도 보기 C의 0.5010/0.5260과 일치한다. 앞의 Final Answer 제목 아래 서술문 때문에 Parser가 모호함으로 분류했다. Judge는 보기의 두 번째 방법명이 Trapezoidal로 잘못 적힌 점을 이유로 Z를 출력했다. 마지막 답 처리와 보기 표기 문제를 함께 검토해야 한다.'),
 'validation_Electronics_27': ('채점 재검토', '모델 최종 답 240°와 정답 -120°는 360° 주기의 위상으로 동치다. Judge는 숫자가 다르다는 이유로 Z를 출력했다. 각도 동치 규칙 누락 의심 사례이며 풀이 전체의 정확성을 인증하는 것은 아니다.'),
 'validation_Chemistry_4': ('채점 재검토', '현재 정답 처리지만 모델은 cis-1-chloro-2-methylcyclohexane, 기준 정답은 trans-1-Chloro-4-methylcyclohexane이다. 입체배치와 치환 위치가 모두 달라 같은 이름이 아니다. Judge의 A 인정은 과대평가 의심 사례다.'),
 'validation_Chemistry_23': ('채점 재검토', '현재 정답 처리지만 모델 최종 수치는 7.5, 기준 정답은 6.5다. 동일 단위에서 약 15.38% 차이가 있는데 Judge는 A를 출력했다. numeric_tolerance가 정의되지 않은 설정이므로 수치 허용 오차 정책을 명시해야 한다.'),
 'validation_Basic_Medical_Science_10': ('채점 재검토', 'open 문제의 실제 정답 C를 합성 보기 A=C, B=Other Answers로 만든 뒤 grading_gold를 C로 유지한다. Judge가 유효하게 낼 수 있는 A/B와 gold C가 불일치한다. 실제 모델 답 d도 정답 C와 다르지만, 이 문항은 채점 구조 자체도 검토해야 한다. D 출력이 25번 반복됐다.'),
 'validation_Electronics_6': ('채점 재검토', '과거 정답 처리에서 이번 오답으로 바뀐 유일한 문항. 4B 응답은 동일하다. 이번 Judge는 약 10.7(마지막 표기 10)과 정답 71.6의 차이로 Z를 출력했다. 과거 마지막 저장 Judge 출력 A와 차이가 있다.'),
}

rows=[]
for r in records:
    g=r['grading']['final']; m=r['model_response']; pre=parsers.grade_response(r)
    field=parsers.extract_final(m['raw'],r['question_type'],parsers.choice_info(r)[0])
    final_payload=field.get('payload') or (field.get('fields') or [{}])[-1].get('payload','')
    if g['correct']: cat='정답'
    elif m['finish_reason']=='length':cat='응답 잘림'
    elif g['method']=='local_qwen3_8b_judge_parser':
        cat={'no_match':'Judge 불일치(Z)','valid_option':'Judge 선택 답 불일치','invalid_option':'Judge 무효 출력','unparsed':'Judge 무효 출력'}.get(g['reason'],'Judge 기타')
    else:cat='Parser 추출 답 불일치'
    flags=[]
    if r['id'] in notes: flags.append(notes[r['id']][0])
    if r['question_type']=='open' and len(str(r['reference_answer']))==1 and str(r['reference_answer']).isupper():flags.append('open 문자 정답 매핑 검토')
    if g['method']=='local_qwen3_8b_judge_parser' and g['correct'] and r['question_type']=='open': flags.append('Judge 정답 인정 검토')
    if m['finish_reason']!='length' and field['status']=='ambiguous':flags.append('Final answer 형식 모호')
    a=attempts[r['id']]
    rows.append(dict(id=r['id'],subject=r['subject'],type=r['question_type'],correct=g['correct'],category=cat,
        method=g['method'],gold=r['reference_answer'],grading_gold=g['grading_gold'],answer=g['extracted_answer'],
        final_payload=final_payload,finish=m['finish_reason'],tokens=m['output_tokens'],seconds=m['elapsed_seconds'],
        parser_reason=pre['reason'],field_status=field['status'],flags=flags,note=notes.get(r['id'],('',''))[1],
        question=r['question'],options=r['options'],response=m['raw'],image_fields=r['image_fields'],
        judge_attempt_count=len(a),judge_attempts=[{k:x[k] for k in ['attempt','raw','decision','finish_reason','elapsed_seconds']} for x in a]))

wrong=[r for r in rows if not r['correct']]
assert len(rows)==900 and len(wrong)==325
cats=dict(collections.Counter(r['category'] for r in wrong))
subjects=[]
for s in sorted({r['subject'] for r in rows}):
    group=[r for r in rows if r['subject']==s]; bad=[r for r in group if not r['correct']]
    subjects.append(dict(subject=s,total=len(group),correct=sum(r['correct'] for r in group),wrong=len(bad),
        length=sum(r['finish']=='length' for r in group),parser_wrong=sum(r['category']=='Parser 추출 답 불일치' for r in bad),judge_wrong=sum(r['category'].startswith('Judge') for r in bad)))
subjects.sort(key=lambda x:(x['correct'],x['subject']))
length=[r for r in rows if r['finish']=='length']
totaltime=sum(r['seconds'] for r in rows); lengthtime=sum(r['seconds'] for r in length)
stats=dict(total=900,correct=575,wrong=325,accuracy=575/900,categories=cats,subjects=subjects,
    length_seconds=lengthtime,total_generation_seconds=totaltime,length_time_share=lengthtime/totaltime,
    type_stats=[dict(type=t,total=len(a:=[r for r in rows if r['type']==t]),correct=sum(r['correct'] for r in a)) for t in ['multiple-choice','open']],
    judge_routing=dict(collections.Counter(r['parser_reason'] for r in rows if r['judge_attempt_count'])),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
dump(OUT/'analysis_summary.json',stats)
dump(OUT/'review_records.json',rows)
# Static JS data works from file:// without fetch or a running server. Escape HTML delimiters.
(OUT/'review-data.js').write_text('window.MMMU_DATA='+json.dumps(dict(stats=stats,rows=rows),ensure_ascii=False).replace('<','\\u003c')+';',encoding='utf8')

table='\n'.join(f"| {s['subject']} | {s['correct']}/30 | {s['correct']/30:.1%} | {s['wrong']} | {s['length']} | {s['parser_wrong']} | {s['judge_wrong']} |" for s in subjects)
ct='\n'.join(f'| {c} | {n} | {n/325:.1%} |' for c,n in cats.items())
cases='\n\n'.join(f"### {id}\n\n{note}\n\n[문항 열기](review.html#{id})" for id,(_,note) in notes.items())
report=f'''# MMMU baseline 결과와 오답 분석

## 먼저 확인할 결과

seed 3407, 900문항 단일 실행의 확정 점수는 **575/900 = 63.89%**, 오답은 **325개**다. 이 문서는 원점수를 바꾸지 않고 오답의 처리 경로와 재검토할 사례를 분리한다.

- [문항 검토 화면](review.html): 전체 900문항을 검색하고 원문 이미지·모델 전체 응답·Judge 시도별 출력을 확인한다. 기본 화면은 오답 325개다.
- [Excel 검토표](mmmu_error_review.xlsx): 요약, 오답 325개, Judge 54개 검토표. 노란 칸에 원인·메모·검토 상태를 기록한다.
- 원본 결과는 상위 폴더 baseline_seed3407, 실행 근거는 provenance에 그대로 있다.

## 오답을 처리 경로로 분해

| 관측된 오답 유형 | 문항 수 | 오답 325개 중 비중 |
|---|---:|---:|
{ct}

이 분류는 로그에서 확인되는 **채점 경로**다. Parser가 답을 추출했지만 틀린 222건을 전부 Parser 오류로 해석해서는 안 된다. 지식 부족·이미지 판독·계산·추론의 세부 원인은 문항과 이미지를 함께 검토해야 한다. 이번 작업은 전체 900개 구조 분석과 아래 대표 사례의 응답/Judge 근거 검토이며, 325개 모두의 의미적 원인을 확정한 수동 검수는 아니다.

## 우선순위 1: 응답 잘림과 계산량

81문항은 32,768 token 상한에서 잘려 모두 오답이다. 전체 문항의 9.00%, 전체 오답의 24.92%다. 이 81문항의 4B 생성 시간 합계는 **{lengthtime/3600:.2f}시간**, 전체 4B 생성 시간의 **{lengthtime/totaltime:.2%}**다. 나머지 819문항의 정답률은 575/819 = 70.21%이며 이는 잘린 문항을 제외한 진단용 지표로, baseline 정확도 63.89%를 대체하지 않는다.

Energy_and_Power 13건, Materials 12건, Architecture_and_Engineering 11건, Mechanical_Engineering 11건으로 네 과목에 47/81건(58.02%)이 집중됐다. 이 과목들은 긴 계산·반복 풀이가 끝나지 않는지 먼저 확인할 가치가 있다. 단순히 토큰 상한만 늘려도 해결된다고 볼 수 없고, 짧게 종료하는 학습 또는 답 형식 준수 개선을 평가할 때 정확도와 잘림률·생성 시간을 함께 측정해야 한다. 이번 baseline의 출력 상한과 오답 정책은 유지한다.

## 우선순위 2: 취약 과목의 정상 종료 오답

Music은 6/30(20.0%), Diagnostics_and_Laboratory_Medicine은 10/30(33.3%), Architecture_and_Engineering·Mechanical_Engineering은 각각 11/30(36.7%)이다. 특히 Diagnostics_and_Laboratory_Medicine은 잘림 없이 20문항이 틀렸으므로 토큰 상한만으로 설명되지 않는다. Music은 24개 오답 중 잘림이 7개이며 나머지 17개를 악보/음악 기호 판독과 음악 개념으로 나눠 검토한다. 아직 이 세부 원인이 확정된 것은 아니다.

객관식은 543/847(64.11%), open은 32/53(60.38%)이다. 문항 수와 난도가 다른 집단이므로 이 차이만으로 문제 유형에 대한 인과적 결론을 내리지 않는다.

## 우선순위 3: 채점기 자체의 과소·과대평가

Judge 54문항 중 32개는 정답, 22개는 오답이다. 오답만 보면 Judge의 과대평가를 놓칠 수 있어 **정답 처리된 32개도 포함**한 Judge 검토표를 만들었다. Judge 진입 이유는 {json.dumps(stats['judge_routing'],ensure_ascii=False)}다. 이 중 Final answer 모호함으로 들어간 사례와 open 정답 표현 차이를 구분해서 본다.

{cases}

## 과목별 결과

| 과목 | 정답 | 정확도 | 오답 | 잘림 오답 | Parser 추출 답 불일치 | Judge 오답 |
|---|---:|---:|---:|---:|---:|---:|
{table}

## 검토를 진행하는 방법

1. 먼저 채점 재검토 6개 사례와 Judge 전체 54개를 확인한다. 정답 인정의 오류와 오답 처리의 오류를 양쪽으로 본다.
2. 잘림 81개 중 집중된 네 과목부터, 반복되는 풀이·중간 답 번복·Final answer 도달 여부를 기록한다.
3. 정상 종료 오답은 이미지 판독/OCR, 도메인 지식, 계산·추론, 지시·형식, 채점·정답 데이터, 복합/불명으로 수동 분류한다. 근거 없이 자동으로 원인을 확정하지 않는다.
4. HTML에서 검토 상태·원인·메모를 저장하고 JSON으로 내보내면 백업/다른 브라우저로 가져올 수 있다. Excel 메모와 HTML 메모는 서로 자동 동기화되지 않으므로 한쪽을 주 검토표로 정한다.
5. 개선 모델의 같은 900문항 결과는 ID로 조인해 오답→정답과 정답→오답을 함께 계산한다. 정확도 외에 잘림률, 과목별 변화, 평균/중앙 생성 토큰, 전체 시간을 같이 비교한다.

Parser/Judge 규칙을 바꿀 경우 평가 프로토콜의 새 버전으로 분리하고, baseline과 개선 모델의 저장 응답 모두 같은 규칙으로 재채점해야 한다. 기준점 점수를 조용히 덮어쓰면 모델 개선과 채점 변경 효과를 구분할 수 없다. 또한 이 validation 문항을 직접 학습에 사용했다면 같은 900문항 점수는 독립적인 일반화 성능으로 볼 수 없으므로 별도 미사용 평가셋이 필요하다.

## 데이터 보존과 재현

원본 results.json SHA256: `{stats['source_sha256']}`. 분석은 원본을 수정하지 않았다. 모델/데이터 revision·seed·패키지·코드 해시는 상위 manifest/provenance와 기존 실행 보고서에 있다. 이미지는 원격 고정 데이터 캐시에서 내보냈으며 평가 당시 RGB hash와 일치하는지 검사한다. 생성 후 analysis_validation.json에 이미지·문항·집계 검증 결과를 기록한다.
'''
(OUT/'README.md').write_text(report,encoding='utf8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
