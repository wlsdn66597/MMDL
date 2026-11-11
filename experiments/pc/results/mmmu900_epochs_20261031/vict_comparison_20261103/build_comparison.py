"""Join frozen baseline error cohorts to existing grades; never regrade responses."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
TAXONOMY = Path('C:/project/mmdl_test/baseline_results.json')
INVENTORY = BASE / 'evidence/pilot/all900_response_inventory.json'
AUDIT = BASE / 'content_audit_20261103/all_reviews.json'
REGRADE = BASE / 'regrade_final_answers_20261103/regrade_records.json'
MODELS = ['baseline', 'epoch_2', 'epoch_3', 'epoch_4']
TYPES = ['V', 'I', 'C', 'T', 'N', 'D', 'U', 'P', 'R']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def pct(n, d):
    return round(100 * n / d, 4) if d else None


source = read(TAXONOMY)
old = {r['id']: r for r in source['records']}
inventory = read(INVENTORY)
audit = {(a['id'], a['model']): a for a in read(AUDIT)['annotations']}
regrade = {(a['id'], a['model']): a for a in read(REGRADE)}
assert len(old) == len(inventory) == len({r['id'] for r in inventory}) == 900
assert set(old) == {r['id'] for r in inventory}
assert sum(r['grading']['final']['hit'] for r in old.values()) == 576

rows, grade_differences = [], []
for item in inventory:
    id_ = item['id']
    prior = old[id_]
    baseline = item['models']['baseline']['record']
    assert prior['model_response']['raw'] == baseline['model_response']['raw'], id_
    for key in ['question', 'options', 'reference_answer', 'input_prompt', 'image_fields']:
        assert prior[key] == baseline[key], (id_, key)
    prior_hit = int(prior['grading']['final']['hit'])
    baseline_hit = int(baseline['grading']['final']['hit'])
    label = (prior.get('error_analysis') or {}).get('primary_type_code')
    if prior_hit != baseline_hit:
        grade_differences.append({'id': id_, 'taxonomy_source_hit': prior_hit, 'baseline_hit': baseline_hit})
    if label:
        assert not prior_hit and not baseline_hit, id_
    row = dict(id=id_, subject=item['subject'], baseline_type=label,
               baseline_group=label or ('baseline_correct' if baseline_hit else 'unclassified_extra'),
               taxonomy_source_hit=prior_hit, reference_answer=baseline['reference_answer'],
               question_type=baseline['question_type'], question=baseline['question'], options=baseline['options'],
               original_error_analysis=prior.get('error_analysis'), models={})
    for model in MODELS:
        record = item['models'][model]['record']
        for key in ['question', 'options', 'reference_answer', 'input_prompt', 'image_fields']:
            assert record[key] == baseline[key], (id_, model, key)
        response, grade = record['model_response'], record['grading']['final']
        raw = response['raw']
        hit = int(grade['hit'])
        assert hit in [0, 1]
        if response['finish_reason'] == 'length':
            assert hit == 0
        review = audit.get((id_, model))
        if review:
            assert review['raw_sha256'] == digest(raw.encode('utf-8'))
            assert review['original_hit'] == hit
        planned = regrade[(id_, model)]
        assert planned['response_sha256'] == digest(raw.encode('utf-8'))
        assert planned['original_hit'] == hit
        fields = re.findall(r'(?im)^\s*(?:\*\*)?Final\s+answer\s*:\s*(.*?)\s*$', raw)
        row['models'][model] = dict(
            hit=hit, finish_reason=response['finish_reason'], output_tokens=response['output_tokens'],
            grading_method=grade['method'], grading_extracted_answer=grade.get('extracted_answer'),
            grading_gold=grade.get('grading_gold'), response_sha256=digest(raw.encode('utf-8')),
            final_line_display=fields[-1] if fields else None,
            response_tail=raw[-1200:],
            transition='kept_correct' if baseline_hit and hit else 'regressed' if baseline_hit else 'recovered' if hit else 'still_wrong',
            regrade_status=planned['proposed']['status'],
            prior_content_review=None if review is None else {k: review[k] for k in ['status', 'scores', 'coverage', 'rationale']},
        )
    rows.append(row)

assert grade_differences == [{'id': 'validation_Electronics_6', 'taxonomy_source_hit': 1, 'baseline_hit': 0}]
counts = dict(Counter(r['baseline_type'] for r in rows if r['baseline_type']))
assert counts == {'C': 76, 'T': 81, 'V': 29, 'D': 16, 'R': 21, 'N': 5, 'I': 46, 'P': 2, 'U': 48}
assert sum(counts.values()) == 324


def cohort_result(group, model):
    recovered = [r for r in group if r['models'][model]['hit']]
    remaining = [r for r in group if not r['models'][model]['hit']]
    length = [r for r in group if r['models'][model]['finish_reason'] == 'length']
    content = Counter(r['models'][model]['prior_content_review']['status'] for r in recovered if r['models'][model]['prior_content_review'])
    return dict(total=len(group), correct=len(recovered), wrong=len(remaining), accuracy_pct=pct(len(recovered), len(group)),
                error_rate_pct=pct(len(remaining), len(group)), truncated=len(length),
                recovered_ids=[r['id'] for r in recovered], remaining_wrong_ids=[r['id'] for r in remaining],
                recovered_content_review_counts=dict(content))


cohorts = {}
for type_ in TYPES + ['VICT', 'other_old_errors', 'unclassified_extra']:
    if type_ == 'VICT':
        group = [r for r in rows if r['baseline_type'] in ['V', 'I', 'C', 'T']]
    elif type_ == 'other_old_errors':
        group = [r for r in rows if r['baseline_type'] in ['N', 'D', 'U', 'P', 'R']]
    elif type_ == 'unclassified_extra':
        group = [r for r in rows if r['baseline_group'] == type_]
    else:
        group = [r for r in rows if r['baseline_type'] == type_]
    cohorts[type_] = {m: cohort_result(group, m) for m in MODELS}

overall = {}
truncation = {}
baseline_correct = sum(r['models']['baseline']['hit'] for r in rows)
for model in MODELS:
    transitions = Counter(r['models'][model]['transition'] for r in rows)
    correct = sum(r['models'][model]['hit'] for r in rows)
    recovered = transitions['recovered']
    regressed = transitions['regressed']
    assert baseline_correct + recovered - regressed == correct
    assert recovered == sum(cohorts[t][model]['correct'] for t in TYPES + ['unclassified_extra'])
    overall[model] = dict(total=900, correct=correct, wrong=900-correct,
                          accuracy_pct=pct(correct,900), error_rate_pct=pct(900-correct,900),
                          recovered=recovered, regressed=regressed,
                          kept_correct=transitions['kept_correct'], still_wrong=transitions['still_wrong'],
                          net_correct_change=correct-baseline_correct,
                          accuracy_change_pp=pct(correct-baseline_correct,900),
                          recovered_ids=[r['id'] for r in rows if r['models'][model]['transition']=='recovered'],
                          regressed_ids=[r['id'] for r in rows if r['models'][model]['transition']=='regressed'])
    was_t = [r for r in rows if r['models']['baseline']['finish_reason']=='length']
    now_t = [r for r in rows if r['models'][model]['finish_reason']=='length']
    remained = [r for r in was_t if r['models'][model]['finish_reason']=='length']
    resolved = [r for r in was_t if r['models'][model]['finish_reason']!='length']
    new = [r for r in now_t if r['models']['baseline']['finish_reason']!='length']
    solved = sum(r['models'][model]['hit'] for r in was_t)
    truncation[model] = dict(total=len(now_t), rate_pct=pct(len(now_t),900),
        old_t_total=len(was_t), old_t_still_truncated=len(remained), old_t_resolved=len(resolved),
        old_t_correct=solved, old_t_finished_but_wrong=len(resolved)-solved,
        new_truncated=len(new), new_truncated_from_baseline_correct=sum(r['models']['baseline']['hit'] for r in new),
        old_t_still_truncated_ids=[r['id'] for r in remained], new_truncated_ids=[r['id'] for r in new])
    assert len(now_t)==len(remained)+len(new)
    assert 81==len(remained)+solved+len(resolved)-solved

limitations = [
    '현재 분석은 V/C/T 개별 학습 실험이 아니라 기존 ERA 2·3·4epoch 체크포인트 비교다. seed 3407 한 번의 결과다.',
    '기존 파서·Qwen3-8B가 저장한 정오를 네 모델에 그대로 사용했다. 서버가 꺼져 있어 수정 규칙의 재채점은 완료되지 않았다.',
    'V/I/C/T는 baseline 오답의 고정 문항 집단이다. 이후에도 틀린 문항의 현재 오류 원인이 같은 유형이라고 재분류한 것은 아니다.',
    '기존 정답에서 새로 생긴 오답의 V/I/C 유형은 미분류다. 따라서 전체 V/I/C 오류 발생률의 증감은 이 자료로 확정할 수 없다.',
    'T만 모든 응답의 finish_reason=length로 직접 재집계했다. 잘림 해소와 정답 회복은 다르다.',
    '과거 원본 576/900과 현재 baseline 575/900의 차이는 Electronics_6의 Judge 판정이다. 900개 생성 응답은 모두 동일하다.',
    '정답 처리 수는 풀이 내용의 정확성을 보증하지 않는다. 기존 E/R/A 검수 결과는 별도 열이며 채점에 사용하지 않았다.',
    '원래 V/I/C 분류도 관찰된 응답 패턴이다. 해당 신경망 모듈이 오류 원인이라는 인과 진단이 아니다.',
]
summary = dict(scope='frozen_grades_fixed_baseline_error_cohorts', seed=3407,
               taxonomy_counts=counts, baseline_grade_reconciliation=grade_differences,
               overall=overall, cohorts=cohorts, truncation=truncation, limitations=limitations)
save('summary.json', summary)
save('question_comparison.json', rows)
save('taxonomy_snapshot.json', dict(source_path=str(TAXONOMY), source_sha256=digest(TAXONOMY.read_bytes()),
    definitions=source['error_type_definitions'], method=source['error_analysis_method'],
    records=[dict(id=r['id'], taxonomy_source_hit=r['taxonomy_source_hit'], baseline_type=r['baseline_type'],
                  error_analysis=r['original_error_analysis']) for r in rows]))
save('validation.json', dict(passed=True, unique_question_ids=900, model_records=3600,
    original_baseline_response_matches=900, question_options_reference_prompt_image_field_matches=True,
    current_baseline_correct=575, source_baseline_correct=576, taxonomy_ids=324, vict_ids=232,
    grade_difference_ids=[r['id'] for r in grade_differences], transition_and_truncation_identities_passed=True,
    content_review_hashes_verified=2700, original_files_modified=False,
    sources=[dict(path=str(p),sha256=digest(p.read_bytes())) for p in [TAXONOMY, INVENTORY, AUDIT, REGRADE]]))

lines = ['MMMU 900: 기존 V/I/C/T 오답 집단의 ERA 결과 비교', '',
         '기존 오답 일부는 회복했지만, 새로 틀린 문항이 더 많아 전체 오답은 늘었다.',
         '비교 기준: baseline과 ERA 2·3·4epoch, seed 3407, 기존 저장 채점 유지.', '',
         '전체 900문항 (회복=baseline 오답→정답; 퇴행=baseline 정답→오답)']
for model in MODELS:
    s=overall[model]
    lines.append(f"{model}: 정답 {s['correct']}/900 ({s['accuracy_pct']:.2f}%), 오답 {s['wrong']}/900 ({s['error_rate_pct']:.2f}%), 회복 {s['recovered']}, 퇴행 {s['regressed']}, 순정답 변화 {s['net_correct_change']:+d}")
lines += ['', '기존 유형별 동일 문항의 회복 수 / 잔존 오답 수 (baseline은 전부 오답)']
for type_ in ['V','I','C','T','VICT','other_old_errors','unclassified_extra']:
    values=[]
    for m in MODELS[1:]:
        c=cohorts[type_][m]
        values.append(f"{m} 정답 {c['correct']}/{c['total']} ({c['accuracy_pct']:.2f}%), 오답 {c['wrong']} ({c['error_rate_pct']:.2f}%)")
    lines.append(type_+': '+' | '.join(values))
lines += ['', 'T 변화: 기존 81개를 정상 종료 정답 / 정상 종료 오답 / 계속 잘림으로 분해']
for m in MODELS[1:]:
    t=truncation[m]
    lines.append(f"{m}: {t['old_t_correct']} / {t['old_t_finished_but_wrong']} / {t['old_t_still_truncated']}; 새 잘림 {t['new_truncated']}, 전체 잘림 {t['total']}/900 ({t['rate_pct']:.2f}%)")
lines += ['', '원문 대조 사례',
    'Math_4: baseline과 세 epoch 모두 C로 오답. 기존 사례의 정답 회복은 없다.',
    'Chemistry_5: 세 epoch 모두 정상 종료했지만 모두 오답. 잘림 해소가 정답 회복으로 이어지지 않은 사례다.',
    'Electronics_9: 세 epoch 모두 잘림은 해소. 2epoch만 정답이며 3·4epoch는 오답이다.',
    'Manage_23: 저장 채점은 세 epoch 모두 정답이나 실제 Final answer는 1441 / 1464 / 1883, 참조는 1464다. 2·4epoch의 정답 처리를 내용 개선으로 해석하면 안 된다. 재채점 대기이며 여기서 수동 보정하지 않았다.',
    'Finance_8: 저장 채점은 4epoch에서만 정답. 그 설명의 정확성은 별도 E/R/A 검수 대상이다.',
    '', '해석 범위']
lines.extend('- '+s for s in limitations)
lines += ['', '파일', 'comparison.html: 유형·모델·정오 변화·잘림 상태로 900문항을 검색하고 최종 응답 및 기존 E/R/A 검수 확인.',
          'question_comparison.json: 전체 문항별 원래 유형·4모델 정오·토큰·잘림·응답 끝부분·채점/검수 상태.',
          'summary.json: 유형별 회복/잔존/퇴행 ID 목록과 집계.', 'validation.json: 입력/응답 정합성 및 집계 항등식 검사.',
          'taxonomy_snapshot.json: 원본 분류와 근거 보존.']
(OUT/'comparison_note.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

template=(OUT/'comparison_template.html').read_text(encoding='utf-8')
data=json.dumps(dict(summary=summary, rows=rows),ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
(OUT/'comparison.html').write_text(template.replace('/*__DATA__*/', 'const DATA='+data+';'),encoding='utf-8')
print(json.dumps({'verified':True,'questions':len(rows),'overall_correct':{m:overall[m]['correct'] for m in MODELS}},ensure_ascii=True))
