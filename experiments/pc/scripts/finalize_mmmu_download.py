"""Audit downloaded MMMU records and add a Korean interpretation to the HTML report."""
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import statistics

from mmmu_multiseed_report import aggregate, compare_repeat, validate, jsonl

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK / 'results/mmmu_multiseed_20261011'
OLD = {3407: WORK / 'results/mmmu_first_baseline_20261009/baseline_seed3407',
       42: WORK / 'results/mmmu_visual_tokens_seed42_20261010/baseline_seed42'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    def cells(tag, row):
        return '<tr>' + ''.join(f'<{tag}>{html.escape(str(x))}</{tag}>' for x in row) + '</tr>'
    return '<table><thead>' + cells('th', headers) + '</thead><tbody>' + ''.join(cells('td', row) for row in rows) + '</tbody></table>'


def main():
    source = read(ROOT / 'aggregate.remote.json')
    hashes = read(ROOT / 'provenance/reused_runs_remote_sha256.json')
    reused_checks = []
    for seed, folder in OLD.items():
        for name, expected in hashes[str(seed)].items():
            actual = sha(folder / name)
            assert actual == expected, (seed, name, actual, expected)
            reused_checks.append({'seed': seed, 'file': name, 'sha256': actual})
    archive = ROOT.parent / 'mmmu_multiseed_20261011.tar.gz'
    expected_archive = (ROOT.parent / (archive.name + '.sha256')).read_text().split()[0]
    assert sha(archive) == expected_archive
    expected_ids = {x['id'] for x in read(ROOT / 'protocol/evaluation_items.json')}
    repeat = compare_repeat(OLD[3407], ROOT / 'baseline_seed3407_repeat', ROOT / 'repeatability.local.json')
    assert repeat['passed'] and source['repeatability']['passed']
    validate(ROOT / 'baseline_seed3407_repeat', expected_ids, read(ROOT / 'protocol/config.seed3407.json'))
    report = aggregate(ROOT, OLD[3407], OLD[42])
    assert report['all_five_complete']
    datasets = {}
    checks = []
    first_manifest = read(OLD[3407] / 'manifest.json')
    base_config = copy.deepcopy(first_manifest['configuration'])
    base_config['sampling'].pop('seed')
    preflight = read(ROOT / 'provenance/preflight.json')
    assert preflight['passed'] and preflight['seed42_only_token_capture_difference']
    for run, remote_run in zip(report['runs'], source['runs']):
        assert run['seed'] == remote_run['seed']
        assert run['results_sha256'] == remote_run['results_sha256']
        folder = Path(run['source'])
        data = read(folder / 'results.json')
        manifest = data['manifest']
        identity_check = 'manifest config and weight hashes equal'
        if run['seed'] == 42:
            # This older manifest records a pinned repo/revision, not weight hashes.
            assert manifest['model_identity'] == {k: base_config['model'][k] for k in ['repo', 'revision']}
            assert preflight['target_and_judge_weight_and_auxiliary_hashes_match']
            identity_check = 'pinned repo/revision equal; weight hashes verified by campaign preflight'
        else:
            assert manifest['model_identity'] == first_manifest['model_identity']
        assert manifest['environment']['versions'] == first_manifest['environment']['versions']
        config = copy.deepcopy(manifest['configuration'])
        assert config['sampling'].pop('seed') == run['seed']
        assert config == base_config
        differing_code = [k for k in first_manifest['code_sha256'] if manifest['code_sha256'][k] != first_manifest['code_sha256'][k]]
        assert not differing_code or (run['seed'] == 42 and differing_code == ['src/evaluate.py']), differing_code
        attempts = jsonl(folder / 'judge_attempts.jsonl')
        datasets[run['seed']] = {x['id']: x for x in data['records']}
        checks.append({'seed': run['seed'], 'records': len(data['records']), 'judge_attempt_records': len(attempts),
                       'results_sha256': run['results_sha256'], 'code_difference_from_3407': differing_code,
                       'model_identity_check': identity_check,
                       'additional_code_files': sorted(set(manifest['code_sha256']) - set(first_manifest['code_sha256'])),
                       'individual_verification_json_present': (folder / 'verification.json').exists()})
    for key in ['mean_accuracy_percent', 'sample_sd_pp', 'always_correct_count', 'always_wrong_count', 'unstable_count']:
        assert report[key] == source[key], key
    items = read(ROOT / 'item_stability.json')
    examples = []
    # Deterministic illustrative selection, not evidence that these are the most representative errors.
    for hits in [2, 3, 0]:
        item = next(x for x in items if x['correct_seed_count'] == hits and not x['length_seeds'])
        first = datasets[3407][item['id']]
        examples.append({**item, 'question': first['question'], 'options': first['options'],
                         'reference_answer': first['reference_answer'],
                         'selection_rule': 'First sorted ID with this hit count and no output truncation',
                         'by_seed': {str(s): {'answer': rows[item['id']]['grading']['final']['extracted_answer'],
                                             'hit': rows[item['id']]['grading']['final']['hit'],
                                             'method': rows[item['id']]['grading']['final']['method'],
                                             'response_tail': rows[item['id']]['model_response']['raw'][-600:]}
                                     for s, rows in datasets.items()}})
    write(ROOT / 'illustrative_items.json', examples)
    events = jsonl(ROOT / 'control_events.jsonl')
    status = read(ROOT / 'status.json')
    assert status['state'] == 'complete' and status['error'] is None
    end = datetime.fromisoformat(status['updated_at_utc'])
    start = datetime.fromisoformat(status['started_at_utc'])
    scores = [x['accuracy_percent'] for x in report['runs']]
    audit = {'passed': True, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
             'archive_sha256': expected_archive, 'reused_file_checks': reused_checks, 'runs': checks,
             'repeatability_recomputed_passed': repeat['passed'], 'remote_local_aggregates_equal': True,
             'preflight_passed': preflight['passed'], 'control_events': events,
             'validation_scope': '900 unique items/run, 30 subjects x30, predictions/results/summary/config consistency, length forced wrong, model/package identity, documented seed42 code exception, source hashes; no new inference or grading.'}
    write(ROOT / 'final_verification.json', audit)
    report['min_accuracy_percent'] = min(scores)
    report['max_accuracy_percent'] = max(scores)
    report['sample_sd_ddof'] = 1
    report['campaign_completed_at_kst'] = end.astimezone(timezone(timedelta(hours=9))).isoformat()
    report['campaign_wall_hours'] = (end-start).total_seconds()/3600
    report['control_event_count'] = len(events)
    report['final_verification'] = 'final_verification.json'
    write(ROOT / 'aggregate.json', report)
    total_length = sum(x['length_count'] for x in report['runs'])
    total_judge = sum(x['judge_count'] for x in report['runs'])
    extra = '<h2>최종 해석과 검증</h2>'
    extra += f'<p>2026년 10월 13일 21:53 KST에 캠페인이 완료되었습니다. 5개 seed 평균은 {statistics.mean(scores):.4f}%, 표본 표준편차는 {statistics.stdev(scores):.4f}%p(ddof=1)입니다. 최소 {min(scores):.2f}%부터 최대 {max(scores):.2f}%까지 {max(scores)-min(scores):.2f}%p 차이가 있습니다. 표준편차는 신뢰구간이나 표준오차가 아닙니다.</p>'
    extra += f'<p>900문항 중 442개는 5회 모두 정답, 203개는 모두 오답이며 255개(28.33%)는 seed에 따라 정오답이 바뀌었습니다. 5회 합계 {total_length}/4500개 응답({total_length/45:.2f}%)은 출력 한도 32,768토큰에 도달해 규약에 따라 오답 처리했습니다. Judge 채점 경로는 {total_judge}/4500개 응답({total_judge/45:.2f}%)에 사용됐습니다. 이 4,500회 응답은 동일 900문항의 반복입니다.</p>'
    extra += '<p>환경은 RTX 4090, vLLM 0.11.0, PyTorch 2.8.0, BF16이며 VLLM_ENABLE_V1_MULTIPROCESSING=0, max_num_seqs=1을 사용했습니다. VLLM_BATCH_INVARIANT는 적용하지 않았습니다. seed 3407의 새 실행과 기존 실행은 타임스탬프·소요 시간을 제외한 900문항 기록과 Judge 시도 78건이 일치합니다. 이는 검사한 고정 환경의 반복 재현성이고 배치 변경 불변성 검사는 아닙니다.</p>'
    extra += '<p>원격 압축 파일 SHA256, 기존 3407·42의 모든 파일 SHA256, 새 3개 seed의 results SHA256을 대조했습니다. 각 run의 고유 ID 900개, 과목별 30개, 응답·최종 채점·summary 일치 및 잘림 강제 오답 규칙을 로컬에서 다시 검증했습니다. 개별 run의 verification.json 유무와 별개로 이번 다운로드 검증 결과는 final_verification.json에 기록했습니다. seed 42의 응답 토큰 기록용 evaluate.py 차이는 기존 preflight의 허용 예외로 명시했습니다.</p>'
    extra += f'<p>이번 캠페인(3407 재검사 + 새 seed 3회)의 실제 벽시계 시간은 {report["campaign_wall_hours"]:.2f}시간입니다. 위 표의 생성 시간은 문항별 elapsed_seconds 합계로, 모델 로딩·Judge·압축 시간을 포함하지 않습니다. 실평가 control_events.jsonl은 없고 동일 드라이버가 완료했으며 중단·재개 기록은 없습니다. resume_tests.log의 paused 출력은 별도 테스트 기록입니다.</p>'
    extra += '<h2>문항 사례</h2><p>잘림 없는 문항 중 정답 횟수가 2회·3회·0회인 첫 ID를 각각 고른 예시입니다. 전체 오류를 대표한다는 뜻은 아닙니다. 아래 표는 저장된 답 추출·채점 기록의 비교이며 이미지·해설의 정답성을 수동으로 인증한 결과가 아닙니다.</p>'
    for example in examples:
        extra += f'<h3>{html.escape(example["id"])}</h3><p>기준 답: {html.escape(str(example["reference_answer"]))} · 정답 {example["correct_seed_count"]}/5회</p><p>{html.escape(example["question"])}</p>'
        extra += table(['Seed', '추출 답', '정답 여부', '채점 경로'], [[s, v['answer'], v['hit'], v['method']] for s,v in example['by_seed'].items()])
    extra += '<p>같은 체크포인트의 추론 seed 변동만 측정했습니다. 학습 seed 안정성이나 개선 모델의 성능 향상을 입증하지 않으며, Judge의 의미적 정확성을 전 문항 수동 확인한 것은 아닙니다. 과거 64.00%와 진단용 반복 실행은 평균에서 제외했습니다.</p>'
    extra += '<p><a href="final_verification.json">최종 검증 기록</a> · <a href="illustrative_items.json">사례 원문·응답 끝부분</a> · <a href="provenance/preflight.json">실행 전 환경·입력 검증</a> · <a href="../mmmu_multiseed_20261011.tar.gz">서버 원본 압축 파일</a></p>'
    path = ROOT / 'report.html'
    path.write_text(path.read_text(encoding='utf-8').replace('</html>', extra + '</html>'), encoding='utf-8')
    print(json.dumps({'passed': True, 'mean': report['mean_accuracy_percent'], 'sample_sd': report['sample_sd_pp'], 'reused_files_checked': len(reused_checks), 'example_ids': [x['id'] for x in examples]}))


if __name__ == '__main__':
    main()
