"""Validated baseline aggregation; stdlib only, usable on Windows and Linux."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import statistics

SEEDS = [3407, 42, 123, 2026, 9876]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()] if Path(path).exists() else []


def stable(value):
    # Only nondeterministic timestamps / timing are excluded from repeatability.
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in ('elapsed_seconds', 'created_at', 'timestamp')}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def validate(folder, expected_ids, config):
    folder = Path(folder)
    data = read(folder / 'results.json')
    rows = data['records']
    predictions = jsonl(folder / 'predictions.jsonl')
    assert len(rows) == len(predictions) == 900
    by_id = {r['id']: r for r in rows}
    pred = {r['id']: r for r in predictions}
    assert set(by_id) == set(pred) == set(expected_ids) and len(by_id) == 900
    assert Counter(r['subject'] for r in rows) == {s: 30 for s in config['dataset']['configs']}
    manifest = read(folder / 'manifest.json')
    assert manifest == data['manifest'] and manifest['configuration'] == config
    assert manifest['mode'] == 'inference'
    for row in rows:
        grade = row['grading']['final']
        assert grade['status'] == 'graded' and grade['hit'] in (0, 1)
        assert row['model_response']['finish_reason'] in ('length', 'stop')
        assert all(row[k] == value for k, value in pred[row['id']].items())
        if row['model_response']['finish_reason'] == 'length':
            assert grade['hit'] == 0 and row['model_response']['output_tokens'] == config['max_new_tokens']
    summary = read(folder / 'summary.json')
    assert summary == data['summary']
    assert summary['correct'] == sum(r['grading']['final']['hit'] for r in rows)
    assert summary['total'] == summary['graded'] == 900
    assert summary['method_counts'] == dict(Counter(r['grading']['final']['method'] for r in rows))
    assert summary['length_limit_incorrect'] == sum(r['model_response']['finish_reason'] == 'length' for r in rows)
    assert abs(summary['accuracy_percent'] - 100 * summary['correct'] / 900) < 1e-10
    for subject, values in summary['by_subject'].items():
        subset = [r for r in rows if r['subject'] == subject]
        assert values['total'] == len(subset) == 30
        assert values['correct'] == sum(r['grading']['final']['hit'] for r in subset)
    assert set(summary['by_subject']) == set(config['dataset']['configs'])
    return data


def compare_repeat(first, second, output):
    a = read(Path(first) / 'results.json')
    b = read(Path(second) / 'results.json')
    old = {r['id']: r for r in a['records']}
    new = {r['id']: r for r in b['records']}
    assert set(old) == set(new) and len(old) == 900
    differences = []
    for rid in old:
        keys = sorted(set(old[rid]) | set(new[rid]))
        changed = [k for k in keys if stable(old[rid].get(k)) != stable(new[rid].get(k))]
        if changed:
            differences.append({'id': rid, 'changed_fields': changed})
    attempts_a = jsonl(Path(first) / 'judge_attempts.jsonl')
    attempts_b = jsonl(Path(second) / 'judge_attempts.jsonl')
    judge_equal = stable(attempts_a) == stable(attempts_b)
    identity_equal = a['manifest']['model_identity'] == b['manifest']['model_identity']
    config_equal = a['manifest']['configuration'] == b['manifest']['configuration']
    code_equal = a['manifest']['code_sha256'] == b['manifest']['code_sha256']
    result = {'passed': not differences and judge_equal and identity_equal and config_equal and code_equal,
              'records_compared': 900, 'record_differences': differences,
              'judge_attempts_equal': judge_equal, 'judge_attempt_counts': [len(attempts_a), len(attempts_b)],
              'model_identity_equal': identity_equal, 'configuration_equal': config_equal,
              'code_hashes_equal': code_equal,
              'excluded_fields': ['elapsed_seconds', 'created_at', 'timestamp'],
              'diagnostic_only': True, 'first': str(first), 'second': str(second)}
    write(output, result)
    return result


def aggregate(root, old3407, old42):
    root = Path(root)
    expected = {r['id'] for r in read(root / 'protocol/evaluation_items.json')}
    paths = {3407: Path(old3407), 42: Path(old42)}
    paths.update({s: root / f'baseline_seed{s}' for s in SEEDS[2:]})
    datasets, metrics = {}, []
    for seed, path in paths.items():
        if not (path / 'summary.json').exists():
            continue
        config = read(root / f'protocol/config.seed{seed}.json')
        data = validate(path, expected, config)
        datasets[seed] = {r['id']: r for r in data['records']}
        if seed != 3407:
            for rid, row in datasets[seed].items():
                for key in ('question', 'options', 'reference_answer', 'input_prompt', 'image_fields', 'rendered_chat'):
                    assert row[key] == datasets[3407][rid][key], (seed, rid, key)
        lengths = [r['model_response']['output_tokens'] for r in data['records']]
        metrics.append({'seed': seed, 'correct': data['summary']['correct'], 'total': 900,
                        'accuracy_percent': 100 * data['summary']['correct'] / 900,
                        'length_count': data['summary']['length_limit_incorrect'],
                        'judge_count': data['summary']['judge_count'],
                        'mean_output_tokens': statistics.mean(lengths),
                        'median_output_tokens': statistics.median(lengths),
                        'p95_output_tokens': sorted(lengths)[854],
                        'generation_hours': sum(r['model_response']['elapsed_seconds'] for r in data['records']) / 3600,
                        'source': str(path), 'results_sha256': hashlib.sha256((path / 'results.json').read_bytes()).hexdigest()})
    available = list(datasets)
    items, subjects, pairs = [], [], []
    for rid in sorted(expected):
        hits = {str(s): datasets[s][rid]['grading']['final']['hit'] for s in available}
        truncated = [s for s in available if datasets[s][rid]['model_response']['finish_reason'] == 'length']
        items.append({'id': rid, 'subject': datasets[3407][rid]['subject'], 'hits': hits,
                      'correct_seed_count': sum(hits.values()), 'available_seeds': len(available), 'length_seeds': truncated})
    for subject in sorted({r['subject'] for r in items}):
        vals = {str(s): 100 * sum(datasets[s][r['id']]['grading']['final']['hit'] for r in items if r['subject'] == subject) / 30 for s in available}
        subjects.append({'subject': subject, 'accuracy_by_seed': vals,
                         'mean_percent': statistics.mean(vals.values()),
                         'sample_sd_pp': statistics.stdev(vals.values()) if len(vals) > 1 else None})
    for i, a in enumerate(available):
        for b in available[i+1:]:
            counts = Counter(f"{datasets[a][rid]['grading']['final']['hit']}_to_{datasets[b][rid]['grading']['final']['hit']}" for rid in expected)
            pairs.append({'seed_a': a, 'seed_b': b, 'transitions': dict(counts),
                          'delta_pp': 100 * (counts['0_to_1'] - counts['1_to_0']) / 900})
    values = [r['accuracy_percent'] for r in metrics]
    repeat_path = root / 'repeatability.json'
    report = {'model': 'Qwen/Qwen3-VL-4B-Instruct', 'protocol': 'mmmu-local-v1',
              'updated_at_utc': datetime.now(timezone.utc).isoformat(),
              'planned_seeds': SEEDS, 'available_seeds': available, 'all_five_complete': available == SEEDS,
              'repeatability': read(repeat_path) if repeat_path.exists() else {'passed': None, 'status': 'pending'},
              'runs': metrics, 'mean_accuracy_percent': statistics.mean(values),
              'sample_sd_pp': statistics.stdev(values) if len(values) > 1 else None,
              'always_wrong_count': sum(r['correct_seed_count'] == 0 for r in items),
              'always_correct_count': sum(r['correct_seed_count'] == len(available) for r in items),
              'unstable_count': sum(0 < r['correct_seed_count'] < len(available) for r in items),
              'subjects': subjects, 'pairwise_seed_transitions': pairs,
              'limitations': ['One checkpoint; inference-seed variation only.',
                             'Repeated fixed 900 items are not 4500 independent questions.',
                             'No candidate checkpoint evaluated. Repeatability run excluded from mean.',
                             'Seed42 code adds token recording only; provenance records the source exception.']}
    write(root / 'aggregate.json', report)
    write(root / 'item_stability.json', items)
    def table(headers, rows):
        return '<table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'
    state = read(root / 'status.json') if (root / 'status.json').exists() else {'state': 'prepared'}
    body = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>MMMU 5-seed baseline</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#182433}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ccd5df;text-align:left}th{background:#edf2f7}code{background:#edf2f7}a{color:#1764a4}.big{font-size:26px;font-weight:700}</style><h1>MMMU baseline · 5개 추론 seed</h1>'
    body += f'<p>완료 {len(available)}/5 · 상태 {html.escape(str(state.get("state")))} · {html.escape(str(state.get("phase", "")))}</p>'
    body += f'<p class="big">{"최종" if available == SEEDS else "중간"} 평균 {statistics.mean(values):.2f}% ± {report["sample_sd_pp"]:.2f}%p (표본 표준편차)</p>'
    body += '<p>같은 Qwen3-VL-4B-Instruct 체크포인트와 고정 900문항의 추론 seed 변동입니다. 개선 모델과의 성능 비교는 포함하지 않습니다. 재현성 검사 실행은 평균에서 제외합니다.</p>'
    body += table(['Seed', '정답 / 900', '정확도', '잘림', 'Judge 문항', '평균 출력 토큰', '생성 시간 합계'], [[r['seed'], r['correct'], f"{r['accuracy_percent']:.2f}%", r['length_count'], r['judge_count'], f"{r['mean_output_tokens']:.1f}", f"{r['generation_hours']:.2f} h"] for r in metrics] + [[s, '대기', '—', '—', '—', '—', '—'] for s in SEEDS if s not in available])
    body += '<h2>문항 안정성</h2>' + table(['현재 완료 seed 기준', '문항 수'], [['모두 정답', report['always_correct_count']], ['모두 오답', report['always_wrong_count']], ['seed에 따라 정오답 변동', report['unstable_count']]])
    body += '<h2>재현성 검사</h2><pre>' + html.escape(json.dumps(report['repeatability'], ensure_ascii=False, indent=2)[:10000]) + '</pre>'
    body += '<h2>실행 방식</h2><p>기존 3407·42 결과 보존 → 3407 새 프로세스/새 폴더 재실행 → 원문·토큰 수·종료 사유·최종 채점·Judge 시도 일치 확인 → 123·2026·9876 순차 실행. 불일치나 실패 시 중단하고 근거를 보존합니다. 기존 실측을 적용한 추가 4회 예상은 약 55~57시간이며 완료 시각 보장은 아닙니다.</p>'
    body += '<h2>과목별 평균</h2>' + table(['과목', '평균 정확도', '표본 표준편차 %p'], [[r['subject'], f"{r['mean_percent']:.2f}%", f"{r['sample_sd_pp']:.2f}"] for r in subjects])
    body += '<p><a href="aggregate.json">전체 집계 JSON</a> · <a href="item_stability.json">900문항 seed별 정오답 JSON</a> · <a href="status.json">실행 상태</a></p></html>'
    (root / 'report.html').write_text(body, encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--baseline3407', required=True)
    parser.add_argument('--baseline42', required=True)
    args = parser.parse_args()
    result = aggregate(args.root, args.baseline3407, args.baseline42)
    print(json.dumps({k: result[k] for k in ('available_seeds', 'mean_accuracy_percent', 'sample_sd_pp')}))
