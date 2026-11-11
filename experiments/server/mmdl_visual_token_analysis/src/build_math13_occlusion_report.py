"""Render measured Math_13 intervention outcomes; never interpolate untested areas."""
import argparse
import copy
from collections import Counter
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import math13_occlusion

NAMES = {'original': '원본', 'attention_high': '높은 attention 영역',
         'geometry_D': '도형 정보: D 꼭짓점', 'random_region': '무작위 영역'}
COLORS = {'original': '#344054', 'attention_high': '#9333a5',
          'geometry_D': '#2563b5', 'random_region': '#bd670b'}
BG = '#f3f5f8'
INK = '#172b4d'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size, bold=False):
    candidates = [Path('C:/Windows/Fonts/' + ('malgunbd.ttf' if bold else 'malgun.ttf')),
                  Path('/usr/share/fonts/truetype/nanum/' + ('NanumGothicBold.ttf' if bold else 'NanumGothic.ttf'))]
    return ImageFont.truetype(str(next(p for p in candidates if p.exists())), size)


def wrapped(draw, text, xy, width, size=22, fill=INK, bold=False, spacing=8):
    f = font(size, bold)
    line, lines = '', []
    for char in text:
        if char == '\n' or (line and draw.textlength(line + char, font=f) > width):
            lines.append(line)
            line = '' if char == '\n' else char
        else:
            line += char
    if line:
        lines.append(line)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=f, fill=fill)
        y += size + spacing
    return y


def verify(root, exp, summary, records):
    seeds = exp['seeds']
    expected = {(s, c['name']) for s in seeds for c in exp['conditions']}
    lookup = {(r['seed'], r['condition']): r for r in records}
    assert len(records) == len(lookup) == len(expected) == summary['completed'] == 12
    assert set(lookup) == expected
    assert all(r['fingerprint'] == exp['fingerprint'] for r in records)
    source = read(root / 'source_record.json')
    scored = read(root / 'results.json')
    assert scored['predictions_sha256'] == digest(root / 'predictions.jsonl')
    raw_records = [json.loads(x) for x in (root / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    raw_lookup = {(r['seed'], r['condition']): r for r in raw_records}
    original = np.asarray(Image.open(root / 'images/original.png').convert('RGB'))
    for c in exp['conditions']:
        a = np.asarray(Image.open(root / c['image']).convert('RGB'))
        assert a.shape == original.shape
        assert digest(root / c['image']) == c['png_sha256']
        assert hashlib.sha256(a.tobytes()).hexdigest() == c['rgb_sha256']
        mask = np.zeros(original.shape[:2], bool)
        if c['box_xyxy_exclusive']:
            x0, y0, x1, y1 = c['box_xyxy_exclusive']
            mask[y0:y1, x0:x1] = True
            assert int(mask.sum()) == 9216
            assert np.all(a[mask] == np.array(exp['mask_rgb']))
        assert np.array_equal(a[~mask], original[~mask])
        assert int(np.any(a != original, axis=2).sum()) == c['changed_pixels']
    for r in records:
        raw = raw_lookup[r['seed'], r['condition']]
        assert all(r[k] == v for k, v in raw.items() if k != 'grading')
        assert r['grading_target_parser'] == raw['grading']
        assert r['grading']['status'] == 'graded', 'Resolve unparsed response before final report'
        g, response, replay = r['grading'], r['model_response'], r['visual_replay']
        assert replay['prompt_token_ids'] == source['visual_replay']['prompt_token_ids']
        assert replay['image_grid_thw'] == source['visual_replay']['image_grid_thw']
        assert len(replay['output_token_ids']) == response['output_tokens']
        assert response['raw'] == (root / 'responses' / f"{r['condition']}_seed{r['seed']}.txt").read_text(encoding='utf-8')
        assert g['hit'] == int(g['extracted_answer'] == 'C' and response['finish_reason'] != 'length')
    for c in summary['conditions']:
        rr = [lookup[s, c['condition']] for s in seeds]
        assert c['correct'] == sum(r['grading']['hit'] for r in rr)
        assert c['length'] == sum(r['model_response']['finish_reason'] == 'length' for r in rr)
        for seed in seeds:
            assert c['answers'][str(seed)] == lookup[seed, c['condition']]['grading']['extracted_answer']
    result = {'passed': True, 'unique_generations': len(records), 'all_conditions_and_seeds': True,
        'all_scored_by_target_parser_or_frozen_judge': True, 'length_forced_incorrect': True,
        'exact_token_ids_present': True, 'same_prompt_and_grid': True,
        'equal_mask_areas': True, 'outside_masks_unchanged': True, 'source_rgb_verified_on_server': exp['source_rgb_matches_pinned_baseline'],
        'sha256': {p.name: digest(p) for p in [root / 'experiment.json', root / 'predictions.jsonl', root / 'summary.json']}}
    write(root / 'verification.json', result)
    return lookup


def answer(record):
    if record['model_response']['finish_reason'] == 'length':
        return '출력 제한'
    return record['grading']['extracted_answer'] or '미해결'


def audit_terminal_answers(root, exp, summary, records):
    """Case-specific analysis of the literal last lines, preserving frozen grades.

All 12 last lines were inspected manually. Two start with 'So Final answer:',
which the unchanged baseline target parser does not recognize as its field.
The intervention analysis must not mistake parser/judge changes for model changes.
"""
    display = copy.deepcopy(records)
    evidence, mismatches = [], []
    for r in display:
        lines = [x.strip() for x in r['model_response']['raw'].splitlines() if x.strip()]
        terminal = lines[-1]
        normalized = terminal.strip('*_` ')
        match = re.fullmatch(r'(?:So )?Final answer:\s*([A-E])', normalized, re.I)
        assert match, 'A new/unreviewed last-line format requires manual review'
        observed = match[1].upper()
        frozen = r['grading']['extracted_answer']
        item = {'condition': r['condition'], 'seed': r['seed'], 'exact_last_nonempty_line': terminal,
            'observed_terminal_answer': observed, 'frozen_baseline_extracted_answer': frozen,
            'frozen_method': r['grading']['method'], 'mismatch': observed != frozen,
            'response_sha256': hashlib.sha256(r['model_response']['raw'].encode()).hexdigest()}
        evidence.append(item)
        if item['mismatch']:
            mismatches.append(item)
        r['frozen_baseline_grading'] = copy.deepcopy(r['grading'])
        r['grading'] = dict(r['grading'], extracted_answer=observed,
            hit=int(observed == r['reference_answer'] and r['model_response']['finish_reason'] != 'length'),
            method='literal_terminal_answer_analysis_with_manual_review')
        r['grading']['correct'] = bool(r['grading']['hit'])
    assert {(x['condition'], x['seed'], x['observed_terminal_answer'], x['frozen_baseline_extracted_answer']) for x in mismatches} == {
        ('geometry_D', 42, 'B', 'E'), ('geometry_D', 3407, 'D', 'E')}
    audited = math13_occlusion.summarize(exp, display)
    audited.update(judge_resolved_records=summary['judge_resolved_records'],
        judge_attempts=summary['judge_attempts'], answer_source='literal_last_lines_manually_checked',
        frozen_grading_answer_mismatches=len(mismatches))
    write(root / 'answer_audit.json', {'method': 'All 12 terminal lines inspected; literal answer compared with untouched frozen grades.',
        'does_not_modify_frozen_predictions_or_results': True, 'mismatches': mismatches, 'records': evidence})
    write(root / 'analysis_summary.json', audited)
    write(root / 'analysis_results.json', {'records': display, 'summary': audited})
    return audited, display, {(r['seed'], r['condition']): r for r in display}


def render_comparison(root, exp, summary, lookup):
    panel_w, panel_h, gap = 574, 764, 24
    canvas = Image.new('RGB', (1220, 1760), BG)
    d = ImageDraw.Draw(canvas)
    d.text((36, 26), 'Math_13 · 같은 영역을 가리면 답이 달라질까?', font=font(34, True), fill=INK)
    d.text((36, 82), '같은 모델·프롬프트 | 3개 seed | 마지막 문장에서 확인한 답 | 원래 정답 C', font=font(22), fill=INK)
    for i, (cond, result) in enumerate(zip(exp['conditions'], summary['conditions'])):
        name = cond['name']
        x, y = 24 + (i % 2) * (panel_w + gap), 132 + (i // 2) * (panel_h + gap)
        d.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=16, fill='white', outline='#d9e0e9', width=2)
        d.text((x + 24, y + 18), NAMES[name], font=font(27, True), fill=COLORS[name])
        im = Image.open(root / cond['image']).convert('RGB')
        canvas.paste(im, (x + 36, y + 62))
        if cond['box_xyxy_exclusive']:
            bx0, by0, bx1, by1 = cond['box_xyxy_exclusive']
            d.rectangle((x + 36 + bx0, y + 62 + by0, x + 36 + bx1 - 1, y + 62 + by1 - 1), outline=COLORS[name], width=4)
        yy = y + 565
        for seed in exp['seeds']:
            r, orig = lookup[seed, name], lookup[seed, 'original']
            change = answer(r) if name == 'original' else f'{answer(orig)} → {answer(r)}'
            color = '#167347' if r['grading']['hit'] else '#a4262c'
            d.text((x + 30, yy), f'seed {seed}: {change}', font=font(23, True), fill=color)
            yy += 35
        d.text((x + 30, yy + 7), f"정답 C 일치 {result['correct']}/3", font=font(22, True), fill=INK)
        if name != 'original':
            d.text((x + 264, yy + 7), f"원본 대비 답 변화 {len(result['paired_answer_changes'])}/3", font=font(20), fill=INK)
    wrapped(d, '96×96 회색 마스크. 테두리 색은 보고서에만 추가했습니다. 파서·judge와 원문 답이 다른 2건은 원문 기준으로 표시했습니다. 3회 탐색 실험이며 일반적인 중요도 순위를 뜻하지 않습니다.', (30, 1716), 1155, size=16, spacing=4)
    canvas.save(root / 'comparison.png')


def render_regions(root, exp, summary):
    canvas = Image.new('RGB', (1280, 760), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((28, 22), 'Math_13 · 실험한 세 영역과 실제 답 변화', font=font(34, True), fill=INK)
    d.text((28, 77), '원문에서 확인한 최종 답 기준입니다. 실험하지 않은 픽셀의 중요도는 측정하지 않았습니다.', font=font(21), fill=INK)
    original = Image.open(root / 'images/original.png').convert('RGB')
    scale = 1.08
    original = original.resize((round(original.width * scale), round(original.height * scale)), Image.Resampling.LANCZOS)
    canvas.paste(original, (24, 132))
    for i, (c, s) in enumerate(zip(exp['conditions'][1:], summary['conditions'][1:]), 1):
        name = c['name']
        x0, y0, x1, y1 = c['box_xyxy_exclusive']
        box = (24 + round(x0 * scale), 132 + round(y0 * scale), 24 + round(x1 * scale), 132 + round(y1 * scale))
        d.rectangle(box, outline=COLORS[name], width=5)
        d.rectangle((box[0], box[1], box[0] + 29, box[1] + 32), fill=COLORS[name])
        d.text((box[0] + 6, box[1] + 1), str(i), font=font(21, True), fill='white')
        y = 139 + (i - 1) * 177
        d.text((614, y), f'{i}. {NAMES[name]}', font=font(27, True), fill=COLORS[name])
        d.text((614, y + 44), f"답 변화 {len(s['paired_answer_changes'])}/3  ·  정답 C 일치 {s['correct']}/3", font=font(24, True), fill=INK)
        d.text((614, y + 83), f"원본 정답 → 오답 {s['correct_to_wrong']}회 / 오답 → 정답 {s['wrong_to_correct']}회", font=font(20), fill=INK)
        d.text((614, y + 117), f"기존 이미지 attention의 {c['fraction_of_baseline_image_attention'] * 100:.3f}% 포함", font=font(20), fill='#56657a')
    original_result = summary['conditions'][0]
    wrapped(d, f"새 원본 응답: 정답 C 일치 {original_result['correct']}/3. 같은 seed끼리 비교한 결과입니다. 회색 가림의 효과에는 정보 제거와 새 회색 패턴의 영향이 함께 포함됩니다.", (28, 692), 1220, size=21, spacing=6)
    canvas.save(root / 'region_effects.png')


def render_report(root, exp, summary, records, lookup):
    rows, details = [], []
    for c, s in zip(exp['conditions'], summary['conditions']):
        name = c['name']
        labels = [answer(lookup[seed, name]) for seed in exp['seeds']]
        rows.append(f"| {NAMES[name]} | {' | '.join(labels)} | {s['correct']}/3 | {len(s['paired_answer_changes'])}/3 | {s['correct_to_wrong']} | {s['wrong_to_correct']} |")
        for seed in exp['seeds']:
            r = lookup[seed, name]
            details.append(f'<details><summary>{html.escape(NAMES[name])} · seed {seed} · 답 {html.escape(answer(r))} · '
                f"{r['model_response']['output_tokens']:,} tokens · {r['model_response']['elapsed_seconds']:.1f}초</summary>"
                f"<pre>{html.escape(r['model_response']['raw'])}</pre></details>")
    orig = summary['conditions'][0]
    manual_review = [
        {'condition': 'geometry_D', 'seed': 42,
         'exact_excerpt': 'Perhaps the quadrilateral ABCD is a square, and the gray square is inside it.',
         'observation_ko': '생성된 설명에서 회색 마스크를 문제의 도형으로 취급할 가능성을 제기했다. 입력 가림의 효과를 순수한 정보 삭제 효과로 분리하기 어렵다는 실제 사례다.'},
        {'condition': 'attention_high', 'seed': 42,
         'exact_excerpt': 'The quadrilateral $ABCD$ is formed by four vertices of the hexagon (A, B, C, D)',
         'observation_ko': '생성 설명은 ABCD의 네 점을 모두 육각형 꼭짓점이라고 서술했다. 원본 이미지에서 A·B는 바깥 팔각형, C·D는 안쪽 육각형의 위쪽 꼭짓점이므로 그림 관계를 잘못 서술했다.'},
    ]
    for item in manual_review:
        assert item['exact_excerpt'] in lookup[item['seed'], item['condition']]['model_response']['raw']
    write(root / 'manual_review.json', {'scope': '실제 원본·마스크 이미지와 아래 두 생성 응답의 발화를 수동 검토. 내부 계산 경로의 증명은 아님.', 'items': manual_review})
    manual_text = [f"{NAMES[item['condition']]} · seed {item['seed']}: {item['observation_ko']}" for item in manual_review]
    if orig['correct'] == 0:
        correct_scope = '새 원본도 세 seed 모두 오답이었다. 따라서 이번 결과는 선택한 답의 변화에 대한 실험이며, 정답을 유지하는 데 필수적인 영역을 입증하는 실험 결과로 읽을 수 없다.'
    else:
        correct_scope = f"새 원본에서 정답을 맞힌 실행은 {orig['correct']}/3회다. 정답을 유지하는 데 미친 영향은 그 실행의 정답→오답 전환과 함께 읽어야 한다."
    observations = []
    for c, s in zip(exp['conditions'][1:], summary['conditions'][1:]):
        changes = len(s['paired_answer_changes'])
        transitions = ', '.join(f'seed {seed}: {before}→{after}' for seed, before, after in s['paired_answer_changes']) or '세 seed 모두 최종 선택지가 같았다'
        observations.append(f"{NAMES[c['name']]}: 이미지 attention의 {c['fraction_of_baseline_image_attention'] * 100:.3f}%가 포함된 영역이다. "
            f'원본 대비 답이 바뀐 횟수는 {changes}/3회이며, {transitions}. '
            f"정답 C 일치 횟수는 원본 {orig['correct']}/3회에서 {s['correct']}/3회였다.")
    exact = summary['original_seed42_matches_historical_output_ids']
    reproduction = ('새 원본 seed 42는 과거 seed 42의 출력 토큰 ID 전체와 정확히 일치했다.' if exact else
                    '새 원본 seed 42는 과거 seed 42와 출력 토큰 ID가 완전히 같지 않았다. 영향 비교는 이번에 함께 생성한 원본을 기준으로 했다.')
    limits = [
        correct_scope,
        '이 결과는 한 문항·세 seed·한 무작위 위치의 탐색 실험이다. 조건 간 일반적인 우열이나 통계적 유의성을 주장하지 않는다.',
        '회색 마스크는 기존 정보를 지우면서 새 시각 패턴을 추가한다. 답 변화가 모두 정보 제거 때문이라고 분리해서 결론낼 수 없다.',
        'attention은 과거 seed 42의 최종 답 직전 재생값이며, 새 실험은 입력을 바꾼 뒤 풀이 전체를 다시 생성했다. 측정 대상과 시점이 다르다.',
        '무작위 영역에도 오른쪽 도형 꼭짓점·변이 포함된다. 도형이 없는 배경만 고른 대조군이 아니다.',
        'D 영역의 마스크는 일부 도형 정보만 지운다. 남아 있는 대칭·변·문제 문장으로 정보가 복원될 수 있다. 답이 유지됐다고 해당 영역이 불필요하다는 뜻은 아니다.',
        '최종 선택지가 같아도 생성한 풀이와 토큰 경로는 달라질 수 있다. 여기서 답 변화는 최종 선택지의 변화만 센다.',
        '정답 C 일치는 원래 문제의 정답을 기준으로 한다. 일부가 가려진 문제 자체가 충분히 풀 수 있는지는 별도로 평가하지 않았다.',
        '이 그림은 세 개의 실험 영역과 측정치를 표시하며, 실험하지 않은 픽셀에 중요도 값을 부여하지 않는다.',
    ]
    source = read(root / 'source_record.json')
    question = source['question']
    options = '\n'.join(f"- {k}: {v}" for k, v in source['options'].items())
    elapsed = summary['generation_seconds']
    audit_note = '원문 마지막 답과 기존 채점값이 다른 2건을 발견했다. D 가림·seed 42는 원문 `So Final answer: B`인데 judge가 E를, D 가림·seed 3407은 원문 `So Final answer: D`인데 fallback parser가 E를 기록했다. 그림과 주 비교표는 원문에서 직접 확인한 답 B·D를 사용한다. 기존 채점 파일은 변경하지 않았다. 둘 다 원래 정답 C와 다르므로 정오답 합계에는 변화가 없다.'
    body = '\n'.join([
        '# Math_13 영역 가리기 실험 결과', '',
        f"기존 `codex/visual-token-analysis` 브랜치에서 4조건 × 3개 seed, 총 {len(records)}회 새로 생성했다. 원래 정답은 C다.", '',
        '![실험 영역과 답 변화](region_effects.png)', '', '## 결과', '',
        '| 조건 | seed 42 | seed 3407 | seed 20261014 | C 일치 | 원본 대비 답 변화 | 정답→오답 | 오답→정답 |',
        '|---|---|---|---|---|---|---|---|', *rows, '',
        *['- ' + x for x in observations], '', reproduction, '',
        '답 변화 횟수는 같은 seed의 새 원본 선택지와 비교했다. 원본 행의 변화는 자기 자신과의 비교다.', '', correct_scope, '',
        '## 최종 답 대조에서 발견한 불일치', '', audit_note, '', '[12개 원문의 마지막 답과 기존 채점 대조](answer_audit.json)', '',
        '![실제 가린 이미지별 답 비교](comparison.png)', '',
        '## 조건과 검증', '',
        '원본 502×491픽셀. 모든 마스크는 96×96픽셀, 전체의 3.739%, RGB(127,127,127). '
        '마스크 밖 픽셀은 정확히 보존했다. 무손실 PNG를 입력했고 이전 JPEG 미리보기를 재사용하지 않았다.', '',
        'seed 42, 3407, 20261014. 기존 Qwen3-VL-4B-Instruct 가중치·프롬프트·해상도·temperature 0.7·top_p 0.8·top_k 20·presence_penalty 1.5·출력 상한 32,768을 유지했다. '
        '각 조건의 질문 토큰 ID와 이미지 격자가 기존 실행과 같음을 확인했다. 이전 풀이를 넣지 않고 매번 처음부터 생성했다.', '',
        f"생성 소요 시간 합계: {elapsed:.1f}초 ({elapsed / 60:.1f}분). 출력 제한: {sum(s['length'] for s in summary['conditions'])}회. "
        f"기존 target parser와 동일한 local judge를 사용했다. judge 처리 응답은 {summary.get('judge_resolved_records', 0)}개다. 원시 응답·정확한 출력 토큰 ID를 보존했다.", '',
        '## 해석 범위', '', *['- ' + x for x in limits], '',
        '## 원문 수동 검토', '', *['- ' + x for x in manual_text], '',
        '위 설명은 생성된 발화의 내용에 대한 검토다. 모델의 실제 내부 계산 과정을 충실하게 설명한다는 뜻은 아니다. [검토한 정확한 발췌](manual_review.json)', '',
        '## 원문', '', question, '', options, '',
        '## 파일', '',
        '- [검토 화면과 전체 응답](index.html)',
        '- [사전 고정 실험 명세](experiment.json)',
        '- [원시 응답 및 토큰 ID](predictions.jsonl)',
        '- [원문 답 기준 분석 결과](analysis_results.json)', '- [원문 답 기준 분석 집계](analysis_summary.json)',
        '- [보존된 기존 채점 결과](results.json)', '- [보존된 기존 채점 집계](summary.json)', '- [검증](verification.json)', '- [실행 provenance](provenance.json)', '',
    ])
    (root / 'report.md').write_text(body, encoding='utf-8')
    htmlrows = ''.join('<tr><th>' + html.escape(NAMES[s['condition']]) + '</th>' +
        ''.join(f'<td class="{"correct" if lookup[seed, s["condition"]]["grading"]["hit"] else "wrong"}">{html.escape(answer(lookup[seed, s["condition"]]))}</td>' for seed in exp['seeds']) +
        f'<td>{s["correct"]}/3</td><td>{len(s["paired_answer_changes"])}/3</td></tr>' for s in summary['conditions'])
    payload = {c['name']: {'title': NAMES[c['name']], 'image': c['image'],
        'box': c['box_xyxy_exclusive'], 'attention': c['fraction_of_baseline_image_attention'] * 100,
        'answers': [{'seed': seed, 'answer': answer(lookup[seed, c['name']]),
            'original': answer(lookup[seed, 'original']), 'correct': lookup[seed, c['name']]['grading']['hit']}
            for seed in exp['seeds']]} for c in exp['conditions']}
    style = '''body{margin:0;background:#f3f5f8;color:#172b4d;font:16px/1.65 "Malgun Gothic",sans-serif}main{max-width:1180px;margin:auto;padding:30px 24px}h1{font-size:30px;margin-bottom:8px}.sub{color:#58677d}.box,details{background:white;border:1px solid #dce3ec;border-radius:14px;padding:20px;margin:20px 0}button{background:white;border:1px solid #becada;border-radius:8px;padding:10px 16px;cursor:pointer;font:inherit;margin:5px}button.active{background:#173c6b;color:white}.viewer{display:grid;grid-template-columns:minmax(300px,1fr) 1fr;gap:30px}.viewer img{width:100%;max-width:502px}table{border-collapse:collapse;width:100%;text-align:left}td,th{padding:12px;border-bottom:1px solid #dce3ec}.correct{color:#167347;font-weight:bold}.wrong{color:#a4262c;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 monospace}.full{width:100%;height:auto}summary{cursor:pointer;font-weight:bold}a{color:#2156a0}.tag{display:inline-block;background:#e8eff7;padding:4px 10px;border-radius:6px}@media(max-width:700px){.viewer{grid-template-columns:1fr}main{padding:16px}table{font-size:13px}td,th{padding:7px}}'''
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Math_13 영역 가리기 실험</title><style>' + style + '</style><main>'
    page += '<span class="tag">12회 생성 완료 · 원시 결과 검증 완료</span><h1>Math_13: 영역을 가렸을 때 답에 미친 영향</h1><p class="sub">4조건 × 3개 seed · 동일 면적의 회색 마스크 · 원래 정답 C · 기존 시각 토큰 분석 브랜치</p>'
    page += '<img class="full" src="region_effects.png" alt="실험한 세 영역과 답 변화"><section class="box"><h2>실제 모델 입력 비교</h2><div id="buttons">'
    page += ''.join(f'<button data-condition="{c["name"]}">{html.escape(NAMES[c["name"]])}</button>' for c in exp['conditions'])
    page += '</div><div class="viewer"><div><img id="inputImage" src="images/original.png" alt="모델에 입력한 이미지"></div><div><h3 id="conditionTitle"></h3><p id="conditionInfo"></p><div id="answers"></div><p>좌표는 원본 이미지 기준입니다. 이 화면에는 모델에 입력한 PNG를 그대로 표시합니다.</p></div></div></section>'
    page += '<section class="box"><h2>같은 seed의 새 원본과 비교</h2><table><tr><th>조건</th><th>42</th><th>3407</th><th>20261014</th><th>C 일치</th><th>답 변화</th></tr>' + htmlrows + '</table><p>' + html.escape(reproduction) + '</p></section>'
    page += '<section class="box"><h2>원문 답과 채점값의 불일치 2건</h2><p>' + html.escape(audit_note) + '</p><p><a href="answer_audit.json">모든 마지막 문장의 대조 기록</a> · <a href="summary.json">보존된 기존 채점 집계</a></p></section>'
    page += '<section class="box"><h2>관찰한 결과</h2><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in observations) + '</ul><h3>해석 범위</h3><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in limits) + '</ul></section>'
    page += '<section class="box"><h2>원문 수동 검토</h2><ul>' + ''.join('<li>' + html.escape(x) + '</li>' for x in manual_text) + '</ul><p>생성된 발화의 검토이며 모델 내부 계산 경로의 증명은 아닙니다. <a href="manual_review.json">정확한 발췌</a></p></section>'
    page += '<details><summary>기존 최종 답 직전 attention</summary><img src="images/baseline_attention.jpg" alt="기존 attention"><p>저장된 seed 42 생성 경로의 Transformers 재생값. 네 decoder layer의 head 평균입니다. 영역 가리기 효과와 다른 측정입니다.</p></details>'
    page += '<h2>생성된 전체 풀이</h2>' + ''.join(details)
    page += '<p><a href="report.md">한국어 보고서</a> · <a href="comparison.png">조건별 비교 PNG</a> · <a href="experiment.json">실험 명세</a> · <a href="analysis_summary.json">원문 답 기준 집계</a> · <a href="verification.json">검증</a></p></main>'
    page += '<script>const data=' + json.dumps(payload, ensure_ascii=True) + ';' + '''function select(name){const c=data[name];document.querySelectorAll('[data-condition]').forEach(b=>b.classList.toggle('active',b.dataset.condition===name));document.getElementById('inputImage').src=c.image;document.getElementById('conditionTitle').textContent=c.title;document.getElementById('conditionInfo').textContent=c.box?'좌표 ['+c.box.join(', ')+') · 기존 이미지 attention의 '+c.attention.toFixed(3)+'% 포함':'마스크가 없는 데이터 원본';const a=document.getElementById('answers');a.replaceChildren();c.answers.forEach(r=>{const p=document.createElement('p');p.className=r.correct?'correct':'wrong';p.textContent='seed '+r.seed+': '+(name==='original'?r.answer:r.original+' → '+r.answer);a.append(p);});}document.querySelectorAll('[data-condition]').forEach(b=>b.onclick=()=>select(b.dataset.condition));select('original');</script></html>'''
    (root / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    root = Path(args.results).resolve()
    exp, summary = read(root / 'experiment.json'), read(root / 'summary.json')
    records = read(root / 'results.json')['records']
    verify(root, exp, summary, records)
    summary, records, lookup = audit_terminal_answers(root, exp, summary, records)
    render_comparison(root, exp, summary, lookup)
    render_regions(root, exp, summary)
    render_report(root, exp, summary, records, lookup)
    print(json.dumps({'report': str(root / 'index.html'), 'verified': len(records)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
