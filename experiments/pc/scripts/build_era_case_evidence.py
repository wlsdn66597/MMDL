from pathlib import Path
import json, hashlib, html, base64
from fractions import Fraction as F

ROOT = Path('C:/project/MMDL')
SOURCE = Path('C:/김선재/대학교/3학년/2학기/멀티모달딥러닝/프로젝트/아이디어 다시 정하기')
OUT = ROOT / 'reports/era_error_cases_20261012'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'images').mkdir(exist_ok=True)
filenames = {1: 'experiment_1_official.json', 3: 'experiment_3_final_answer_line.json'}
datasets = {n: json.loads((SOURCE / f).read_text(encoding='utf-8-sig')) for n, f in filenames.items()}
sha = lambda b: hashlib.sha256(b).hexdigest()

cases = [
dict(key='A1', id='validation_Math_5', exp=3, title='영역의 경계와 x축 절편을 혼동한 적분 문제', category='A 시각적 해석 및 영역 대응',
question_ko='그림의 영역 R의 넓이를 나타내는 적분식을 선택한다.', answer='C: 적분 상한 2 → A: 적분 상한 1.5',
quotes=['And the shaded region $ R $ is bounded between $ x = 0 $ and $ x = 2 $, as shown in the graph.', 'Final answer: C'],
observed='모델은 두 함수식을 정확히 옮겼지만, R의 오른쪽 경계를 x=2로 잡았다. 그림의 (2,0)은 f의 x축 절편이며 두 곡선의 교점이 아니다. 실제로 f(2)=0, g(2)=2이다. x=1에서 f>g임을 확인한 뒤 이를 잘못 정한 구간 전체에 적용했다.',
era=['Evidence: 그림의 R은 원점부터 두 곡선의 첫 양의 교점까지의 영역이다. f(x)=x³−6x²+8x, g(x)=−x²/2+2x이며 (2,0)은 f의 x축 절편이다.', 'Rule: 두 곡선 사이 넓이는 해당 영역의 교점을 경계로 하여 위 함수에서 아래 함수를 뺀 값을 적분한다.', 'Apply: f−g=x(x−1.5)(x−4)이므로 R의 경계는 0과 1.5이다. 이 구간에서 f>g이므로 ∫₀¹·⁵[f(x)−g(x)]dx를 선택한다.', 'Final answer: A'],
hypothesis='그림의 숫자를 곧바로 적분 경계에 대입하지 않고, 경계가 의미하는 교점 조건과 대조하도록 학습한다.',
diagnostic='그림의 축 눈금·절편·음영 영역을 바꾼 별도 문제에서 경계 추출 정확도와 최종 정답률을 측정한다.',
limits='실험 1에서도 C로 오답이다. 올바른 넓이 공식을 썼다는 사실은 확인되지만, 모든 넓이 문제의 규칙을 안정적으로 안다는 뜻은 아니다.'),
dict(key='A2', id='validation_Physics_13', exp=1, title='속도 그래프의 좌표를 잘못 읽은 위치 복귀 문제', category='A 시각적 해석 및 도표 대응',
question_ko='속도와 시간 그래프에서 물체가 처음 위치로 다시 돌아오는 시간을 찾는다.', answer='C: 2초 → B: 1초와 2초 사이',
quotes=['we need to find when the **displacement** (area under the velocity-time graph) returns to zero', 'The graph goes from (1, -1) up to (2, 2)', '**C. 2 s**'],
observed='모델은 누적 변위가 0이 되는 시점을 구해야 한다는 규칙을 명시했다. 그러나 실제 최저점 (0.5,−1)을 (1,−1)로 읽었고, 실제 (1,0)에서 시작하는 양의 구간을 잘못 구성했다. 그 결과 1~2초의 면적을 +0.5로 계산하여 2초를 선택했다.',
era=['Evidence: 그래프의 주요 점은 (0,0), (0.5,−1), (1,0), (2,2)이다. 0~1초의 부호 있는 면적은 −0.5m이다.', 'Rule: 처음 위치로 복귀하려면 t>0에서 누적 변위, 즉 속도 그래프의 부호 있는 면적 합이 0이어야 한다.', 'Apply: 1~2초에서는 v(t)=2(t−1)이므로 추가 변위는 (t−1)²이다. −0.5+(t−1)²=0에서 t=1+√0.5≈1.707초이므로 1초와 2초 사이이다.', 'Final answer: B'],
hypothesis='알고 있는 변위 규칙을 실제 그래프의 좌표와 연결하도록, 주요 점·구간을 먼저 확인하는 풀이를 학습한다.',
diagnostic='최저점·영점·기울기를 바꾼 독립적인 그래프에서 좌표 추출, 부호 있는 면적, 최종 시간 구간을 각각 평가한다.',
limits='실험 3에서는 B로 정답이었다. 따라서 두 실험에서 지속된 오류라고 쓰지 않는다. 이 사례는 규칙을 언급해도 시각적 근거를 잘못 읽으면 실패할 수 있음을 보여준다.'),
dict(key='C1', id='validation_Accounting_1', exp=3, title='총고정비와 단위당 고정비를 혼동한 원가 문제', category='C 도메인 개념 및 조건 적용',
question_ko='25,000개 생산 시의 단위당 원가 표를 보고 30,000개 생산 시 단위당 제조간접비를 구한다. 관련 범위는 15,000~30,000개이다.', answer='C: $8 → B: $7',
quotes=['so per unit it remains $6 even if volume changes within the relevant range', '= $6 + $2 = **$8**', '**Final answer: C**'],
observed='모델은 표의 고정 제조간접비 $6과 변동 제조간접비 $2를 읽었다. 그러나 관련 범위 안에서 단위당 고정비가 $6으로 유지된다고 설명하며 $6+$2=$8을 선택했다. 유지되는 것은 총고정비이며, 생산량이 늘면 단위당 고정비는 감소한다.',
era=['Evidence: 기준 생산량은 25,000개, 변경 생산량은 30,000개이다. 표의 단위당 고정 제조간접비는 $6, 변동 제조간접비는 $2이며 변경 생산량은 관련 범위 안이다.', 'Rule: 관련 범위에서는 총고정비와 단위당 변동비를 일정하게 두고, 새 단위당 고정비는 총고정비를 새 생산량으로 나누어 계산한다.', 'Apply: 총고정비=$6×25,000=$150,000. 새 단위당 제조간접비=$150,000/30,000+$2=$7.', 'Final answer: B'],
hypothesis='고정이라는 단어에 반응하는 데서 그치지 않고, 무엇이 고정되는지와 단위·기준 생산량을 명시해 조건에 맞게 변환하도록 학습한다.',
diagnostic='표의 행 순서, 생산량, 관련 범위를 바꾼 별도 문항에서 총액과 단위당 금액을 구별하는지 평가한다.',
limits='실험 1에서는 B로 정답이었다. 이 응답에는 고정비 개념 자체의 잘못된 서술도 있어, 규칙을 이미 완전히 알지만 적용만 실패했다고 단정할 수 없다. 정답 해설은 기준 정답 B에 따라 표의 명시된 fixed/variable manufacturing overhead 두 항목을 사용한다. 원문에는 average variable costs라는 표현과 indirect materials 별도 행도 있으므로, 독립 진단 문항 제작 시 총액/단위당 금액 및 포함 범위를 명확히 한다.'),
dict(key='C2', id='validation_Physics_18', exp=3, title='병렬회로에서 전체 전류와 분기 전류를 혼동한 전구 문제', category='C 규칙 적용과 A 회로 연결 해석의 복합 사례',
question_ko='동일한 전구 네 개의 회로에서 스위치를 닫아 4번 전구를 켰을 때, 나머지 중 어떤 전구가 더 밝아지는지 묻는다.', answer='D: 1·2·3번 → A: 1번만',
quotes=['I2 = I_total = 3ε/(4R)  [since in parallel, voltage is the same]', 'When S is closed, the branch has bulb #4 and switch S in series, resistance 2R.', 'Final answer: D'],
observed='모델은 연결 구조를 여러 번 바꾸어 설명했다. 풀이 중 병렬 연결은 전압이 같다는 이유로 2번 전구의 전류를 전체 전류와 같게 놓았고, 닫힌 스위치에 전구와 같은 저항 R을 더하는 서술도 했다. 최종적으로 1·2·3번이 모두 밝아진다고 선택했다. 출력은 7,556토큰이며 stop으로 끝나 길이 제한 오답이 아니다.',
era=['Evidence: 1번 전구는 전체 회로와 직렬이다. 2번과 3번은 같은 두 노드 사이의 병렬 가지이고, 4번과 스위치의 직렬 가지가 이 두 노드에 추가된다.', 'Rule: 이상적인 전원·스위치 및 동일 저항 R 모델에서, 병렬 가지에는 같은 전압이 걸리고 전체 전류는 가지 전류의 합이다. 전구 밝기는 전력으로 비교한다.', 'Apply: 스위치가 열리면 전체 저항은 R+R/2=3R/2, 닫히면 R+R/3=4R/3이다. 전체 전류는 2ε/(3R)에서 3ε/(4R)로 증가하므로 1번은 밝아진다. 병렬부 전압은 ε/3에서 ε/4로 감소하여 2·3번은 어두워진다.', 'Final answer: A'],
hypothesis='회로 노드와 가지를 일관되게 확정하고, 병렬 전압·전류 분배·전력을 구분하여 적용하는 풀이를 학습한다.',
diagnostic='같은 회로를 다른 배치로 그린 그림과 연결을 실제로 바꾼 별도 회로를 함께 평가한다. 연결 목록, 가지별 전류, 밝기 선택을 확인한다.',
limits='A와 C가 함께 나타나는 복합 사례다. 원본 집계의 단일 대표 유형을 재확정한 것이 아니며, 각 오류가 최종 선택에 기여한 정도는 분리해서 입증하지 않았다. 실험 1의 같은 문항은 length 종료이므로 여기서는 실험 3만 대표 근거로 사용한다. 전력 수치는 동일한 고정 저항 R을 사용하는 교과서 근사에 따른다.'),
]

checks = {'math_f2': 0, 'math_g2': 2, 'math_intersection_1_5': str(F(15,8)),
          'physics_negative_area': str(-F(1,2)), 'physics_positive_area_to_2': 1,
          'accounting_overhead': str(F(6*25000,30000)+2),
          'bulb1_power_ratio': str((F(3,4)/F(2,3))**2),
          'bulb2_3_power_ratio': str((F(1,4)/F(1,3))**2)}
assert F(3,2)**3-6*F(3,2)**2+8*F(3,2) == -F(1,2)*F(3,2)**2+2*F(3,2) == F(15,8)
assert checks['accounting_overhead'] == '7'

for c in cases:
    records = datasets[c['exp']]['records']
    index, r = next((i,r) for i,r in enumerate(records) if r['id']==c['id'])
    assert r['grading']['final']['correct'] is False
    assert r['model_response']['finish_reason'] == 'stop'
    for q in c['quotes']:
        assert q in r['model_response']['raw'], (c['key'], q)
    old = ROOT / 'tmp/baseline_audit_20260928/images' / c['id'] / 'image_1.png'
    new = ROOT / 'results/mmmu_first_baseline_20261009/analysis/images' / (c['id']+'_image_1.png')
    assert old.read_bytes() == new.read_bytes(), c['id']
    image_bytes = old.read_bytes()
    image_name = c['id']+'.png'
    (OUT / 'images' / image_name).write_bytes(image_bytes)
    c.update(source_file=str(SOURCE / filenames[c['exp']]), source_file_sha256=sha((SOURCE / filenames[c['exp']]).read_bytes()),
             source_pointer=f'/records/{index}', source_record=r, image_file='images/'+image_name,
             image_sha256=sha(image_bytes), raw_response_sha256=sha(r['model_response']['raw'].encode()),
             quote_offsets=[{'quote':q, 'start':r['model_response']['raw'].index(q)} for q in c['quotes']])
    c['other_experiment'] = [{'experiment':n, 'correct':s['grading']['final']['correct'], 'extracted_answer':s['grading']['final']['extracted_answer'], 'finish_reason':s['model_response']['finish_reason']} for n,d in datasets.items() for s in d['records'] if s['id']==c['id']]

intro = '실험 1과 실험 3의 저장된 원본 응답에서 선정한 실제 오답 4건이다. ERA 부분은 오류 분석을 바탕으로 작성한 목표 풀이 예시이며, 학습 후 모델의 실제 출력이나 개선 성과가 아니다. 아래 A·C 표기는 이번 사례 검토에서 확인한 오류 성격이며, 기존 집계표의 문항별 대표 분류를 재현한 것은 아니다.'
protocol = '이 MMMU validation 문항과 정답·ERA 재작성은 연구 동기 설명 및 오류 분석에만 사용한다. 학생·교사 학습 데이터와 개발셋에 넣거나 해당 문항을 변형해 학습시키지 않는다. 별도 출처·문항군으로 일반 풀이와 ERA 풀이를 짝지어 LoRA-SFT하고, 동일한 baseline 평가 프롬프트에서 1회 생성하여 비교한다. 이미 분석에 사용한 MMMU validation 성능과 별도의 미열람 진단셋 성능을 구분해서 보고한다.'
md = ['# ERA 연구 근거용 실제 오답 사례', '', '작성일: 2026-10-12', '', intro, '', '## 사례 요약', '', '| 사례 | 실험 | 문항 | 오류 성격 | 기존 답 → 기준 정답 |','|---|---|---|---|---|']
for c in cases:
    md.append(f"| {c['key']} | {c['exp']} | {c['id']} | {c['category']} | {c['answer']} |")
md += ['', '## 공통 해석과 실험 원칙', '', '실제 오답은 연구 필요성의 근거다. ERA의 효과를 입증하려면 같은 문제·근거·규칙·계산·정답을 가진 일반 풀이 학습과 ERA 풀이 학습을 비교해야 한다. ERA 표제의 출력 여부만으로 추론 개선을 판정하지 않는다.', '', protocol, '']
for c in cases:
    r=c['source_record']
    md += [f"## {c['key']} {c['title']}", '', f"- 문항: `{c['id']}` / 실험 {c['exp']} / {c['category']}", f"- 결과: **{c['answer']}**", f"- 종료: `{r['model_response']['finish_reason']}` / 출력 {r['model_response']['output_tokens']:,}토큰", '', c['question_ko'], '', f"![원본 문제 이미지]({c['image_file']})", '', '**기존 응답의 실제 인용**', '']
    for q in c['quotes']: md += ['> '+q, '']
    md += ['**관찰된 오류**', '', c['observed'], '', '**ERA 목표 풀이 예시 — 개선 결과가 아닌 학습 방향**', '', '```text', *c['era'], '```', '', '**개선 가설과 확인 방법**', '', c['hypothesis'], '', c['diagnostic'], '', '**해석 범위**', '', c['limits'], '', '**원본 추적 정보**', '', f"원본: `{c['source_file']}`", '', f"JSON 위치: `{c['source_pointer']}`. 전체 질문·보기·원문 응답·채점 기록은 동봉된 `evidence.json`에 보존했다.", '', f"응답 SHA-256: `{c['raw_response_sha256']}`", '']
md += ['## 발표용 결론 문장', '', '> 저장된 응답에서는 올바른 공식을 언급하더라도 그래프의 경계와 좌표를 잘못 대응하여 오답을 내거나, 고정비와 병렬회로의 적용 조건을 혼동하는 사례가 관찰되었다. 이에 시각적 근거, 적용 규칙과 조건, 문제별 적용 과정을 연결한 ERA 풀이를 학습시키고자 한다. 이 사례들은 연구 가설의 동기이며, ERA의 개선 효과는 동일한 평가 프롬프트에서 일반 풀이 SFT와 비교하여 검증한다.', '', '## 검증 범위', '', '- 네 대표 응답은 모두 원본 채점에서 오답이며 length 종료가 아니다.', '- 인용문은 원문 부분 문자열로 검증했다. 실험 1·3과 이후 최초 baseline 재실행은 섞지 않았다.', '- 이미지는 기존 이미지 추출본과 후속 baseline의 동일 ID 이미지가 바이트 단위로 일치함을 확인했다. 원본 실험 파일은 이미지 픽셀을 포함하지 않으므로 당시 입력 텐서까지 동일하다고 주장하지 않는다.', '- 원본 파일과 응답, 이미지의 SHA-256 및 JSON 인덱스를 evidence.json에 기록했다.', '- 수학적 교점, 면적, 고정비 재배분, 동일 저항 회로의 전력 비율을 확인했다.', '']
(OUT/'report.md').write_text('\n'.join(md),encoding='utf-8')
(OUT/'evidence.json').write_text(json.dumps({'created_at':'2026-10-12','status':'observational_cases_and_proposed_era_not_training_results','cases':cases,'checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')

e=html.escape
parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>ERA 연구 근거용 실제 오답 사례</title><style>body{font:16px/1.75 "Malgun Gothic",sans-serif;color:#17212b;max-width:1040px;margin:40px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd3da;padding:10px;text-align:left}img{max-width:100%;max-height:420px;object-fit:contain}blockquote{background:#f3f5f7;padding:12px 18px;border-left:4px solid #708397;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:18px;font:14px/1.7 Consolas,"Malgun Gothic",monospace}small{overflow-wrap:anywhere}section{border-top:1px solid #ccd3da;margin-top:36px;padding-top:12px}.note{background:#fff5d9;padding:14px}summary{cursor:pointer;font-weight:bold}@media print{body{margin:0;font-size:11pt}section{break-before:page}details{display:none}}</style><h1>ERA 연구 근거용 실제 오답 사례</h1><p>2026-10-12 · 실험 1과 실험 3 · Qwen3-VL-4B-Instruct</p>',f'<p class="note">{e(intro)}</p>','<table><tr><th>사례</th><th>실험과 문항</th><th>오류 성격</th><th>기존 답 → 기준 정답</th></tr>']
for c in cases: parts.append(f"<tr><td>{c['key']}</td><td>{c['exp']} / {e(c['id'])}</td><td>{e(c['category'])}</td><td>{e(c['answer'])}</td></tr>")
parts += ['</table>', f'<p>{e(protocol)}</p>']
for c in cases:
    r=c['source_record']; img=base64.b64encode((OUT/c['image_file']).read_bytes()).decode()
    parts += [f"<section><h2>{c['key']} {e(c['title'])}</h2><p><b>실험 {c['exp']} · {e(c['id'])}</b><br>{e(c['category'])}<br>{e(c['answer'])} · 종료 {r['model_response']['finish_reason']} · {r['model_response']['output_tokens']:,}토큰</p><p>{e(c['question_ko'])}</p><img alt='원본 문제 이미지' src='data:image/png;base64,{img}'><h3>기존 응답의 실제 인용</h3>"]
    parts += [f'<blockquote>{e(q)}</blockquote>' for q in c['quotes']]
    parts += [f"<h3>관찰된 오류</h3><p>{e(c['observed'])}</p><h3>ERA 목표 풀이 예시</h3><p>아래는 개선 결과가 아닌 학습 방향이다.</p><pre>{e(chr(10).join(c['era']))}</pre><h3>개선 가설과 확인 방법</h3><p>{e(c['hypothesis'])}</p><p>{e(c['diagnostic'])}</p><h3>해석 범위</h3><p>{e(c['limits'])}</p><details><summary>원본 질문·보기·전체 응답 확인</summary><pre>{e(r['question'])}\n\n{e(json.dumps(r['options'],ensure_ascii=False,indent=2))}\n\n{e(r['model_response']['raw'])}</pre></details><p><small>원본: {e(c['source_file'])}<br>JSON 위치: {c['source_pointer']}<br>응답 SHA-256: {c['raw_response_sha256']}</small></p></section>"]
parts += ['<section><h2>발표용 결론 문장</h2><p>저장된 응답에서는 올바른 공식을 언급하더라도 그래프의 경계와 좌표를 잘못 대응하여 오답을 내거나, 고정비와 병렬회로의 적용 조건을 혼동하는 사례가 관찰되었다. 이에 시각적 근거, 적용 규칙과 조건, 문제별 적용 과정을 연결한 ERA 풀이를 학습시키고자 한다. 이 사례들은 연구 가설의 동기이며, ERA의 개선 효과는 동일한 평가 프롬프트에서 일반 풀이 SFT와 비교하여 검증한다.</p><p>인용문은 원문과 대조했으며 각 대표 응답은 오답 및 stop 종료임을 확인했다. 사용 이미지는 기존 추출본과 후속 baseline의 동일 ID 이미지가 바이트 단위로 같았다. 당시 입력 텐서의 동일성까지 검증한 것은 아니다. 원본 기록·파일 해시·계산 점검은 evidence.json을 참조한다.</p></section></html>']
(OUT/'report.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps({'out':str(OUT),'cases':len(cases),'quotes_verified':sum(len(c['quotes']) for c in cases),'checks':checks},ensure_ascii=False))

