"""Build the final Korean report from verified data and inspected cases."""
from pathlib import Path
import datetime as dt
import json
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/mmmu_visual_tokens_seed42_20261010'
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def duration(seconds):
    seconds = round(seconds)
    return f'{seconds//3600}시간 {(seconds%3600)//60}분 {seconds%60}초'
def date(value):
    return dt.datetime.fromisoformat(value).astimezone(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')

v = read(RUN / 'verification.json')
assert all(v['checks'].values())
new = read(RUN / 'baseline_seed42/results.json')
old = read(ROOT / 'results/mmmu_first_baseline_20261009/baseline_seed3407/results.json')
a = read(RUN / 'analysis/analysis_summary.json')
status = read(RUN / 'status.json')
launch = read(RUN / 'status_at_launch.json')
records = new['records']
result_time = dt.datetime.fromisoformat(new['created_at'])
end_time = dt.datetime.fromisoformat(status['updated_at'])
evaluation_seconds = (result_time-dt.datetime.fromisoformat(launch['updated_at'])).total_seconds()
after_grading_seconds = (end_time-result_time).total_seconds()
old_records = {r['id']: r for r in old['records']}
length_old = {r['id'] for r in old['records'] if r['model_response']['finish_reason']=='length'}
length_new = {r['id'] for r in records if r['model_response']['finish_reason']=='length'}
prefix = '../results/mmmu_visual_tokens_seed42_20261010/'
lines = ['# MMMU baseline seed 42: 900문항 평가 및 시각 토큰 분석', '',
    f'**551/900 정답(61.22%)**, 오답 349문항. 900문항 채점과 heatmap 생성은 **{date(status["updated_at"])}**에 완료됐다. '
    'seed 3407의 575/900(63.89%) 대비 24문항, 2.67%p 낮다. 모델 학습이나 가중치 변경 없이 추론 seed를 바꾼 실행이며, 이 차이만으로 모델 자체의 성능 퇴화를 주장하지 않는다.', '',
    f'[전체 원본·heatmap·답변 검토 화면]({prefix}analysis/index.html) · [검증 결과]({prefix}verification.json) · [원시 집계]({prefix}analysis/analysis_summary.json)', '',
    '## 검증된 완료 범위', '',
    f'- 예측과 최종 결과 각각 고유 ID 900개, 고정 문항 목록 및 이전 baseline과 ID 일치. 전 문항 graded.',
    '- 문항별 정답 합계·전체/과목별 집계·채점 경로 집계를 재계산하여 저장 결과와 일치. length 83건 모두 오답.',
    '- 생성 당시 저장한 입력/출력 토큰 수·ID가 predictions와 최종 결과에서 일치. 모든 분석 metadata의 원본 record hash 및 fingerprint 일치.',
    f'- 900개 metadata·HTML, {v["attention_arrays_verified"]:,}개 원시 layer별 attention 배열, {v["query_probes_verified"]:,}개 query probe와 {v["image_files_verified"]:,}개 원본/overlay JPEG 검증. 배열의 크기·유한성·확률 범위·이미지 격자 및 mass 집계를 확인했다.',
    '- 서버와 로컬 아카이브 SHA256 일치. 서버에서 driver 종료, GPU 사용률 0%, 메모리 24 MiB를 확인했다.',
    '- 사전 기록에서 900개 프롬프트·보기·정답 및 원본 이미지 RGB hash 일치. 이번 검증이 Judge의 의미적 정확성을 900문항 모두 인증하는 것은 아니다.', '',
    '## seed 3407과 비교', '',
    '| 항목 | seed 3407 | seed 42 |', '|---|---:|---:|',
    '| 정답 | 575/900 | 551/900 |', '| 정확도 | 63.89% | 61.22% |',
    '| 출력 상한 종료 | 81 (9.00%) | 83 (9.22%) |',
    '| 정상 종료 문항만의 정답률(진단용) | 575/819 = 70.21% | 551/817 = 67.44% |', '',
    '| 문항 전환 | 수 |', '|---|---:|', '| 정답 → 정답 | 499 |', '| 오답 → 정답 | 52 |',
    '| 정답 → 오답 | 76 |', '| 오답 → 오답 | 273 |', '',
    f'정오답이 바뀐 문항은 128/900(14.22%)다. 두 실행 모두 length인 문항 {len(length_old & length_new)}개, '
    f'이번에 새로 length가 된 문항 {len(length_new-length_old)}개, 이전 length에서 벗어난 문항 {len(length_old-length_new)}개다. '
    '잘림 수의 순증은 2개이지만 개별 문항은 바뀌었다. 잘림 증가만으로 24문항 점수 차이를 설명하지 않는다. '
    '추론 샘플링과 저장 응답의 차이를 포함한 두 실행의 비교이며, 여러 학습 seed에 대한 안정성 실험은 아니다.', '',
    '## 실측 소요 시간', '',
    '| 구간 | 시간 |', '|---|---|',
    f'| campaign 시작(사전 검증 포함) | {dt.datetime.fromtimestamp(status["started_at_unix"],dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")} |',
    f'| 평가 호출 시작 | {date(launch["updated_at"])} |',
    f'| 채점 결과 저장 | {date(new["created_at"])} |',
    f'| 전체 완료 | {date(status["updated_at"])} |',
    f'| 사전 검증부터 전체 완료 | {duration(status["elapsed_seconds"])} |',
    f'| 평가 호출부터 채점 결과 저장 | {duration(evaluation_seconds)} |',
    f'| 채점 결과 저장 이후 전체 완료 | {duration(after_grading_seconds)} |',
    f'| 문항별 생성 시간 합계 | {duration(v["generation_seconds_sum"])} |',
    f'| length 문항 생성 시간 합계 | {duration(v["length_generation_seconds_sum"])} |', '',
    f'length 83문항은 전체 생성 시간 합계의 {v["length_generation_seconds_sum"]/v["generation_seconds_sum"]*100:.2f}%를 차지했다. '
    '평가 호출 구간에는 엔진 초기화·추론·Judge가 포함된다. 채점 이후 구간에는 결과 검증·분석 모델 준비·heatmap 재생·파일 및 HTML 생성이 포함되며 순수 GPU 계산 시간과 같지 않다. '
    '로컬 다운로드·최종 검수 시간은 위 서버 소요 시간에서 제외했다.', '',
    '## 채점 경로', '', '| 경로 | 문항 수 | 정답 수 |', '|---|---:|---:|']
for method, numbers in v['methods'].items():
    lines.append(f'| {method} | {numbers["count"]} | {numbers["correct"]} |')
lines += ['', f'Judge 대상은 {v["judge_unique_ids"]}문항, 실제 시도는 {v["judge_attempts"]}회다. '
          '최종 valid_option 26, no_match 21, unparsed 2이며 무작위 fallback은 0건이다. '
          '정답 표기는 고정된 기존 채점 규칙의 결과이며, 이번 분석에서 수동으로 점수를 고치지 않았다.', '',
          '## 이미지 attention에서 확인한 점', '',
          '아래 비중은 모든 causal key에 대한 softmax에서 이미지 토큰에 배정된 확률의 합이다. 이미지 안에서만 정규화한 점수가 아니다. '
          '네 decoder layer(0-based 8, 17, 26, 35)의 head 평균을 다시 평균했다.', '',
          '| 그룹 | 첫 답변 토큰 직전 | 최종 답 값 직전 | 마지막 토큰 직전 |', '|---|---:|---:|---:|']
for label, group in [('정답', 'correct'), ('오답', 'incorrect'), ('length 종료', 'finish_length')]:
    values = a['attention_by_group'][group]
    cells = [f'{values[n]["mean_image_attention_mass"]*100:.3f}% (n={values[n]["count"]})' for n in ['before_first_token','before_final_answer','before_last_token']]
    lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
lines += ['', '정답·오답의 평균은 유사하다. 이미지 attention 총량의 평균만으로 정오답을 구분할 근거는 보이지 않는다. '
          '마지막 단계에서 비중이 낮아지는 현상은 텍스트 문맥 길이와 위치의 영향도 받으므로 이미지 정보를 잊었다는 증거로 해석하지 않는다. '
          'Final answer가 있는 문항만 해당 probe가 있어 그룹별 표본 수가 다르다. 전체 900문항의 완전한 층별·head별·시간별 분석이나 인과적 검증은 아니다.', '',
          '## 직접 검토한 대표 사례', '',
          '아래 네 문항은 원본 이미지, 실제 overlay, 모델 응답과 채점을 함께 검토했다. 모든 900문항의 오류 원인을 수동 분류한 것은 아니다.', '',
          '### Accounting_1: 표의 수치를 읽었어도 비용 개념에서 오류', '',
          '두 seed 모두 C로 오답이다. 원본 표에는 고정 제조간접비 6, 변동 제조간접비 2가 보인다. 응답도 6과 2를 명시하지만 고정비의 단위당 비용이 생산량과 무관하게 6이라고 설명했다. '
          '25,000개 생산 기준 고정 총액을 30,000개로 나누면 6×25,000/30,000+2=7로 정답 B다. 따라서 이 사례에서는 단순 수치 누락보다 총액·단위당 비용 개념 오류가 응답에 드러난다. '
          '최종 답 직전 이미지 attention 비중은 3.24%이고 overlay의 강한 부분은 표 위쪽 여백이다. 이 분포만으로 특정 숫자를 읽었는지 여부를 판정할 수 없다.', '',
          f'[원본·overlay·전체 응답]({prefix}analysis/validation_Accounting_1/index.html)', '',
          '### Electronics_6: 오답에서 정답으로 바뀌었지만 heatmap은 공백 집중', '',
          'seed 3407은 약 10.7의 계산 결과로 오답이었고, seed 42는 풀이에서 약 71.68 V를 계산한 뒤 최종 72를 출력했다. 기준 답은 71.6이고 고정 Judge가 이번 응답을 정답 처리했다. '
          '원본의 회로·임피던스 표시와 overlay를 비교하면, 선택한 probe의 강한 지점은 오른쪽 위 공백에 있다. 이미지 attention 비중은 첫 답변 직전 8.63%에서 최종 답 직전 2.01%였다. '
          '이는 정답 전환 사례지만, 공백의 높은 attention을 회로를 이해한 핵심 근거라고 부를 수 없다. Judge의 수치 허용 범위가 명시적으로 정의되지 않았다는 기존 제한도 남아 있다.', '',
          f'[원본·overlay·전체 응답]({prefix}analysis/validation_Electronics_6/index.html)', '',
          '### Chemistry_12: 보기 이미지 attention 감소와 반복 풀이', '',
          '원본에는 세 교차점이 있는 Fischer 구조와 보기 네 장이 있다. 두 seed 모두 출력 상한 종료로 오답이다. 이번 응답 끝부분은 H·Br·OH·Me와 Et의 배치를 여러 방식으로 다시 해석하며 확정 답 없이 끊긴다. '
          '다섯 이미지 전체의 attention 비중은 첫 답변 직전 8.30%에서 마지막 토큰 직전 3.77%로 감소했다. 그중 문제 이미지 하나는 3.65%→3.11%이고, 보기 네 장의 합은 4.65%→0.66%다. '
          '마지막 문제 이미지 overlay는 상단 여백에 강하게 집중된다. 반복 추론과 보기 참조 감소를 함께 관찰할 수 있지만, 이 변화가 잘림의 원인이라는 인과적 결론은 내리지 않는다. Final answer probe는 없다.', '',
          f'[다섯 이미지와 overlay·전체 응답]({prefix}analysis/validation_Chemistry_12/index.html)', '',
          '### Music_21: 질문 대상이 있는 페이지와 강한 attention 위치 불일치', '',
          '질문은 11마디의 제3호른과 제2바이올린 사이 음정을 묻는다. 두 번째 이미지에는 10마디부터의 악보와 번호 3의 괄호가 보인다. 모델은 설명 없이 B를 출력했고 기준 정답은 C여서 오답이다. '
          '최종 답 직전 이미지 attention 비중은 첫 페이지 3.93%, 두 번째 페이지 0.95%다. 강한 overlay 지점은 첫 페이지 상단 공백이며, 질문 대상인 두 번째 페이지의 괄호 영역이 뚜렷하게 강조되지는 않는다. '
          '질문 대상 영역과 분포의 불일치를 추가 검토할 후보로 볼 수 있지만, 네 층 평균의 단일 query만으로 실제 시각 정보 이용을 완전히 설명할 수는 없다.', '',
          f'[두 페이지 원본·overlay·답변]({prefix}analysis/validation_Music_21/index.html)', '',
          '## 과목별 비교', '', '| 과목 | seed 3407 | seed 42 | 정답 수 차이 |', '|---|---:|---:|---:|']
for subject, info in new['summary']['by_subject'].items():
    prior = old['summary']['by_subject'][subject]['correct']
    lines.append(f'| {subject} | {prior}/30 | {info["correct"]}/30 ({info["accuracy"]*100:.2f}%) | {info["correct"]-prior:+d} |')
lines += ['', '## 실험 조건과 해석 제한', '',
          '- 모델: Qwen/Qwen3-VL-4B-Instruct, revision ebb281ec70b05090aa6165b016eac8ec08e71b17. BF16, RTX 4090.',
          '- 데이터: MMMU/MMMU validation 900문항, revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68.',
          '- sampling: seed 42, temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, 출력 상한 32,768. Judge seed 3407 고정.',
          '- 코드 브랜치 codex/visual-token-analysis. 서버 commit d549434, 로컬 구현 commit c43b705. Windows/Linux 줄바꿈 차이는 사전 소스 검증에서 정규화했다.',
          '- Python 3.12.14, vLLM 0.11.0, torch 2.8.0, Transformers 4.57.0. 기존 seed 3407 조건에서 sampling.seed만 변경한 설정임을 비교 확인했다.',
          '- Heatmap은 생성 당시 vLLM attention을 직접 기록한 것이 아니라 같은 가중치·입력·정확한 생성 토큰 경로를 Transformers SDPA에서 teacher forcing한 결과다. backend 수치적 동일성을 주장하지 않는다.',
          '- 색상 최대값은 문항 안의 모든 이미지·probe에 공통이다. 서로 다른 문항의 색 진하기를 직접 비교하지 않는다. 테두리·공백의 강한 지점은 attention sink 가능성이 있으며 의미적 중요도의 증명이 아니다.',
          '- 첫 query와 최종 답·마지막 query만 관찰했으므로 중간 풀이 중의 이미지 참조는 관찰하지 않았다. 36개 중 네 layer의 평균은 개별 head·layer의 차이를 숨길 수 있다.',
          '- 다음 분석 후보는 동일 원시 배열의 층별 비교, 풀이 중간의 query 추가, 관심 영역 가림에 따른 답 확률 변화다. 이번 점검에서는 추가 전체 평가·새 seed·가중치 변경을 하지 않았다.', '',
          '## 보존 파일', '',
          f'- [전체 검토 화면]({prefix}analysis/index.html)',
          f'- [최종 결과 JSON]({prefix}baseline_seed42/results.json)',
          f'- [파일 및 수치 검증]({prefix}verification.json)',
          f'- [전송 SHA256 검증]({prefix}transfer_verification.json)',
          f'- [입력·환경·소스 provenance]({prefix}provenance/preflight.json)',
          f'- [자동 분석 집계]({prefix}analysis/analysis_summary.json)',
          '- 원본 서버 결과 및 실행 코드 브랜치를 보존했다. GitHub push나 main 병합은 수행하지 않았다.']
(ROOT / 'reports/mmmu_visual_token_result_20261010.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

# Compact scientific comparison sheet, assembled from the actual generated images.
cases = [('Accounting_1', 'before_final_answer'), ('Electronics_6', 'before_final_answer'), ('Chemistry_12', 'before_last_token')]
canvas = Image.new('RGB', (1400, 1050), '#eef2f7')
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
draw.text((24, 14), 'Original image                               Attention overlay (selected query, four-layer mean)', fill='#172b49', font=font)
for index, (suffix, probe) in enumerate(cases):
    y = 60 + index * 325
    draw.text((24, y), f'{suffix} | {probe}', fill='#172b49', font=font)
    for column, filename in enumerate(['original_image_1.jpg', f'{probe}_image_1.jpg']):
        image = Image.open(RUN / 'analysis' / ('validation_' + suffix) / filename).convert('RGB')
        image.thumbnail((650, 270))
        canvas.paste(image, (24 + column * 700, y + 35))
draw.text((24, 1020), 'Brightness is descriptive attention, not proof of causal importance. Scales differ between questions.', fill='#172b49', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18))
canvas.save(RUN / 'representative_heatmaps.jpg', quality=94)
print(str(ROOT / 'reports/mmmu_visual_token_result_20261010.md'))
