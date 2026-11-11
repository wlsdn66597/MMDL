"""Build the Korean baseline report from verified records without new inference."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK / 'results/mmmu_multiseed_20261011'
OUT = WORK / 'reports/mmmu_5seed_final_report_20261014.docx'
QA = WORK / 'reports/mmmu_5seed_report_qa'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
a = read(ROOT / 'aggregate.json')
audit = read(ROOT / 'final_verification.json')
assert a['all_five_complete'] and audit['passed']
items = read(ROOT / 'item_stability.json')
runs = {x['seed']: read(Path(x['source']) / 'results.json')['records'] for x in a['runs']}
examples = read(ROOT / 'illustrative_items.json')
hist = Counter(x['correct_seed_count'] for x in items)
methods = defaultdict(lambda: {'count': 0, 'correct': 0})
times = {}
for seed, records in runs.items():
    total = sum(x['model_response']['elapsed_seconds'] for x in records)
    limited = sum(x['model_response']['elapsed_seconds'] for x in records if x['model_response']['finish_reason'] == 'length')
    times[seed] = {'hours': total / 3600, 'length_hours': limited / 3600, 'length_time_percent': limited / total * 100}
    for row in records:
        final = row['grading']['final']
        methods[final['method']]['count'] += 1
        methods[final['method']]['correct'] += final['hit']
all_hours = sum(t['hours'] for t in times.values())
length_hours = sum(t['length_hours'] for t in times.values())
assert sum(hist.values()) == 900 and sum(k*v for k,v in hist.items()) == sum(x['correct'] for x in a['runs'])
QA.mkdir(exist_ok=True)
write_data = {'hit_histogram': dict(sorted(hist.items())), 'methods': dict(methods), 'times': times,
              'total_generation_hours': all_hours, 'length_generation_hours': length_hours}
(QA / 'report_metrics.json').write_text(json.dumps(write_data, ensure_ascii=False, indent=2), encoding='utf-8')

doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21); sec.page_height = Cm(29.7)
sec.top_margin = Cm(1.8); sec.bottom_margin = Cm(1.7)
sec.left_margin = Cm(2.0); sec.right_margin = Cm(2.0)
sec.footer_distance = Cm(.8)
for name in ['Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption']:
    style = doc.styles[name]
    style.font.name = 'Malgun Gothic'
    style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), '맑은 고딕')
    style.font.color.rgb = RGBColor(0,0,0)
normal = doc.styles['Normal']
normal.font.size = Pt(10.5)
normal.paragraph_format.line_spacing = Pt(14.5)
normal.paragraph_format.space_after = Pt(7)
for name, size in [('Title',22),('Heading 1',16),('Heading 2',12),('Subtitle',11)]:
    doc.styles[name].font.size = Pt(size)
    doc.styles[name].paragraph_format.space_after = Pt(10)
    doc.styles[name].paragraph_format.space_before = Pt(10 if name != 'Title' else 0)
for style in doc.styles:
    prop = style._element.find(qn('w:pPr'))
    if prop is not None:
        for border in list(prop.findall(qn('w:pBdr'))): prop.remove(border)
    if style.type == 1:
        prop = style._element.get_or_add_pPr()
        grid = OxmlElement('w:snapToGrid'); grid.set(qn('w:val'), '0'); prop.append(grid)
for grid in list(sec._sectPr.findall(qn('w:docGrid'))): sec._sectPr.remove(grid)
doc.styles['Title'].paragraph_format.line_spacing = Pt(28)
doc.styles['Heading 1'].paragraph_format.line_spacing = Pt(21)
doc.styles['Heading 2'].paragraph_format.line_spacing = Pt(17)
doc.styles['Caption'].font.size = Pt(9)
doc.styles['Caption'].font.italic = False
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)
doc.core_properties.title = 'MMMU 5개 추론 seed 기준 모델 평가 보고서'
doc.core_properties.subject = 'Qwen3 VL 4B Instruct 고정 환경 반복 평가'
doc.core_properties.author = ''


def p(text, bold=False, small=False):
    para = doc.add_paragraph()
    run = para.add_run(text); run.bold = bold
    if small: run.font.size = Pt(9)
    return para


def h(text, level=1):
    return doc.add_heading(text, level)


def page(title):
    doc.add_page_break(); h(title)


def table(headers, rows, widths, size=9.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for c,w in zip(t.columns,widths): c.width = Cm(w)
    pr = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        el=OxmlElement('w:'+edge); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
    pr.append(borders)
    for i,row in enumerate([headers]+rows):
        cells = t.rows[0].cells if i == 0 else t.add_row().cells
        trpr = cells[0]._tc.getparent().get_or_add_trPr()
        no_split=OxmlElement('w:cantSplit');trpr.append(no_split)
        if i == 0: trpr.append(OxmlElement('w:tblHeader'))
        for j,(cell,text,width) in enumerate(zip(cells,row,widths)):
            cell.width=Cm(width);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=cell._tc.get_or_add_tcPr(); margins=OxmlElement('w:tcMar')
            for edge,val in [('top',65),('bottom',65),('left',85),('right',85)]:
                el=OxmlElement('w:'+edge);el.set(qn('w:w'),str(val));el.set(qn('w:type'),'dxa');margins.append(el)
            cp.append(margins)
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'DCE6EF' if i == 0 else ('F6F8FA' if i%2==0 else 'FFFFFF'));cp.append(shade)
            para=cell.paragraphs[0];para.paragraph_format.space_after=Pt(0);para.paragraph_format.line_spacing=Pt(size+3)
            para.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
            run=para.add_run(str(text));run.font.size=Pt(size);run.bold=i==0
    spacer=doc.add_paragraph();spacer.paragraph_format.space_after=Pt(3);spacer.paragraph_format.line_spacing=Pt(2)
    spacer.add_run().font.size=Pt(1)
    return t


doc.add_paragraph('MMMU 5개 추론 seed\n기준 모델 평가 보고서', 'Title')
p('Qwen/Qwen3-VL-4B-Instruct  |  작성일 2026년 10월 14일', small=True)
h('1 평가 결과 요약')
p('MMMU validation의 고정 900문항을 동일한 Qwen3-VL-4B-Instruct 체크포인트로 5회 평가했다. 추론 seed만 바꾼 본평가의 평균 정확도는 62.87%, 표본 표준편차는 1.27%p였다. seed 3407을 독립적으로 한 번 더 실행한 재현성 검사에서는 900문항 응답·채점 기록과 Judge 시도 기록이 일치했다. 이 결과를 향후 개선 체크포인트와 비교할 기준 성능으로 사용한다.')
table(['추론 seed','정답 수','평가 문항 수','정확도'], [[x['seed'],x['correct'],900,f"{x['accuracy_percent']:.2f}%"] for x in a['runs']], [4,4,4,5],10)
p('평균 정확도 62.87%   표본 표준편차 1.27%p   범위 61.22%~64.33%',bold=True)
p('최고 점수와 최저 점수의 차이는 3.11%p로 900문항 기준 28개 정답 차이에 해당한다. 하나의 seed에서 얻은 최고 점수만 제시하면 이번 실행에서 관찰된 변동을 드러내지 못한다. 따라서 기준 성능은 5회 평균과 표본 표준편차를 함께 보고한다.')
h('평가 범위',2)
p('5회 모두 정답인 문항은 442개(49.11%), 모두 오답인 문항은 203개(22.56%), seed에 따라 정오답이 달라진 문항은 255개(28.33%)였다. 전체 평균의 변동보다 개별 문항의 응답 변동이 더 넓게 나타나므로 후속 비교에서는 점수와 문항별 변화 모두를 확인해야 한다.')
p('본 보고서는 하나의 고정 체크포인트에 대한 추론 seed 실험이다. 학습 seed를 바꾼 반복 학습이나 개선 모델과의 성능 비교는 포함하지 않는다. 과거 64.00% 결과와 재현성 검사용 반복 실행은 평균에서 제외했다. [1, 2]')
h('완료와 검증',2)
p('전체 캠페인은 2026년 10월 13일 21:53 KST에 완료됐다. 원본 다운로드 후 파일 해시, 고유 문항 수, 과목별 문항 수, 응답·채점·집계 정합성을 확인했다. 이 검증은 채점 결과의 내용적 타당성을 전 문항 수동 검수했다는 뜻은 아니다. [3]')

page('2 실험 설계와 채점 방법')
p('실험 단위는 고정된 900문항 전체에 대한 한 번의 추론과 채점이다. 과목별 30문항씩 30과목을 사용했으므로 과목 정확도의 단순 평균과 전체 정답 비율이 같다. 본평가 seed는 결과를 집계하기 전에 정한 3407, 42, 123, 2026, 9876이다. [2, 4]')
table(['설정 항목','실제 적용값'],[
['평가 모델','Qwen/Qwen3-VL-4B-Instruct'],['데이터','MMMU/MMMU validation · 30과목 × 30문항'],
['정밀도와 장비','BF16 · 양자화 없음 · RTX 4090 24 GB'],['환경','Python 3.12.14 · vLLM 0.11.0 · PyTorch 2.8.0'],
['샘플링','temperature 0.7 · top_p 0.8 · top_k 20'],['반복 및 존재 페널티','repetition_penalty 1.0 · presence_penalty 1.5'],
['생성 및 문맥 상한','32,768토큰 · 64,000토큰'],['이미지 설정','최대 5장 · pixels 1,003,520~4,014,080'],
['동시 처리','max_num_seqs 1 · max_num_batched_tokens 4,096'],['실행 방식','enforce_eager true · tensor_parallel_size 1'],
['재현성 환경변수','VLLM_ENABLE_V1_MULTIPROCESSING=0'],['배치 불변성','VLLM_BATCH_INVARIANT 미설정'],
['Judge','Qwen/Qwen3-8B · temperature 0 · seed 3407'],['Judge 상한','출력 4,096토큰 · 최대 25회 시도 · thinking false']],[5,12],9.2)
p('각 본평가 seed는 평가 모델의 LLM 엔진과 모든 SamplingParams 요청에 함께 적용했다. Judge의 엔진·요청 seed는 모든 실행에서 3407로 고정했다. seed 42의 evaluate.py에는 토큰 기록을 위한 차이가 있으며, 추론 설정 변경이 없는 허용 예외로 실행 전 검증 기록에 보존했다. [4]')
h('채점 순서와 집계 기준',2)
p('출력 길이 한도에 도달한 응답은 먼저 오답 처리한다. 나머지는 마지막 Final answer 필드를 우선 추출하고, 필요한 경우 공식 파서의 fallback 규칙을 적용한다. 그래도 해결되지 않은 문항만 로컬 Judge에 전달한다. Judge의 Z는 즉시 오답이며, 재시도 후에도 무효인 출력 역시 오답이다. 무작위 답 선택은 사용하지 않았다. [2]')
p('평균은 seed별 정확도 5개의 산술평균이다. 표본 표준편차는 각 정확도와 평균 간 편차의 제곱합을 4로 나눈 뒤 제곱근을 취했다(ddof=1). 보고서의 ±1.27%p는 표준편차이며 신뢰구간이나 표준오차가 아니다.')

page('3 문항별 안정성과 seed 간 변화')
p('다섯 seed는 같은 문제를 반복한다. 따라서 정답 수가 비슷하더라도 맞힌 문항이 같다는 뜻은 아니다. 900개 고유 ID를 기준으로 seed별 최종 정오답을 연결해 정답 횟수와 전환을 계산했다. [5]')
table(['5회 중 정답 횟수','문항 수','900문항 대비 비율'],[[n,hist[n],f'{hist[n]/9:.2f}%'] for n in range(6)],[6,5,6])
h('seed 3407을 기준으로 한 정오답 전환',2)
rows=[]
for x in a['pairwise_seed_transitions']:
    if x['seed_a']==3407:
        tr=x['transitions'];rows.append([x['seed_b'],tr['0_to_1'],tr['1_to_0'],tr['0_to_1']+tr['1_to_0'],f"{x['delta_pp']:+.2f}"])
table(['비교 seed','오답에서 정답','정답에서 오답','달라진 문항','정확도 차이 %p'],rows,[2.5,3.6,3.6,3.6,3.7],9)
p('3407과 123의 전체 점수 차이는 4문항(0.44%p)에 불과하지만, 64개 문항이 오답에서 정답으로, 60개가 정답에서 오답으로 바뀌어 총 124문항의 판정이 달랐다. 점수 차이만으로 응답 안정성을 판단하기 어렵다는 사례다. 서로 다른 seed 사이의 전환을 학습에 따른 개선이나 퇴행으로 해석하지 않는다.')
h('과목별 결과 해석',2)
subjects=sorted(a['subjects'],key=lambda x:x['mean_percent'])
p('과목별 평균은 Music 24.00%, Diagnostics and Laboratory Medicine 30.00%, Mechanical Engineering 39.33%에서 낮았고, Economics와 Marketing은 각각 84.00%, Literature는 83.33%였다. 과목별 표본 표준편차는 Chemistry 7.45%p, Clinical Medicine 7.30%p, Computer Science 7.23%p로 상대적으로 컸다. 전체 30과목 결과는 부록에 제시했다.')
lit=sum(0<x['correct_seed_count']<5 for x in items if x['subject']=='Literature')
p(f'Literature는 모든 seed의 점수가 25/30으로 같고 표준편차가 0이다. 문항별로 확인한 정오답 변동 문항은 {lit}개다. 과목별 분모가 30문항으로 작아 한 문항 차이가 3.33%p에 해당하므로, 과목 순위만으로 일반적인 분야 역량을 단정하지 않는다.')

page('4 저장된 응답의 사례 비교')
p('잘림이 없는 문항 중 5회 정답 횟수가 2회, 3회, 0회인 사례를 각각 하나씩 제시한다. 사례는 해당 조건을 만족하는 ID의 사전식 첫 항목으로 고정해 선정했다. 저장된 추출 답과 채점 기록의 차이를 보여주는 예시이며 전체 오류의 대표 표본이나 정답의 수동 인증 결과는 아니다. [6]')
labels={'validation_Architecture_and_Engineering_23':'건축 및 공학 23번','validation_Accounting_1':'회계 1번','validation_Accounting_15':'회계 15번'}
for i,e in enumerate(examples,1):
    h(f'사례 {i} {labels[e["id"]]}',2)
    p(f"문항 ID {e['id']}\n데이터셋 기준 답 {e['reference_answer']} · 정답 {e['correct_seed_count']}/5회",small=True)
    table(['seed','3407','42','123','2026','9876'],[['추출 답']+[e['by_seed'][str(s)]['answer'] for s in a['planned_seeds']],['정오답']+['정답' if e['by_seed'][str(s)]['hit'] else '오답' for s in a['planned_seeds']]],[3,2.8,2.8,2.8,2.8,2.8])
    if e['correct_seed_count']:
        p('출력 한도에 도달하지 않은 응답에서도 seed에 따라 최종 선택이 달랐다. 변동의 원인을 이미지 인식, 계산, 추론 과정, 답 추출 중 어느 단계로 귀속할지는 원본 이미지와 전체 응답을 함께 검토해야 한다.')
    else:
        p('모든 seed에서 오답으로 채점됐다. 반복 오답이라는 사실은 확인되지만, 공통된 원인을 확정하지는 못한다. 후속 오류 분석에서는 문제 조건과 이미지, 응답의 중간 계산 및 최종 답을 대조할 필요가 있다.')
p('사례의 질문·선택지·기준 답·seed별 추출 답·응답 끝부분은 illustrative_items.json에, 전체 응답은 각 seed의 results.json에 보존돼 있다.',small=True)

page('5 출력 한도와 추론 비용')
p('5회 본평가의 4,500개 응답 중 399개(8.87%)가 32,768토큰 출력 한도에 도달해 오답으로 처리됐다. 이 수치는 399개의 서로 다른 문제를 뜻하지 않는다. 잘림은 같은 문항에서 여러 seed에 걸쳐 발생할 수 있다. [1, 7]')
table(['Seed','잘림 문항','Judge 문항','평균 출력 토큰','생성 시간 h','잘림 시간 비율'],[[x['seed'],x['length_count'],x['judge_count'],f"{x['mean_output_tokens']:,.1f}",f"{x['generation_hours']:.2f}",f"{times[x['seed']]['length_time_percent']:.1f}%"] for x in a['runs']],[2,2.4,2.5,3.5,3,3.6],9)
p(f'문항별 생성 시간 합계는 본평가 5회 전체에서 {all_hours:.2f}시간이다. 이 중 잘림 응답에 사용한 시간은 {length_hours:.2f}시간으로 {length_hours/all_hours*100:.1f}%를 차지했다. 잘림은 정확도 손실뿐 아니라 계산 비용 측면에서도 점검할 가치가 있다. 다만 이번 결과만으로 출력 상한을 바꾸었을 때의 점수나 비용 변화를 예측할 수는 없다.')
p('seed별 출력 토큰 중앙값은 277~292.5개인 반면 평균은 약 3,900~4,320개였고, 95백분위수는 모두 32,768개였다. 짧은 응답이 많은 가운데 일부 매우 긴 응답이 평균과 실행 시간을 크게 높였다.')
h('최종 채점 경로',2)
names={'explicit_final_parser':'명시적 최종 답 파서','official_parser_fallback':'공식 파서 fallback','local_qwen3_8b_judge_parser':'로컬 Qwen3 8B Judge','output_limit_incorrect':'출력 한도 강제 오답'}
table(['채점 경로','응답 수','정답 수','전체 응답 비율'],[[names[k],v['count'],v['correct'],f"{v['count']/45:.2f}%"] for k,v in methods.items()],[7,3,3,4],9.5)
p('Judge 대상은 합계 265개 응답(5.89%)이며 시도 횟수는 340회다. 대상 문항 수와 재시도 횟수는 다르다. 채점 경로별 정답률은 난이도가 다른 응답 집합의 결과이므로 파서와 Judge의 성능 비교로 해석하지 않는다.')
p('이번 추가 캠페인인 3407 반복 검사와 123·2026·9876의 4회 실행은 벽시계 기준 54.67시간이었다. 표의 생성 시간은 각 응답의 elapsed_seconds 합계로, 모델 로딩·Judge·압축 시간을 포함하지 않는다. 실제 평가의 중단·재개 기록은 없으며 resume_tests.log의 중단 출력은 별도 기능 테스트 기록이다.')

page('6 재현성 검증과 후속 평가')
p('seed 3407의 기존 결과와 새 프로세스·새 출력 폴더에서 생성한 결과를 비교했다. 같은 seed를 반복한 검사는 서로 다른 seed의 샘플링 변동을 측정하는 본평가와 목적이 다르다. [3, 8]')
table(['검증 항목','확인 결과'],[['문항 기록','900개 비교 · 차이 0건'],['Judge 시도 기록','78건 대 78건 일치'],['모델 식별 정보','일치'],['설정 및 코드 해시','일치'],['비교 제외 필드','elapsed_seconds · created_at · timestamp'],['다운로드 및 집계','원본 SHA256과 로컬 집계 검증 통과']],[6,11],9.5)
p('응답 원문, 토큰 수, 종료 사유와 최종 채점 기록의 일치를 확인했다. 저장되지 않은 내부 연산값이나 모든 출력 token ID까지 비교했다고 주장하지 않는다. 검증 범위는 현재 GPU·패키지 버전·실행 설정이며, 배치 크기 변경이나 다른 하드웨어·버전의 결과 동일성을 보장하지 않는다.')
h('현재 결과의 적용 범위',2)
p('VLLM_BATCH_INVARIANT는 이번 실험에 적용하지 않았다. 따라서 이 보고서의 성능과 재현성 결과는 해당 옵션이 없는 기존 환경의 결과로 명시한다. 배치 불변성을 최종 조건으로 채택한다면 별도 지원 환경에서 다중 이미지·긴 문맥·배치 구성 변경을 검증하고, baseline과 개선 모델을 동일한 새 조건에서 평가해야 한다. 현재 결과에 옵션 적용 사실을 소급해 붙이지 않는다.')
p('현재 성능은 프로젝트에서 고정한 프롬프트와 채점 규칙에 따른 baseline이며 공식 모델 리포트 점수의 완전 재현으로 제시하지 않는다. 같은 900문항의 반복을 4,500개 독립 표본으로 취급하지 않고, 5개 추론 seed만으로 학습 안정성이나 새로운 문제에 대한 일반화 성능을 입증하지 않는다.')
h('후속 비교 권고',2)
p('기존 환경을 유지한다면 개선 체크포인트 하나를 같은 5개 seed로 평가하고, 개선 모델에서도 seed 3407 독립 반복 검사를 수행한다. 각 seed에서 개선 모델 정확도와 baseline 정확도의 차이를 구한 뒤 5개 차이의 평균과 표준편차를 보고한다. 문항별 정오답 전환, 잘림 빈도, 생성 비용도 함께 비교한다. 최고 점수 seed만 선택하지 않는다.')
h('근거 자료',2)
p('[1] aggregate.json 및 각 seed의 summary.json\n[2] reports/mmmu_evaluation_protocol.md\n[3] final_verification.json 및 transfer_verification.json\n[4] protocol/config.seed*.json 및 provenance/preflight.json\n[5] item_stability.json\n[6] illustrative_items.json\n[7] 각 seed의 predictions.jsonl 및 results.json\n[8] repeatability.json 및 repeatability.local.json',small=True)
p('결과 기준 폴더 C:/project/MMDL/results/mmmu_multiseed_20261011/\n전체 결과 묶음 C:/project/MMDL/results/mmmu_all_5seeds_complete_20261013.zip',small=True)

page('부록 30과목의 seed별 정확도')
p('각 과목은 seed마다 30문항이다. 정확도와 평균의 단위는 %, 표본 표준편차의 단위는 %p이며 ddof=1을 사용했다. 계산은 반올림 전 값으로 수행했다.',small=True)
ko=['회계','농업','건축 및 공학','미술','미술 이론','기초 의학','생물학','화학','임상 의학','컴퓨터 과학','디자인','진단 및 검사 의학','경제학','전자공학','에너지 및 전력','금융','지리','역사','문학','경영','마케팅','재료','수학','기계공학','음악','약학','물리학','심리학','공중보건','사회학']
table(['과목','3407','42','123','2026','9876','평균','표준편차'],[[label]+[f"{s['accuracy_by_seed'][str(seed)]:.2f}" for seed in a['planned_seeds']]+[f"{s['mean_percent']:.2f}",f"{s['sample_sd_pp']:.2f}"] for label,s in zip(ko,a['subjects'])],[4,1.8,1.8,1.8,1.8,1.8,2,2],8.5)
p('과목명은 원본 dataset_config의 순서에 맞춰 한국어로 표기했다. 원본 영문 식별자와 문항별 정오답은 aggregate.json과 item_stability.json에서 확인할 수 있다.',small=True)
OUT.parent.mkdir(exist_ok=True)
doc.save(OUT)
print(json.dumps({'output':str(OUT),'planned_pages':7,'methods':dict(methods),'histogram':dict(hist),'hours':all_hours,'length_hours':length_hours},ensure_ascii=True))
