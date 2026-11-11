"""Korean research report from audited Math_13 occlusion artifacts."""
import json
from pathlib import Path
from statistics import mean
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK / 'results/mmmu_math13_occlusion_20261014'
OUT = WORK / 'reports/math13_occlusion_report_20261014.docx'
QA = WORK / 'reports/math13_occlusion_report_qa'
QA.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
exp = read(ROOT / 'experiment.json')
summary = read(ROOT / 'analysis_summary.json')
audit = read(ROOT / 'answer_audit.json')
records = read(ROOT / 'analysis_results.json')['records']
assert summary['completed'] == 12 and len(audit['mismatches']) == 2
assert [len(x['paired_answer_changes']) for x in summary['conditions']] == [0,1,2,2]
assert read(ROOT / 'verification.json')['passed']

# Scientific montage: unchanged actual input pixels with labels outside inputs.
montage = Image.new('RGB', (1100, 1150), 'white')
draw = ImageDraw.Draw(montage)
f = ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf', 28)
labels = ['원본', '높은 attention 영역', '도형 정보 영역 D', '무작위 영역']
for i, c in enumerate(exp['conditions']):
    x, y = 24 + (i % 2) * 550, 15 + (i // 2) * 565
    draw.text((x, y), labels[i], font=f, fill='black')
    montage.paste(Image.open(ROOT / c['image']).convert('RGB'), (x, y+48))
montage.save(QA / 'actual_inputs.png')

doc = Document()
s = doc.sections[0]
s.page_width = Cm(21); s.page_height = Cm(29.7)
s.top_margin = Cm(1.7); s.bottom_margin = Cm(1.6)
s.left_margin = Cm(2); s.right_margin = Cm(2)
s.footer_distance = Cm(.65)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
    st = doc.styles[name]
    st.font.name = 'Malgun Gothic'
    st._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'맑은 고딕')
    st.font.color.rgb = RGBColor(0,0,0)
    pr = st._element.find(qn('w:pPr'))
    if pr is not None:
        for border in list(pr.findall(qn('w:pBdr'))): pr.remove(border)
normal = doc.styles['Normal']
normal.font.size = Pt(10.5)
normal.paragraph_format.line_spacing = Pt(15.5)
normal.paragraph_format.space_after = Pt(7)
for name, size in [('Title',23),('Subtitle',10),('Heading 1',15),('Heading 2',11.5),('Caption',9)]:
    st=doc.styles[name]; st.font.size=Pt(size)
    st.paragraph_format.space_after=Pt(7)
    st.paragraph_format.space_before=Pt(10 if name.startswith('Heading') else 0)
    st.paragraph_format.keep_with_next=True
doc.styles['Caption'].paragraph_format.keep_with_next=False
doc.core_properties.title='Math 13 시각 attention과 영역 가림 실험 결과 보고서'
doc.core_properties.subject='MMMU Math_13 동일 면적 마스크와 최종 선택지 변화 분석'
doc.core_properties.author='MMDL'

footer=s.footer.paragraphs[0]
footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=footer.add_run(); r.font.size=Pt(9)
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); r._r.addnext(fld)

def p(text, bold=False, style=None):
    par=doc.add_paragraph(style=style)
    par.add_run(text).bold=bold
    return par

def h(text,level=1): doc.add_heading(text,level)
def page(): doc.add_page_break()

def table(headers, rows, widths):
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    pr=t._tbl.tblPr
    borders=OxmlElement('w:tblBorders')
    for edge in ['top','left','bottom','right','insideH','insideV']:
        el=OxmlElement('w:'+edge); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
    pr.append(borders)
    margins=OxmlElement('w:tblCellMar')
    for edge,val in [('top','85'),('bottom','85'),('left','100'),('right','100')]:
        el=OxmlElement('w:'+edge); el.set(qn('w:w'),val); el.set(qn('w:type'),'dxa'); margins.append(el)
    pr.append(margins)
    for j,w in enumerate(widths): t.columns[j].width=Cm(w)
    for i,values in enumerate([headers]+rows):
        row=t.rows[0] if i==0 else t.add_row()
        rowpr=row._tr.get_or_add_trPr()
        cant=OxmlElement('w:cantSplit'); rowpr.append(cant)
        if i==0: rowpr.append(OxmlElement('w:tblHeader'))
        for j,value in enumerate(values):
            c=row.cells[j]; c.width=Cm(widths[j]); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'233C5A' if i==0 else ('F0F4F8' if i%2==0 else 'FFFFFF')); c._tc.get_or_add_tcPr().append(shade)
            cp=c.paragraphs[0]; cp.paragraph_format.space_after=Pt(0); cp.paragraph_format.line_spacing=Pt(13)
            cp.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
            cr=cp.add_run(str(value)); cr.font.size=Pt(9.3); cr.bold=i==0
            if i==0: cr.font.color.rgb=RGBColor(255,255,255)
    doc.add_paragraph().paragraph_format.space_after=Pt(1)
    return t

def pic(path, width, caption):
    par=doc.add_paragraph(); par.paragraph_format.keep_with_next=True
    par.paragraph_format.line_spacing=1.0
    par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=par.add_run(); r.add_picture(str(path),width=Cm(width))
    doc.inline_shapes[-1]._inline.docPr.set('descr',caption)
    p(caption,style='Caption')

def link(label, path):
    par=doc.add_paragraph()
    url=path if isinstance(path,str) and path.startswith('https:') else Path(path).resolve().as_uri()
    rel=doc.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True)
    hyp=OxmlElement('w:hyperlink'); hyp.set(qn('r:id'),rel)
    rr=OxmlElement('w:r'); props=OxmlElement('w:rPr')
    color=OxmlElement('w:color'); color.set(qn('w:val'),'2156A0'); props.append(color); rr.append(props)
    tx=OxmlElement('w:t'); tx.text=label; rr.append(tx); hyp.append(rr); par._p.append(hyp)
    return par

# Page 1: concise result, literal answer matrix, spatial evidence.
doc.add_paragraph('Math 13 시각 attention과\n영역 가림 실험 결과 보고서',style='Title')
p('MMMU validation Math_13  |  실험 및 보고일 2026년 10월 14일  |  MMDL',style='Subtitle')
p('동일 면적의 영역을 가린 뒤 풀이를 처음부터 다시 생성한 결과, 원본 대비 최종 답은 높은 attention 영역에서 1/3회, D 꼭짓점 영역과 무작위 영역에서 각각 2/3회 바뀌었다. 이 사례에서는 attention의 크기와 답 변화 횟수의 순서가 일치하지 않았다.')
p('원본도 세 seed 모두 오답이었다. 따라서 이번 실험으로 확인한 것은 입력 가림에 대한 선택지 변화이며, 정답을 내는 데 필수적인 시각 영역을 확인한 것은 아니다.',bold=True)
headers=['입력 조건','seed 42','seed 3407','seed 20261014','답 변화','C 일치']
rows=[]
for label,c in zip(['원본','높은 attention','D 꼭짓점','무작위 영역'],summary['conditions']):
    rows.append([label,*[c['answers'][str(seed)] for seed in exp['seeds']],f"{len(c['paired_answer_changes'])}/3",f"{c['correct']}/3"])
table(headers,rows,[3.5,2.2,2.2,3.0,2.05,2.05])
p('표 1  원문 마지막 문장에서 확인한 선택지. 답 변화는 같은 seed의 새 원본과 비교했다. 원래 정답은 C이며, C 일치는 세 번의 생성 중 C를 선택한 횟수다.',style='Caption')
pic(ROOT/'region_effects.png',16.7,'그림 1  실험한 세 영역과 답 변화. 색 테두리는 위치 표시이며 모델에 입력한 마스크의 테두리가 아니다. 미실험 영역의 중요도는 측정하지 않았다.')

# Page 2: reproducible protocol with exact scope and attention denominator.
page(); h('실험 설계와 측정 기준')
h('질문과 비교 목적',2)
p('Math_13은 변의 길이가 같은 정육각형과 정팔각형이 겹친 그림에서 사각형 ABCD의 넓이를 고르는 문제다. 원본에서는 A와 B가 바깥 팔각형의 위쪽 꼭짓점, C와 D가 안쪽 육각형의 위쪽 꼭짓점이다. 기존 seed 42 응답은 B를 선택했다.')
p('실험 질문은 “기존 attention이 높은 곳을 가릴 때, 도형 정보를 가리거나 임의 위치를 가릴 때보다 최종 선택지가 더 자주 바뀌는가”이다. 각 조건에는 같은 질문만 제공했으며 과거 풀이를 붙이지 않았다.')
table(['통제 항목','설정'],[
    ['모델','Qwen3-VL-4B-Instruct  BF16'],
    ['실행','vLLM 0.11.0  RTX 4090  단일 요청 순차 생성'],
    ['생성 seed','42  3407  20261014'],
    ['샘플링','temperature 0.7  top_p 0.8  top_k 20'],
    ['반복 관련 설정','repetition_penalty 1.0  presence_penalty 1.5'],
    ['출력 및 이미지','출력 상한 32,768토큰  원본 502 × 491픽셀'],
    ['가림 방식','96 × 96픽셀  RGB 127 127 127  전체 면적의 3.739%'],
    ['입력 보존','마스크 밖 RGB 픽셀 불변  무손실 PNG  기존 전처리 해상도 유지']
],[4.2,12.8])
table(['마스크','좌표 x0 y0 x1 y1','선택 근거','이미지 attention 비중'],[
    ['높은 attention','406 7 502 103','최대 attention 토큰 중심과 이미지 경계', '74.976%'],
    ['D 꼭짓점','112 128 208 224','D 라벨과 꼭짓점 및 인접 변 수동 선택','0.252%'],
    ['무작위 영역','372 279 468 375','위치 seed 20261014로 한 번 추출','0.028%'],
],[3.1,4.0,6.5,3.4])
p('좌표는 원본 기준이며 x1과 y1은 포함하지 않는다. 무작위 위치는 결과를 보고 재추출하지 않았으며, 오른쪽 아래 꼭짓점과 변 일부를 포함한다. 배경만 남긴 대조군은 아니다.',style='Caption')
h('attention과 답 변화의 정의',2)
p('기존 attention은 vLLM 생성 당시 직접 저장한 값이 아니라, 정확한 생성 토큰 경로를 Transformers로 재생한 값이다. 최종 답 첫 토큰 직전 query에서 decoder layer 8, 17, 26, 35의 head 평균을 구한 뒤 네 layer를 평균했다. 이 순간 전체 key에 대한 attention 중 이미지 비중은 약 1.43%다.')
p('위 표의 비중은 그 이미지 attention 안에서 각 사각형이 포함하는 비중이다. 전체 attention의 74.976%라는 뜻이 아니다. 이번 가림 실험은 입력을 바꿔 전체 풀이를 새로 생성하므로, 한 시점의 attention과 측정 범위가 다르다.')

# Page 3: actual input evidence, no inferred heatmap.
page(); h('실제 입력 이미지와 seed별 변화')
p('모든 마스크는 같은 크기와 같은 회색을 사용했다. 아래 그림은 실제 모델에 입력한 네 PNG를 나란히 배치한 것이다. 원본은 데이터의 RGB hash가 기존 실행과 같은지 검증했다.')
pic(QA/'actual_inputs.png',16.7,'그림 2  네 조건의 실제 입력 이미지. 오른쪽 위는 주로 흰 배경, D 조건은 라벨과 인접 변, 무작위 조건은 오른쪽 아래 도형 정보를 가린다.')
p('seed 42에서는 무작위 영역에서만 B→E로 바뀌었다. seed 3407에서는 D 영역에서만 B→D로 바뀌었다. seed 20261014에서는 세 마스크 모두 원본 A에서 B로 바뀌었다. 영역 효과의 패턴이 seed에 따라 달랐다.')
p('같은 선택지가 유지됐다고 풀이가 같다는 뜻은 아니다. 예를 들어 seed 42의 원본은 10,049토큰, 높은 attention 가림 응답은 10,538토큰이었지만 둘 다 B로 끝났다.')

# Page 4: separate observational behavior, grading audit and confounding.
page(); h('응답 내용과 채점값 대조')
h('모델 응답에서 관찰한 내용',2)
p('높은 attention 영역을 가린 seed 42 응답은 ABCD의 네 점을 모두 육각형 꼭짓점이라고 서술했다. 실제 그림에서 A와 B는 바깥 팔각형에 있다. 밝게 표시된 위치에 attention이 집중됐다는 사실만으로 도형 관계를 올바르게 이해했다고 판단할 수 없는 사례다.')
p('D를 가린 seed 42 응답은 “Perhaps the quadrilateral ABCD is a square, and the gray square is inside it.”라고 서술했다. 회색 마스크를 지워진 정보의 표시로만 취급하지 않고, 새로운 도형일 가능성도 제기했다. 따라서 결과에는 정보 제거 효과와 새 패턴의 효과가 함께 포함될 수 있다.')
p('이 두 관찰은 실제 이미지와 생성된 발화를 수동 검토한 결과다. 생성 설명이 내부 계산 경로를 충실하게 보고한다는 증거로 해석하지 않는다.')
h('원문 마지막 답과 기존 채점값의 불일치',2)
table(['조건','원문 마지막 문장','기존 채점값','경로'],[
    ['D 가림  seed 42','So Final answer: B','E','local judge'],
    ['D 가림  seed 3407','So Final answer: D','E','fallback parser']
],[3.6,6.4,2.5,4.5])
p('표 2  채점 도구의 답을 모델이 실제로 선택한 답으로 오인할 수 있었던 두 사례.',style='Caption')
p('기존 target parser는 “So ”가 앞에 붙은 최종 답 필드를 인식하지 않았다. seed 42는 judge로 넘어갔고, judge 첫 응답의 “Final Answer: **B**”도 judge parser가 인식하지 못해 재시도됐다. 두 번째 judge 응답은 E였다. seed 3407은 fallback parser가 E를 기록했다.')
p('가림 효과 분석에서는 12개 원문의 마지막 문장을 모두 대조해 B와 D를 사용했다. 기존 results.json과 summary.json은 보존하고, 원문 기준 결과는 analysis_results.json과 analysis_summary.json에 별도로 기록했다. 두 답 모두 정답 C가 아니므로 정오답 합계는 바뀌지 않았다.')
p('이 대조를 생략하면 D 영역의 답 변화가 실제 2/3회가 아니라 3/3회로 집계된다. 정답률이 같아도 선택지 전환 분석에는 채점기의 추출 오류가 영향을 줄 수 있다.',bold=True)
h('완료와 재현성 확인',2)
p(f"12회 생성과 채점이 완료됐고 출력 제한 종료는 0회였다. 생성 소요 시간 합계는 {summary['generation_seconds']:.1f}초, 약 {summary['generation_seconds']/60:.1f}분이다. 이 값에는 모델 적재, judge, 전송, 문서 작성 시간이 포함되지 않는다.")
p('새 원본 seed 42의 출력 토큰 ID 10,049개는 과거 실행과 전부 일치했다. 모든 조건의 prompt token IDs와 image grid가 기존 입력과 같았으며, 정확한 출력 토큰 ID와 원시 응답을 보존했다. 전송한 원시 아카이브와 최종 이미지의 서버·로컬 hash 일치도 확인했다.')

# Page 5: bounded interpretation, concrete next experiment, provenance links.
page(); h('해석의 범위와 후속 실험 제안')
h('현재 결과로 말할 수 있는 범위',2)
p('이 문항의 세 seed에서는 높은 attention 영역보다 D 영역과 무작위 영역에서 최종 선택지가 더 자주 바뀌었다. 그러나 비교 대상은 한 문항, 세 seed, 한 무작위 위치뿐이다. 통계적 유의성이나 일반적인 영역 중요도 순위를 주장하지 않는다.')
p('원본도 0/3회 정답이므로 정답을 유지하는 데 필요한 영역을 평가할 성공 사례가 없다. 또한 같은 면적을 가려도 제거되는 의미 정보의 양은 같지 않다. D 정보는 다른 꼭짓점과 대칭으로 일부 복원될 수 있고, 무작위 영역에도 실제 도형 정보가 포함된다.')
p('따라서 본 결과는 세 개의 구체적인 가림에 대한 응답 민감도 기록으로 활용할 수 있다. 모든 픽셀의 인과적 중요도 지도나 모델의 시선 추적으로 해석하지 않는다.')
h('다음 실험의 우선순위',2)
p('첫째, 같은 위치를 흰색 채움과 흐림 처리로도 비교한다. 회색이 새 도형으로 인식되는 효과를 점검하기 위한 실험이다. 오른쪽 위는 흰 배경이므로 흰색 처리 전후 RGB가 같다면 독립적인 정보 삭제 실험이 아니라 입력 불변 확인으로 구분해야 한다.')
p('둘째, 원본에서 정답을 안정적으로 맞히는 문항을 포함하고, 여러 무작위 위치와 seed를 사전에 정한다. 같은 면적의 배경 영역과 도형 영역을 나누어 비교하면 위치와 의미 정보의 영향을 더 구체적으로 검토할 수 있다.')
p('셋째, 본 실험에서 발견한 최종 답 형식을 parser 회귀 사례로 추가하고 별도 검증한 뒤 후속 평가에 적용한다. 과거 baseline 결과를 수정할 경우에는 원래 채점과 수정 채점의 차이를 별도 버전으로 보고해야 한다.')
p('다수의 마스크와 모델 출력을 이용해 중요도를 추정하는 후속 방법으로 RISE를 참고할 수 있다. RISE는 임의로 가린 이미지에 대한 출력을 이용하는 방법이며, 본 실험은 RISE를 구현하거나 픽셀별 중요도 지도를 산출한 것이 아니다. [외부 참고 1]')
h('실행 및 근거 자료',2)
p('브랜치 codex/visual-token-analysis  |  서버 aix408-40\n실험 코드 보존 commit  로컬 635dc7d  서버 bd5e8f4',style=None)
p('모델 revision ebb281ec70b05090aa6165b016eac8ec08e71b17\n데이터 revision 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68')
p('결과 폴더  C:/project/MMDL/results/mmmu_math13_occlusion_20261014\n아래 링크는 이 컴퓨터의 보존된 실험 자료를 연다. 다른 컴퓨터에서 열려면 결과 폴더도 함께 전달해야 한다.')
link('실험 명세와 입력 hash',ROOT/'experiment.json')
link('원문 답 기준 집계와 seed별 결과',ROOT/'analysis_summary.json')
link('12개 최종 답의 대조 기록',ROOT/'answer_audit.json')
link('검토 화면과 생성된 전체 풀이',ROOT/'index.html')
link('외부 참고 1  Petsiuk 등 2018  RISE', 'https://arxiv.org/abs/1806.07421')

doc.save(OUT)
metrics={'output':str(OUT),'planned_pages':5,'generations':12,'answer_change_counts':[0,1,2,2],
    'all_correct_counts':[0,0,0,0],'answer_audit_mismatches':2,'generation_seconds':summary['generation_seconds'],
    'figures':2,'tables':4,'source_summary':str(ROOT/'analysis_summary.json')}
(QA/'content_validation.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metrics,ensure_ascii=True))
