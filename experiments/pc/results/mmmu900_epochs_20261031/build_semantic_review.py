"""Materialize assistant qualitative annotations, never alter official grading."""
from pathlib import Path
import json,re,collections
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'evidence/pilot/prespecified30_responses.json').read_text(encoding='utf8'))
models=['baseline','epoch_2','epoch_3','epoch_4']
# Each entry: E/R/A score, component rationale. None means not established.
data=[
('Accounting',[
((2,2,2),'E: 만기·행사가·현물가가 같고 가격은 10/12임을 읽음. R: 다른 조건이 같은 콜의 가격과 변동성 관계를 사용. A: B를 일관되게 선택.'),
((2,2,2),'E: 표의 공통 조건과 10/12를 정확히 추출. R: 문제에서 같은 무배당·금리 조건을 전제로 가격/변동성 관계가 적합. A: 12>10으로 B 도출.'),
((2,2,2),'E: 동일 조건 및 가격 차이 정확. R: 동일 조건의 콜 가격 비교 적합. A: B 도출 정확.'),
((2,2,2),'E: 동일 조건 및 가격 차이 정확. R: 동일 조건의 콜 가격 비교 적합. A: B 도출 정확.')]),
('Agriculture',[
((0,0,0),'E: 잎의 면과 병반을 잘못 해석. R: 흰 생장을 흰가루병으로 단정하는 감별이 부족하며 역병에도 흰 생장이 가능. A: 정답 E 대신 D.'),
((0,0,0),'E: 잎 뒷면을 upper surface로 기술. R: 흰 가루 모양만으로 흰가루병을 확정할 수 없음. A: 갈색 병반과 역병 가능성을 배제하고 D.'),
((0,0,0),'E: 잎의 면과 병변을 잘못 연결. R: 흰 생장만으로 흰가루병에 특이적이라고 판단. A: 역병 E 대신 D.'),
((0,0,0),'E: 잎 뒷면/병변 해석 오류. R: 역병에 흰 생장이 없다는 대조는 잘못. A: 잘못된 감별로 D.')]),
('Architecture_and_Engineering',[
((1,0,0),'E: 일부 수치와 축척은 맞지만 장문에서 지상 범위 해석이 흔들림. R: 중복도와 기선 관계를 잘못 적용하는 구간 확인. A: 길이 제한으로 정상 최종 풀이 없음; 발췌에서 핵심 오류 확인.'),
((0,0,0),'E: 관측 대신 잘못된 축척 정의를 기재. R: 높이/사진폭 및 높이×중복도×2는 축척·기선 공식이 아님. A: 최종 기선 625는 맞지만 축척 산술과 지표 고도 무시는 끝까지 잘못되어 정답을 지지하지 못함.'),
((2,0,0),'E: 높이·고도·중복도 등 주요 입력은 추출. R: 비행 높이와 지표 고도를 더하고 사진폭으로 나누는 잘못된 축척. A: 반복/길이 종료 및 잘못된 계산 확인.'),
((2,0,0),'E: 입력 높이 1500, 지표 250, 크기 250 및 중복도 추출. R: 축척은 f/(H-h)인데 (H+h)/사진폭으로 제시. A: 1750/250에서 1cm=62.5m가 나오지 않으며 수정문도 625를 유도하지 못함.')]),
('Art',[
((0,0,0),'E/R/A: Final answer: B만 있어 근거·규칙·적용을 관찰할 수 없음. 오답 추론이라는 뜻이 아니라 설명 부재.'),
((2,2,2),'E: 조각이 말을 한다는 표현을 확인. R: 무생물에 인간 행위를 부여하는 의인화 정의 적합. A: 해당 표현과 B 연결.'),
((2,2,2),'E: 조각의 발화 표현 추출. R: 의인화 정의 정확. A: B 연결 정확.'),
((2,2,2),'E: 조각의 발화 표현 추출. R: 의인화 정의 정확. A: B 연결 정확.')]),
('Art_Theory',[
((2,1,1),'E: 하기아 소피아 인식 적합. R/A: 모자이크 결론은 맞지만 최초 장식과 후대 인물 모자이크를 구분하지 않아 시기 설명 불충분.'),
((2,1,1),'E: 실내 금색과 평면도를 올바르게 관찰. R: 모자이크를 stone에 설치한다고 제한하고 스투코가 평범한 표면이라는 구분은 과도. A: 금색만으로 원래 마감 재료를 확정하기에는 근거 부족.'),
((2,2,1),'E: 돔 교회와 실내 관찰 적합. R: 하기아 소피아 원래 유리 모자이크 사실과 부합. A: 현재 실내/평면만으로 원래 재료를 확인한다는 연결은 약함.'),
((2,2,1),'E: 돔 교회와 실내 관찰 적합. R: 하기아 소피아 모자이크 사실과 부합. A: 현재 사진과 원래 마감 사이 근거 설명 부족.')]),
('Basic_Medical_Science',[
((2,1,1),'E: 유전자·플라스미드·세균 과정은 대체로 맞음. R/A: genetic engineering은 전체 과정의 넓은 이름; 정답 transformation이라는 세포 유입 단계를 특정하지 않음.'),
((2,2,1),'E: 플라스미드 절단·삽입·세균 유입 관찰 정확. R: 재조합 DNA 기술 설명 타당. A: 넓은 기술명으로 답하여 transformation 단계 특정은 부족; 공식 Judge는 정답 처리.'),
((2,2,0),'E: 전체 과정 설명은 타당. R: 재조합 DNA 기술 정의 타당. A: 삽입을 3으로 지목하지만 3은 절단된 유전자, 삽입은 5; 7은 세균 증식 그림.'),
((2,2,0),'E: 전체 과정 설명은 타당. R: 기술 정의 타당. A: 5를 세균 형질전환, 6을 단백질 생산으로 잘못 매핑; 실제 6은 세균 유입.')]),
('Biology',[
((2,1,0),'E: Rb/E2F 관계 인식. R: 전사 조절 설명은 있으나 표적 단백질 기능 규칙이 부족. A: 전사인자 E2F 자체와 E2F가 발현시키는 S기 DNA 복제 단백질을 혼동.'),
((2,2,2),'E: Rb 방출 후 E2F의 S기 유전자 활성화 추출. R: S기에 DNA 복제가 필요함을 연결. A: DNA replication A 정확.'),
((2,1,0),'E: E2F/Rb 과정 대체로 맞음. R: 전사인자 기능 설명만으로 표적 단백질 기능을 설명하지 못함. A: 표적 단백질과 전사인자 기능 혼동.'),
((2,1,0),'E: E2F/Rb 과정 대체로 맞음. R: 전사인자에 관한 규칙은 불충분. A: 표적 단백질의 기능 대신 DNA 결합을 선택.')]),
('Chemistry',[
((0,2,0),'E: 치환기 위치/방향 오독. R: cis/trans의 같은 면/반대 면 정의 자체는 적합. A: cis-1,2로 잘못 명명; 그림은 trans-1-chloro-4-methylcyclohexane.'),
((0,0,0),'E: 1,4 치환을 1,2로 오독. R: 1,2에서 두 axial이 cis라는 설명은 잘못. A: cis 명명 오류; 공식 Judge 정답 처리와 설명 정확성 분리.'),
((0,0,0),'E: 결합 방향/위치 오독. R: cis/trans를 정하는 데 CIP 우선순위를 잘못 적용. A: 잘못된 구조명; 공식 Judge 정답 처리.'),
((0,2,1),'E: 위치 또는 방향 기술에 오류. R: cis/trans 정의는 적합. A: 최종 trans는 맞지만 1,4 위치를 명확히 지정하지 않은 불완전한 이름; 공식 Judge가 수용.')]),
('Clinical_Medicine',[
((None,None,None),'E/R/A: 태반 세부 조직상과 Rh 용혈의 기전/정답 C 표현에 전문 검토가 필요. 공식 정답을 설명 타당성 2점으로 자동 인정하지 않음.'),
((None,0,0),'E: 세부 조직 판독 미확정. R: anti-RhD가 영양막 수용체를 공격해 태아 빈혈을 만든다는 설명은 적혈구 항원 표적과 다름. A: 항수용체 A를 선택한 연결 오류.'),
((None,2,0),'E: 태반 조직 판독 미확정이며 Rh 음성은 주어진 사실이 아니라 추론. R: RhD 항체에 의한 태아 적혈구 파괴는 타당. A: 이를 항수용체 A로 매핑하는 것은 부적절.'),
((None,0,0),'E: 세부 조직 판독 미확정. R: anti-Rh를 anti-receptor로 동일시. A: C 기전을 언급하면서 A가 더 맞다고 선택해 기전/선지 매핑 실패.')]),
('Computer_Science',[
((0,0,0),'E: 17을 24에 연결하는 등 트리 오독. R: leftist rank를 왼쪽 최장 경로처럼 설명. A: 잘못된 병합으로 C를 놓침.'),
((1,1,0),'E: 대부분 노드는 맞으나 26의 좌우 관계 오독. R: min root라는 성질만 쓰고 delete-min 병합 규칙 없음. A: 삭제 뒤를 묻는데 기존 6이 root라는 사실로 답함.'),
((0,1,0),'E: 17/24 연결 오독. R: 일반 최소 힙 설명만 있고 왼쪽 편향 병합이 부족. A: 8이 root라고 하면서 8이 root가 아니라는 A를 선택.'),
((0,1,0),'E: 트리의 자식 관계 오독. R: 가장 작은 자식 교환만으로 leftist 병합을 설명하지 못함. A: 자체 설명과 A 선택이 충돌.')]),
('Design',[
((2,2,2),'E: 카메라 디자인의 시각 특징 인식. R: Beau Brownie 제품 식별 타당. A: 해당 이름의 선지 선택.'),
((2,2,2),'E: 이미지 디자인과 카메라 유형 일치. R: Beau Brownie 식별 적합. A: 제품명 선택 정확.'),
((2,2,2),'E: 이미지 카메라 특징 일치. R: 제품명 식별 적합. A: 정답 매핑 일관.'),
((2,2,2),'E: 이미지 카메라 특징 일치. R: 제품명 식별 적합. A: 정답 매핑 일관.')]),
('Diagnostics_and_Laboratory_Medicine',[
((0,0,0),'E/R/A: 답만 제시되어 설명 부재. 실제 진단 추론의 정확성을 관찰할 수 없음.'),
((0,None,None),'E: 뇌간 단면을 좌우 대뇌반구/고랑으로 기술하는 관찰 오류. R/A: PSP와 Parkinson 감별의 조직학적 타당성은 전문 검토 필요; 공식 오답은 별도 기록.'),
((None,None,None),'E/R/A: cistern/ventricle 해석과 단독 PSP 확진의 타당성을 확정할 수 없어 미확정. 자동 오답을 곧바로 세 항목 0점으로 바꾸지 않음.'),
((0,None,None),'E: 제시된 뇌간 단면에서 frontal lobe atrophy를 주장하는 근거 오류. R/A: 질환 감별에 대한 전문 검토 필요.')]),
('Economics',[
((2,2,2),'E: 수출 50/수입 40 추출. R: 순수출=수출−수입. A: 10 정확.'),
((2,2,2),'E: 표 수출 50/수입 40 정확. R: 순수출 정의 정확. A: 50−40=10.'),
((2,2,2),'E: 표 수출 50/수입 40 정확. R: 순수출 정의 정확. A: 50−40=10.'),
((2,2,2),'E: 표 수출 50/수입 40 정확. R: 순수출 정의 정확. A: 50−40=10.')]),
('Electronics',[
((0,1,0),'E: 회로 연결과 전압 v2 위치를 잘못 읽는 구간. R: 임피던스 일부는 맞으나 연결 모델이 잘못. A: 잘못된 회로 계산과 length 종료; 발췌 검토.'),
((0,0,0),'E: 전류원 8A를 그대로 특정 저항전류로 간주. R: 각주파수 250에서 80mH 임피던스는 j20인데 j40 등의 계산. A: 연결/임피던스 오류와 length 종료.'),
((0,1,0),'E: 배터리와 분기 연결 해석 오류. R: 중첩 원칙 자체는 맞음. A: 전압원을 끈 상태의 저항 전압이 무조건 0이라는 처리 등 잘못된 회로 해석; 발췌.'),
((0,0,0),'E: v2를 인덕터 전압으로 취급하지만 도면은 15Ω 양단. R: 다른 회로에 대한 식을 구성. A: 잘못된 복소수 회로 계산과 length 종료; 발췌.')]),
('Energy_and_Power',[
((2,2,2),'E: 직경과 절대압/대기압 관계를 최종적으로 정확히 사용. R: 연속식과 Bernoulli 적용. A: 중간 혼동을 수정한 최종 유도에서 5.25 kg/s 도출.'),
((0,2,0),'E: 출구 직경 4cm를 높이처럼 해석. R: Bernoulli와 연속식 자체는 타당. A: 대기압 차를 사용하지 않고 계산한 중간값이 최종 5.25와 일치하지 않음.'),
((1,2,0),'E: 입력 일부는 정확하나 압력 단위/높이 해석이 혼재. R: Bernoulli/연속식 적합. A: 상류속도항 무시와 압력 해석으로 자기 계산이 최종값을 지지하지 못함.'),
((0,0,0),'E: H를 분출 높이로 오독; 실제 정체압 측정관. R: 대기압/정체압 관계 오류. A: 차원 및 압력 오류, 반복과 length 종료; 발췌.')]),
('Finance',[
((1,0,0),'E: 환율 표는 사용하지만 문제의 미래 실제환율 0.50 CHF/$를 누락. R: 선물환 계약이 명시되지 않았는데 선물환율 적용. A: 1,000,000/0.50=2,000,000이어야 함.'),
((1,1,0),'E: 표 숫자 정확하나 결정적 조건 0.50 누락. R: 달러/프랑 환산 자체는 맞지만 적용 조건(선물환 계약) 미확인. A: 1.6460을 적용해 잘못된 금액.'),
((1,0,0),'E: 결정적 조건 0.50 누락. R: 미래 지급에는 항상 선물환율을 쓴다는 규칙 잘못. A: 문제에서 준 실제환율 대신 1.6460 적용.'),
((1,0,0),'E: 결정적 조건 0.50 누락. R: 미래 지급에 무조건 선물환율 적용하는 규칙 잘못. A: 1.6460 적용으로 오답.')]),
('Geography',[
((2,2,2),'E: Tampa 주변 지명과 해안선 관찰. R: 지역 지명으로 지도 위치 식별. A: Tampa 지역 일관.'),
((2,0,1),'E: 지도 지명 대체로 정확. R: 가장 큰 글자 지명을 고른다는 규칙은 지역 식별의 타당한 근거가 아님. A: Tampa 정답이지만 제시 규칙으로 도출하지 못함.'),
((0,2,1),'E: 지도 공군기지 명칭을 McAlester로 환각. R: 지명/해안선 활용 자체는 적절. A: Tampa 정답이나 잘못 읽은 지명에 일부 의존.'),
((2,0,0),'E: 지도에 보이는 지명 대체로 읽음. R: 수역 표기를 곧 지역명으로 취급. A: Tampa 대신 Bay를 답함.')]),
('History',[
((0,0,0),'E/R/A: C 선택만 있고 추론 설명은 없음; 관찰 불가와 오답 추론을 구별.'),
((2,0,0),'E: 유럽 식민지 지도를 맞게 읽음. R: mercantilism을 수출 억제/수입 장려로 거꾸로 설명. A: 이 규칙이 선택 C를 뒷받침하지 못함.'),
((2,2,2),'E: 아메리카 식민 지배 분포 추출. R: 식민지 원료·본국 부의 축적이라는 중상주의 설명 적합. A: C 연결 정확.'),
((2,2,2),'E: 식민 지배 분포 추출. R: 중상주의 설명 적합. A: C 연결 정확.')]),
('Literature',[
((0,0,0),'E/R/A: 답만 제시되어 설명 부재.'),
((0,1,0),'E: 남자아이를 girl로도 부르고 문으로 향한다고 오독. R: 증거로 소거한다는 일반 원칙은 있으나 작품 내용 지식 없음. A: D를 틀렸다고 한 뒤 D를 선택, 중복 섹션.'),
((0,0,0),'E: 그림에서 문 쪽으로 뛴다고 오독. R: 소년이 여성의 신발 자체를 훔치려 했다는 작품 설명은 잘못; 지갑을 노린 동기가 신발 구매. A: 잘못된 줄거리를 근거로 D.'),
((0,0,0),'E: 문으로 향한다는 이미지 해석 오류. R: 여성의 신발을 훔치는 이야기라고 설명해 대상/동기 혼동. A: 올바른 D를 잘못된 근거로 정당화.')]),
('Manage',[
((2,0,0),'E: 기간 표 일부 정확. R: backward pass/슬랙 식을 잘못 쓰는 구간 확인. A: 계산과 선지 해석 혼란; 발췌 검토. 문제 본문(완료일)과 선지(슬랙 벡터) 불일치 의심.'),
((1,0,1),'E: 일부 선행 관계를 부정확하게 기술. R: 가장 이른 경로 종료를 프로젝트 완료로 설명하는 오류. A: 최종 14는 맞으나 F→G를 추가하고 A의 슬랙 벡터를 유도하지 않음.'),
((2,0,0),'E: 표의 기간/관계 일부 추출. R: 선행 관계와 슬랙 계산법을 변경. A: 잘못된 슬랙과 반복; 발췌 검토.'),
((2,0,0),'E: 입력 표 대체로 추출. R: 슬랙 계산식 잘못. A: B 기간 2를 3처럼 계산하고 F→G 관계를 추가하여 틀린 벡터.')]),
('Marketing',[
((1,2,2),'E: 0부터36이 38개라는 표현은 부정확(00 포함시38). R: 균등확률 규칙 정확. A: 세 숫자/38=3/38 계산은 맞음.'),
((2,2,2),'E: 0/00 포함 38칸과 세 숫자 추출. R: 유리한 경우/전체 경우. A: 3/38 정확.'),
((1,2,2),'E: first dozen 범위가 1~18이라는 부수 설명 오류(실제1~12). R: 세 숫자 확률 3/38 규칙 맞음. A: 물은 베팅 확률 계산은 정확.'),
((1,2,2),'E: first dozen 범위 부수 오류. R: 균등한 38결과 중3이라는 규칙 타당. A: 질문한 확률 3/38 정확.')]),
('Materials',[
((0,1,0),'E: 힘 방향/모멘트 팔 해석 오류가 있는 발췌. R: 평형 개념은 사용. A: 잘못된 평형식과 장문 반복; length 종료.'),
((0,2,0),'E: 75도를 수직 기준으로 잘못 쓰고 FC 모멘트 팔을 0.8로 오독. R: 모멘트 평형·σ=F/A 자체는 타당. A: 수평 반력 FC=0을 임의 가정하여 맞는 수치에 도달; 모멘트가0인 것과 힘이0인 것은 다름.'),
((2,2,1),'E: 기본 하중과 면적 추출. R: 평형 후 응력 계산 원칙은 맞음. A: 다수 잘못된 식 뒤 최종 2FB sin75=70으로 수정해121; 최종 계산은 맞으나 반력/기하 설명에 오류가 남음.'),
((0,0,1),'E: FC의 모멘트 팔 오독. R: FB를 빠뜨린 모멘트 식은 잘못. A: 70/2=7 등 오류 후 마지막에 2FB sin75=70 및121로 수정; 최종 계산과 앞선 Rule 불일치.')]),
('Math',[
((2,2,2),'E: 원뿔/원기둥 기하 정확. R: 닮음 r=R(1-h/H), 부피 최대화 정확. A: h=H/3,r=2R/3과 경계 비교 일관.'),
((2,0,0),'E: 기하 변수 추출 적합. R: 처음 r/h=R/H라는 잘못된 닮음 규칙. A: 후반 h=H/3을 얻고도 r=R/3이라 계산하여 오답.'),
((2,0,0),'E: 기하 변수 추출 적합. R: 부피 최대에 dr/dh=0이라고 기술(필요한 것은 dV/dh). A: h=2H/3로 미분 결과를 잘못 계산.'),
((1,1,1),'E: R을 꼭짓점-밑면 거리라고 부르는 부수 오류. R: 부피 분모 H²를 H로 누락. A: 상수배 오류는 최적점에 영향 없어 h=H/3,r=2R/3은 맞지만 제시 부피식은 부정확.')]),
('Mechanical_Engineering',[
((1,None,1),'E: 도형 일부 인식하나 투상 면/절삭부 해석 불명확. R: 3D 복원 타당성 미확정. A: B 정답이나 그림으로 전체 유도 확인이 부족.'),
((0,0,0),'E: main/top view를 뒤바꿔 읽음. R: 정면도를90도 돌리는 것이 좌측 투상도라는 규칙은 잘못. A: 올바른 B를 잘못된 방법으로 선택.'),
((0,0,0),'E: 도면에 없는 중앙 사각 구멍을 가정. R: 그 구멍이 반원 내 원으로 보인다는 근거 없는 투상. A: 잘못된 D.'),
((0,0,0),'E: 중앙 사각 구멍 환각. R: 투상 관계 부적절. A: 잘못된 D.')]),
('Music',[
((0,0,0),'E/R/A: 선택지만 있어 설명 부재.'),
((0,0,0),'E: 음표 길이/묶음을 잘못 읽음. R: 2/4가 4분음표+쉼표여야 한다는 제한은 잘못. A: 박 묶음이 적합한 B를 놓침.'),
((0,1,0),'E: 8분음표를 4분음표처럼 세는 등 오독. R: 2/4의 총2박 개념은 있으나 세부 적용 조건 부족. A: 실제 음가/빔 묶음과 다른 판정.'),
((0,0,0),'E: 음가와 묶음 오독. R: 특정 음표/쉼표 조합만 허용한다는 규칙 잘못. A: 적절한 B를 놓침.')]),
('Pharmacy',[
((2,2,None),'E: 영양표15cal,2g,30g 추출. R: 열량/질량 및 단위변환 원칙 맞음. A: 긴 응답 발췌에서7.5의1유효숫자 반올림 혼란; 마지막 수정의 일관성을 확정하지 않아 미확정.'),
((2,2,1),'E: 표 핵심 값 정확. R: 열량/질량, 1유효숫자 기준과 g→kg 변환 적합. A: 결과는 맞으나 15가1유효숫자라는 잘못된 설명 포함(제한값2가1자리).'),
((0,2,0),'E: %DV를 하루 열량 비율로 혼동하고 용기당14회를 30g당14회로 사용. R: 열량/질량의 차원 관계는 맞음. A: .043kg 등 계산이 최종 선택과 모순.'),
((2,2,1),'E: 주요 표 숫자 정확. R: 필요한 나눗셈/단위변환 적합. A: 결과는 맞으나 15의 유효숫자 수를 잘못 설명.')]),
('Physics',[
((2,0,0),'E: nf가 양쪽보다 큰 배치 인식. R: 두 반사 모두π라는 오해. A: λ/2를 선택.'),
((2,0,0),'E: 굴절률 배치는 정확. R: 높은→낮은 반사의 위상 규칙을 반대로 기술하고 적용도 모순. A: 두번π라고 계산해 λ/2.'),
((2,2,0),'E: 배치 정확. R: 낮은→높은 반사의π 규칙 자체는 맞음. A: 아래 nf→n2 반사에도π를 적용하여 오답.'),
((0,0,0),'E: nf→n2에도 위상 반전이라고 관측문에 잘못 기술. R: 반사는 반대 위상이라고도 하여 E와 모순; 조건 표현도 부정확. A: 두 면 모두 위상 반전이라고 유지하면서 λ/4를 도출, 정답 D지만 추론 오류.')]),
('Psychology',[
((0,0,0),'E/R/A: 답만 제시되어 설명 부재.'),
((2,2,0),'E: 초기 급락 및 반복 복습의 완만화 관찰 정확. R: 복습의 망각 감소 효과 적합. A: 점수 개선을 말하며 초기 급락/이후 유지라는 선택과 어긋남.'),
((2,1,1),'E: 곡선 경향 정확. R: 매년 시험이 곧 네 번째 복습 곡선을 따른다는 특정화는 근거 부족. A: A와 대략 연결하나 초기 급락 설명 부족.'),
((2,2,1),'E: 곡선의 초기 급락/이후 완만화 정확. R: 반복 복습으로 망각 완화 적합. A: 정답 A지만 급락하지 않는다고 설명해 초기 구간과 모순.')]),
('Public_Health',[
((0,2,0),'E: 비질환군의 TCT 층들을 합하지 않고 일부 셀만 선택. R: 특이도=TN/(TN+FP)는 맞음. A: 정답 A를 골랐어도 설명 계산이465/500과 다름; 발췌 확인.'),
((0,2,0),'E: TN=245,FP=10만 사용하여 다른 TCT 층을 누락. R: 특이도 정의 정확. A: 실제 TN220+245=465,FP10+25=35이므로.93인데.96으로 계산.'),
((0,0,0),'E: 표의 층 집계 누락. R: 뒤에서 질환군 등을 특이도 분모로 사용. A: 잘못된 집계와 식으로 오답.'),
((0,2,0),'E: TN/FP를 일부 층에서만 추출. R: 특이도 식 정확. A: 245/(245+10) 대신465/500이어야 함.')]),
('Sociology',[
((0,0,0),'E/R/A: 답만 제시되어 설명 부재.'),
((2,None,1),'E: 조직도 역할과 부서를 정확히 읽음. R: 효율·전문성은 알려진 관료제 특징이나 가치/도덕의 완전 부재라는 강한 귀속은 이 검토에서 확정하지 못함. A: A/B/C→D 매핑은 맞지만 Rule에 C 근거 설명이 부족.'),
((2,None,1),'E: 조직도 부서와 계층 인식 정확. R: 가치/도덕 부재라는 Weber 해석은 1차 문헌 검토 없이 확정하지 않음. A: 선지 조합은 맞으나 원인 설명이 선지 재진술 수준.'),
((2,None,1),'E: 조직도 내용 정확. R: Weber의 가치/도덕 부재 주장 귀속 미확정. A: D 매핑은 맞지만 선택지 재진술 중심.')]),
]
assert len(data)==30
sources={
 'Agriculture':['https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/late-blight'],
 'Art_Theory':['https://www.metmuseum.org/pt/essays/glass-ornaments-in-late-antiquity-and-early-islam-ca-500-1000'],
 'History':['https://openstax.org/books/world-history-volume-2/pages/5-3-the-mercantilist-economy'],
 'Physics':['https://openstax.org/books/college-physics/pages/27-7-thin-film-interference'],
 'Computer_Science':['https://ece.uwaterloo.ca/~dwharder/aads/Algorithms/Leftist_heaps/'],
 'Clinical_Medicine':['https://www.ncbi.nlm.nih.gov/sites/books/NBK2269/?report=reader'],
 'Literature':['https://openlab.citytech.cuny.edu/schoning1121sp2026/files/2024/04/Thank-You-Maam-Hughes.pdf'],
}
partial={('Architecture_and_Engineering','baseline'),('Architecture_and_Engineering','epoch_3'),*[( 'Electronics',m) for m in models],('Energy_and_Power','epoch_4'),('Manage','baseline'),('Manage','epoch_3'),('Materials','baseline'),('Pharmacy','baseline'),('Public_Health','baseline')}
annotations=[]
for row,(subject,notes) in zip(rows,data):
    assert row['subject']==subject
    for model,(scores,note) in zip(models,notes):
        record=row['models'][model]['record'];raw=record['model_response']['raw']
        q=[]
        for name in ('Evidence','Rule','Application'):
            matches=list(re.finditer(r'(?im)^'+name+r':\s*',raw))
            if matches:
                start=matches[0].end(); end=re.search(r'(?im)^(Evidence|Rule|Application|Final answer):',raw[start:])
                text=raw[start:start+end.start()] if end else raw[start:]
                q.append({'component':name,'exact_quote':text[:400],'truncated':len(text)>400,'offset':start})
        if not q:q=[{'component':'unstructured','exact_quote':raw[:600],'truncated':len(raw)>600,'offset':0}]
        annotations.append({'id':row['id'],'subject':subject,'model':model,'official_correct':record['grading']['final']['hit'],
          'scores':dict(zip(('E','R','A'),scores)),'rationale':note,'reviewer':'assistant qualitative review; not independent human expert',
          'coverage':'prefix_and_tail_excerpts' if (subject,model) in partial else 'full_response',
          'image_reviewed':True,'explanation_absent':model=='baseline' and subject in {'Art','History','Literature','Diagnostics_and_Laboratory_Medicine','Music','Psychology','Sociology'},
          'sources':sources.get(subject,[]),'quotes':q,'raw_response_chars':len(raw)})
summary={}
for model in models:
    group=[r for r in annotations if r['model']==model]
    summary[model]={'responses':30,'official_correct':sum(r['official_correct'] for r in group),'full_response_read':sum(r['coverage']=='full_response' for r in group),'excerpts_read':sum(r['coverage']!='full_response' for r in group),'explanation_absent':sum(r['explanation_absent'] for r in group),
      'joint_all2':sum(all(v==2 for v in r['scores'].values()) for r in group),
      'correct_with_observed_critical_error':sum(r['official_correct'] and 0 in r['scores'].values() and not r['explanation_absent'] for r in group),
      'components':{c:dict(collections.Counter('unknown' if r['scores'][c] is None else str(r['scores'][c]) for r in group)) for c in ('E','R','A')}}
output={'scope':'prespecified 30 questions, 120 responses, 31 original images; no extrapolation to all 900 content judgments',
 'rubric':{'0':'critical error or missing explanation; missing is flagged separately','1':'partially accurate, incomplete, or residual inconsistency','2':'sufficiently accurate for this question','null':'not established, excluded from scored denominator'},
 'policy':'Written E/R/A content judged against image/question, not just final answer. Explicit corrected work can receive partial/full application credit; unresolved false Rule remains marked. Long excerpts can establish an observed critical error but are not evidence that every sentence was reviewed.',
 'summary':summary,'annotations':annotations}
(ROOT/'semantic_review.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
