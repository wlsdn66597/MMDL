import json
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
stats=read(R/'evidence/summary.json')['metrics'];pairs=read(R/'paired_results.json');review=read(R/'semantic_review.json');cases=review['discordant_prespecified_pairs'];summary=review['summary'];idx=read(R/'all900_index.json')
models=['baseline']+[f'{arm}_epoch_{e}' for e in (2,3,4) for arm in ('plain','era')]
labels={m:'Baseline' if m=='baseline' else ('일반 ' if m.startswith('plain') else 'ERA ')+m[-1] for m in models}
def table(head,body):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |']+['| '+' | '.join(map(str,row))+' |' for row in body])
report=['# 일반 풀이와 ERA LoRA의 MMMU 900문항 대조','기존 200문항으로 학습한 두 decoder LoRA 모두 baseline 575/900보다 낮았다. 일반 풀이와 ERA 사이의 순위는 epoch에 따라 달라졌다. 총점 차이는 1~11문항이지만, 어느 문항을 맞혔는지는 약 5분의 1에서 달랐다.',
'## 공식 점수','동일 seed 3407, 동일한 900개 고유 문항. 괄호 안 차이는 baseline 대비 퍼센트포인트다. 반복 실행은 재현 확인용이다.',
table(['모델','정답/900','정답률 (Δ%p)','표제율','완전 ERA율','길이 상한 오답','동일 줄 3회 이상'],[[labels[m],f"{stats[m]['correct']}/900",f"{stats[m]['accuracy_percent']:.2f}% ({stats[m]['accuracy_percent']-stats['baseline']['accuracy_percent']:+.2f})",f"{stats[m]['legacy_strict_headings_percent']:.2f}%",f"{stats[m]['complete_era_format_percent']:.2f}%",stats[m]['length_count'],stats[m]['repeated_line_3plus_count']] for m in models]),
'완전 ERA는 Evidence → Rule → Application 세 구간이 정확히 한 번씩 순서대로 나오고 내용이 채워지며, 정상 종료한 응답 끝에 최종 답이 있는 경우다. 일반 풀이에는 ERA 표제를 지시하지 않았으므로 이 열은 출력 양식을 나타낸다.']
report+=['## 같은 문항에서 정오가 바뀐 방향',
table(['비교 (왼쪽 → 오른쪽)','둘 다 정답','왼쪽만 정답','오른쪽만 정답','둘 다 오답','순변화'],[[labels[l]+' → '+labels[r],v['both_correct'],v['left_only'],v['right_only'],v['both_wrong'],v['right_only']-v['left_only']] for k,v in pairs.items() for l,r in [k.split('__')]]),
'같은 epoch에서 일반·ERA의 정오가 엇갈린 비교는 193, 197, 199건이다. 이 중 오답 쪽이 길이 상한에 걸린 비교는 각각 18, 22, 11건이다. 나머지 175, 175, 188건은 둘 다 정상 종료한 뒤 정답이 갈렸다. 같은 문항이 여러 epoch에 등장할 수 있다.',
'같은 문항의 쌍을 사용한 양측 정확 McNemar 검정의 탐색적 p값은 '+', '.join(f"{pairs[f'plain_epoch_{e}__era_epoch_{e}']['mcnemar_exact_two_sided']:.3f}" for e in (2,3,4))+'이다. 이 단일 seed에서는 두 풀이 방식 중 어느 하나의 일관된 정답률 우위를 확인하기 어렵다. 일반 풀이도 baseline 대비 50~73문항 감소했다. 두 LoRA의 차이에는 풀이 형식과 함께 목표 설명의 사실·길이·문체가 들어 있다.',
'## 정답이 갈린 풀이의 직접 대조','사전 선정한 분야별 1문항, 총 30문항에서 원본 이미지 31장을 확인했다. 새 일반 풀이 90응답을 검토하고 기존 baseline·ERA 120응답의 주석을 연결했다. 이 표본 안에서 정오가 갈린 14개 epoch 비교(11개 고유 문항)를 아래에 모두 싣는다. E는 질문·이미지 근거 선택, R은 규칙·적용 조건, A는 계산·논리·최종 답 매핑이다. 지목한 단계는 작성된 풀이에서 확인한 차이다.']
for x in cases:
    report+=['### '+x['id'].removeprefix('validation_')+f" · {x['epoch']}epoch · "+('일반만 정답' if x['plain_correct'] else 'ERA만 정답'),'**'+x['stage']+'**. '+x['rationale'],f"> 일반: “{x['plain_quote']}”\n>\n> ERA: “{x['era_quote']}”",f"[문항·이미지·일곱 모델 원문](explorer.html#{x['id']})"]
report+=['## 정답과 설명의 정확성','공중보건 9번에서 baseline은 공식 정답 0.93을 골랐다. 그러나 말미에는 민감도 70/75가 0.93이라는 이유를 들었다. 표의 비질환군에서 전자검사 음성은 220+245=465명이므로 실제 특이도는 465/500=0.93이다. 일반 풀이와 ERA도 두 층을 모두 집계하지 못했다. [공중보건 원문](explorer.html#validation_Public_Health_9)',
'화학 4번의 기준답은 trans-1-chloro-4-methylcyclohexane이다. 일반 3epoch는 cis-1-chloro-2-methylcyclohexane이라고 썼는데도 공식 Judge가 정답으로 처리했다. 원래 채점 정책에 따른 비교와 설명 내용 검수를 구분해야 하는 사례다. [화학 원문](explorer.html#validation_Chemistry_4)',
'박막 사례에서는 낮은 굴절률에서 높은 굴절률 경계로 반사될 때 한 번만 위상이 반전한다. 따라서 보기 중 가장 작은 보강 두께는 막 안 파장의 1/4이다. [OpenStax 물리학](https://openstax.org/books/college-physics/pages/27-7-thin-film-interference)',
'### 사전 선정 30문항의 내용 검수','0은 핵심 오류 또는 필요한 설명 부재, 1은 부분 정확, 2는 충분 정확이다. 미확정은 해당 성분 분모에서 제외했다. 일반 풀이도 표제 유무와 무관하게 실질 내용으로 평가했다. 정답 중 0 포함은 공식 정답인데 어느 성분이 0인 응답 수이며 설명 생략도 포함한다.',
table(['모델','공식 정답/30','E 충분/판정','R 충분/판정','A 충분/판정','E/R/A 모두 2','정답 중 0 포함','긴 응답 발췌'],[[labels[m],q['correct'],*[f"{q['components'][k].get('2',0)}/{30-q['components'][k].get('None',0)}" for k in 'ERA'],q['all2'],q['correct_with_zero'],q['excerpt']] for m in models for q in [summary[m]]]),
'기존 에너지·동력 3번의 ERA 3epoch A 점수는 원문 말미를 다시 대조해 0에서 1로 수정했다. 실제로 압력 차와 출구 면적으로 C를 근사했다. 공식 정답 점수는 기존 값을 사용한다. 그 외 기존 모델의 내용 점수는 이전 검토를 재사용했다. 새 일반 풀이 검수는 assistant의 정성 판단이며, 임상 조직 사진 등 전문 판독이 불확실한 항목은 미확정으로 남겼다. [근거와 발췌 범위](semantic_review.json)',
'## 분야별 정답과 정오 변화','각 분야 30문항. 일반/ERA 칸은 동일 epoch에서 각 방식만 맞힌 개수다.',
table(['분야','기준','일반2','ERA2','일반3','ERA3','일반4','ERA4','2 일반만/ERA만','3 일반만/ERA만','4 일반만/ERA만'],[[s,*[stats[m]['by_subject'][s]['correct'] for m in models],*[f"{v['left_only']}/{v['right_only']}" for e in (2,3,4) for v in [pairs[f'plain_epoch_{e}__era_epoch_{e}']['by_subject'][s]]]] for s in sorted(stats['baseline']['by_subject'])]),
'## 검증과 범위','세 일반 LoRA는 같은 4epoch 학습의 26·39·52번째 optimizer step 체크포인트다. 학습 쌍 검증 기록은 동일한 초기 adapter·학습률 이력·입력 문항·동결된 기반 가중치를 확인한다. 결과 회수 후 SHA256, 모든 모델의 900개 ID·순서·프롬프트·이미지 필드·토큰 수, 최종 채점 합계, 길이 상한 오답 정책을 대조했다. 일곱 모델 모두 반복 실행의 응답·채점 및 Judge 기록이 시간 필드를 제외하고 일치했다. 중단 전 보존 파일 54개, 실행 코드 12개 및 처음 423개 응답 원본도 서버에서 재확인했다.',
'전체 900문항의 원문과 정오 비교는 탐색기에서 확인할 수 있다. E/R/A의 직접 검수는 사전 선정 30문항으로 한정하며 표본에서 관찰한 오류를 전체 불일치의 원인 비율로 확대하지 않는다. 이전 8문항 예비 평가와 별도로 집계했다.',
'[전체 900문항 탐색기](explorer.html) · [모든 불일치 ID](discordant_items.json) · [분야별 쌍 비교](paired_results.json) · [검증 기록](verification.json)']
report.insert(report.index('## 같은 문항에서 정오가 바뀐 방향'), '완전 ERA 응답 중에서도 오답은 2epoch 311/792, 3epoch 329/813, 4epoch 330/808이었다. 형식이 완성된 응답의 약 39~41%에서 최종 정답을 얻지 못했다.')
report.insert(report.index('## 정답이 갈린 풀이의 직접 대조'), '세 epoch 중 한 번이라도 두 LoRA의 정오가 갈린 **고유 문항은 403/900**이다. [불일치 전체 목록](discordant_items.json)')
report.insert(report.index('## 분야별 정답과 정오 변화'), '검토한 ERA 30응답에서 A가 충분한 것은 각 epoch 모두 6/29였다. R이 충분한 경우는 12/28, 14/28, 12/28이었다. 따라서 검수에서는 올바른 규칙을 말한 뒤에도 대입·선지 선택이 그 규칙을 따르는지 별도로 확인할 필요가 있다. 분야당 한 문항인 이 표본에서 관찰한 수치다.')
report.insert(report.index('## 검증과 범위'), '생물학 18번의 표적 유전자와 DNA 합성의 연결은 실험 연구에서도 확인된다. [DeGregori 등, E2F1 표적 연구](https://pubmed.ncbi.nlm.nih.gov/7623816/)')
(R/'report.md').write_text('\n\n'.join(report)+'\n',encoding='utf8')
print('summary',[(m,summary[m]['all2'],summary[m]['correct_with_zero'],summary[m]['excerpt']) for m in models])

