from pathlib import Path
import json
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
a=read(R/'analysis_metrics.json'); s=read(R/'semantic_review.json'); rows=read(R/'evidence/pilot/prespecified30_responses.json')
labels={'baseline':'Baseline','epoch_2':'2 epoch','epoch_3':'3 epoch','epoch_4':'4 epoch'}
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
def percent(n,d=900):return f'{100*n/d:.2f}%'
def quote(subject,model,fragment):
    raw=next(r for r in rows if r['subject']==subject)['models'][model]['record']['model_response']['raw']
    assert fragment in raw,(subject,model,fragment)
    return '> '+fragment.replace('\n','\n> ')
text='''# ERA decoder LoRA — MMMU 900문항 파일럿 결과

2026-11-02 · 단일 추론 seed 3407 · 기존 baseline과 동일 조건

**ERA 형식은 대부분의 문항으로 전이됐지만, 정답률은 baseline보다 낮아졌다.** 2→4 epoch로 학습을 늘려도 정답률 개선은 관찰되지 않았다. 형식 실패가 아닌, 입력 근거의 오독·부적합한 규칙·계산 및 선지 연결 오류가 실제 응답에서 확인됐다. 다만 200문항의 양, 정답 설명 품질, 학습률, LoRA 대상 중 무엇이 주원인인지는 이번 비교만으로 분리할 수 없다.

[전체 900문항 탐색기](explorer.html) · [30문항/120응답 상세 내용 검토](semantic_review.md) · [정량 지표 JSON](analysis_metrics.json) · [독립 검증 기록](local_verification.json)

## 평가 범위와 보존

사용자의 범위 변경에 따라 epoch4 seed3407 반복 재현 검사까지만 완료했다. seeds 42·123·2026·9876은 실행되지 않았고 정지된 원래 제어기는 종료됐다. 새 학습이나 추가 추론은 실행하지 않았다. 원래 plan과 scope_change, 완료된 원본 출력은 보존했다. 원래 campaign.exit=143은 사용자가 요청한 제어기 종료이며 평가 실패가 아니다. 완료 근거는 pilot_completed.json과 독립 검증이다.

비교는 baseline과 epoch2/3/4 각각 같은 900개 문항, 각 분야30개다. 후보 모델의 본평가 응답은2700개이며, baseline 재현900개 및 후보 반복2700개는 점수 집계에서 제외했다. **5 seed 평균·표준편차나 4500개 독립 문항 통계는 보고하지 않는다.** 기존 학습20문항/MMMU8문항 진단과 별개인 결과다.

학생은 동일 revision의 Qwen3-VL-4B-Instruct, 후보는 같은 4epoch 학습 실행의 epoch2/3/4 체크포인트다. frozen baseline 평가 코드, vLLM0.11.0, BF16, 동일 GPU 환경, 입력 프롬프트·이미지 처리·sampling·Judge를 유지했다. 추론 설정은 temperature0.7, top_p0.8, top_k20, presence_penalty1.5, 출력 상한32768, 이미지 면적1,003,520~4,014,080 pixels다. 이는 **학습** 길이4096·학습 이미지 면적과 다른 기존 baseline **평가** 설정이다. 입력에 ERA 형식을 요구하는 추가 문구는 없다.

로컬에서 118개 원본 파일의 SHA256, 각 실행900개 고유ID, 최종 채점900개, 정답 합계, 문항 순서·프롬프트·이미지 메타데이터·입력 토큰·설정 동일성을 검증했다. baseline 및 세 epoch 모두 시간 필드를 제외한 원문 응답·생성 정보·채점·Judge 기록이 반복 실행과 일치했다. checkpoint 해시는 plan/launcher/manifest 간 일치한다. 가중치 자체는 내려받지 않았고 원격 사전 검증 기록을 보존했다. 재현 통과는 이 실행 환경에서의 결정성을 뜻하며, 학습 seed 간 변동을 검증한 것은 아니다.

## 전체 900문항 정량 결과

'''
text+=table(['모델','정답','정답률','baseline 대비','ERA 표제율(legacy)','완전 ERA 형식률'],[[labels[m],f"{x['correct']}/900",percent(x['correct']),f"{(x['correct']-575)/9:+.2f}pp" if m!='baseline' else '—',f"{percent(x['legacy_strict_headings_count'])} ({x['legacy_strict_headings_count']})",f"{percent(x['complete_era_format_count'])} ({x['complete_era_format_count']})"] for m,x in a['metrics'].items()])
text+='''
표제율은 기존 파일럿의 Evidence/Rule/Application 정규식이다. 완전 형식은 순서대로 유일한 세 표제, 비어 있지 않은 본문, 마지막 Final answer 행, 정상 stop을 모두 요구한다. 본문의 사실성·계산 타당성·정답 여부는 검사하지 않는다. baseline은 ERA를 요구받지 않았으므로 0% 자체가 문제는 아니다.

2/3/4 epoch에서 완전한 ERA이면서 오답인 응답은 각각 **311/329/330개**다. 완전 형식과 정답을 동시에 만족하는 응답은481/484/478개(53.44/53.78/53.11%)다. 이 수치 역시 설명 내용이 옳다는 뜻은 아니다.

2 epoch는 이번 단일 seed에서 정답률이 가장 높고, 3 epoch는 완전 형식률과 출력 길이가 유리하다. 2→4 epoch 정답 차이는11문항이며, 체크포인트 간 우열이 다른 seed/새 검증셋에서도 유지된다고 단정할 수 없다. 모든 문항에서 ERA를 출력한다는 목표도 아직 충족하지 않았다.

## 같은 문항에서의 정오 변화

'''
text+=table(['비교','오답→정답','정답→오답','둘 다 정답','둘 다 오답'],[[k.replace('baseline','Baseline').replace('epoch_','ep').replace('_to_',' → '),v['wrong_to_right'],v['right_to_wrong'],v['both_correct'],v['both_wrong']] for k,v in a['paired'].items()])
text+='''
Baseline에서 맞다가 틀린139/142/146개 중 새 응답이 완전 ERA인 경우가113/119/121개다. 길이 상한 도달은 그중14/11/14개다. 따라서 퇴행 대부분을 형식 미준수나 출력 중단만으로 설명할 수 없다.

객관식847문항만 따져도 정답 수는543→489/485/475개다. 주관식53문항은32→25/24/28개다. 주관식 Judge의 느슨한 채점 사례가 있지만 전체 성능 하락이 그 문제만으로 생긴 것은 아니다.

## 길이와 반복

'''
text+=table(['모델','평균 출력 토큰','중앙값','길이 상한 오답','동일 줄3회 이상','평균 반복4-gram 비율'],[[labels[m],f"{x['mean_output_tokens']:.1f}",x['median_output_tokens'],x['length_count'],x['repeated_line_3plus_count'],f"{100*x['mean_repeated_4gram_fraction']:.2f}%"] for m,x in a['metrics'].items()])
text+='''
LoRA 모델은 출력이 짧아지고 길이 상한·반복도 줄었다. 다만 동일 줄 진단은 반복의 휴리스틱이며 정당한 반복도 포함할 수 있다. 길이 제한 응답은 baseline과 동일하게 전부 오답 처리했다. 출력이 짧다는 사실만으로 추론의 충분성이나 효율 향상을 확정하지 않는다.

## 사전 선정 30문항의 E/R/A 내용 검토

출력 전 고정된 분야별1개, 기존8문항 제외 표본을 사용했다. 동일 pinned cache의 frozen prepare.load_data로 회수한 원본 이미지31장을 직접 열어 질문·정답과 대조했다. 120응답 중108개는 전체를 읽고, 장문 반복12개는 앞부분과 끝부분을 발췌 검토했다. 발췌 검토로 핵심 오류를 확인할 수는 있지만 전체 문장을 검수했다고 보지는 않는다. 전체900문항은 자동 집계·원문 탐색 자료를 제공했으며 **900개 모두의 의미 정확성을 직접 검수한 것은 아니다.**

E는 입력 근거, R은 규칙과 적용 조건, A는 계산·논리·선지 매핑을 평가했다. 0=핵심 오류 또는 설명 부재, 1=부분 정확/불충분, 2=해당 문항에 충분히 정확, ?=미확정이다. 설명 부재는 별도 표시하며 오답 추론으로 취급하지 않는다. 명시적으로 잘못을 고친 응답은 최종 적용에 부분/충분 점수를 줄 수 있지만, Rule에 남은 오류는 따로 기록했다. 이 평가는 assistant 정성검토이며 사람 전문가의 독립 검수가 아니다. 전문 판단이 어려운 조직 이미지·일부 문헌 해석은 미확정으로 남겼다.

'''
text+=table(['모델','표본 공식 정답','E 2점/판정 가능','R 2점/판정 가능','A 2점/판정 가능','E/R/A 모두2','정답이지만 관찰된 핵심 오류'],[[labels[m],f"{v['official_correct']}/30",*[f"{v['components'][c].get('2',0)}/{30-v['components'][c].get('unknown',0)}" for c in 'ERA'],f"{v['joint_all2']}/30",v['correct_with_observed_critical_error']] for m,v in s['summary'].items()])
text+='''
Baseline의7응답은 정답 선택만 있고 설명이 없어 E/R/A 0으로 표시했다. 이를 근거로 baseline 추론 능력이 낮다고 비교하면 안 된다. 이 작은 층화 표본의 점수는 900개 전체 내용 정확도의 추정치나 체크포인트 순위 검정으로 쓰지 않는다. 실제로 표본 정답 수에서는2epoch가 baseline보다 높지만 전체900에서는 낮다. 핵심 오류 수는 불완전한 검토로 발견한 사례 수이며 모든 오류를 셌다는 의미가 아니다.

### 실제 응답에서 확인된 패턴

**1. E가 틀리면, 맞는 R도 오답을 만든다 — Public_Health_9.** 2epoch는 특이도의 정의를 맞게 썼지만 표의 TCT 층 하나만 읽어 TN=245, FP=10으로 계산했다. 비질환군 전체를 합치면 TN=220+245=465, FP=10+25=35, 특이도465/500=.93이다. 4epoch도 같은 문제를 보였다. baseline은 공식 정답 A지만 설명 계산 역시 전체 표 집계와 맞지 않는다. [원문/이미지 비교](explorer.html#validation_Public_Health_9)

'''
text+=quote('Public_Health','epoch_2',next(r for r in rows if r['subject']=='Public_Health')['models']['epoch_2']['record']['model_response']['raw'])+'\n\n'
text+='''**2. 입력을 일부 정확히 읽어도 R의 적용 조건이 틀릴 수 있다 — Finance_10.** 3·4epoch는 미래 지불이면 선물환율을 쓰라는 규칙을 제시했다. 그러나 문제에는 선물환 계약 체결이 없고, 90일 뒤 실제환율0.50 CHF/$가 주어져 있다. 1,000,000/0.50=$2,000,000이어야 한다. 표의1.6460 자체는 잘 읽었지만 결정적인 본문 조건을 빠뜨렸다. [비교](explorer.html#validation_Finance_10)

'''+quote('Finance','epoch_3','Rule: To pay for watches in francs at a future date, use the forward rate to convert francs to dollars.')+'\n\n'
text+='''**3. 올바른 규칙과 잘못된 적용을 구분해야 한다 — Math_24.** 3epoch는 닮음 관계를 적었지만 최대 부피에 dr/dh=0이라고 하고 h=2H/3로 잘못 계산했다. 4epoch는 h=H/3, r=2R/3으로 정답을 회복했지만 부피식 분모 H²를 H로 누락했다. 그 오류는 h에 무관한 상수배라 최적점에는 영향이 없지만, 설명 전체를 정확하다고 하기는 어렵다. [비교](explorer.html#validation_Math_24)

'''+quote('Math','epoch_4','The volume is V = πr²h = πR²(H-h)²h/H.')+'\n\n'
text+='''**4. 정답을 맞혀도 E/R/A가 서로 모순될 수 있다 — Physics_15.** 4epoch는 두 반사 모두 위상 반전이라고 적으면서 반사 위상이 반대라고도 적고, 최종 λ/4를 선택했다. 실제로 낮은 굴절률→높은 굴절률 반사에만π가 붙으며, 아래 nf→n2 반사는 위상 반전이 없다. [OpenStax의 반사 위상 규칙](https://openstax.org/books/college-physics/pages/27-7-thin-film-interference)과 [원문 비교](explorer.html#validation_Physics_15).

'''+quote('Physics','epoch_4','Because n1 < nf and n2 < nf, both reflections are phase reversals.')+'\n\n'
text+='''**5. 규칙 문장은 외형이 그럴듯해도 사실이 반대일 수 있다 — History_14.** 2epoch는 중상주의를 수출 억제/수입 장려로 서술하고도 정답 C를 선택했다. 3·4epoch는 이 규칙을 바로잡았다. 중상주의의 수출 확대·수입 억제라는 원칙은 [OpenStax 교과서](https://openstax.org/books/world-history-volume-2/pages/5-3-the-mercantilist-economy)에서 확인했다. [비교](explorer.html#validation_History_14)

**6. 제대로 연결되는 ERA도 존재한다 — Accounting_29, Economics_12.** 동일 조건의 콜 가격10/12에서 B의 변동성이 더 높다고 연결하거나, 수출50−수입40=순수출10을 계산한 응답은 E/R/A가 일관됐다. ERA를 출력할 능력 자체가 없는 상태는 아니다. [회계](explorer.html#validation_Accounting_29) · [경제](explorer.html#validation_Economics_12)

**7. 정답 수용과 설명 타당성은 다르다 — Chemistry_4.** 그림의 치환기는1,4 위치의 trans 관계인데 baseline과2·3epoch는 위치/입체 관계를 잘못 설명한다. 4epoch는 trans를 맞췄지만 위치를 명확히 쓰지 않은 불완전한 이름이다. 공식 Judge는 네 응답을 정답으로 수용했다. 이 사례를 별도 주의 대상으로 기록하고 공식 점수는 변경하지 않았다. [이미지와 네 응답](explorer.html#validation_Chemistry_4)

**8. 선택지/라벨 검토가 필요한 사례.** Manage_26은 본문이 프로젝트 완료일을 묻지만 선택지는 작업별 슬랙 벡터다. 표에서 완료기간14, 슬랙(0,2,11,0,2,11,0)을 얻을 수 있으나 질문·선지 정합성 검토가 필요하다. Clinical_Medicine_8은 정답 C가 complement-mediated destruction인데 Rh 항체의 주된 적혈구 파괴는 혈관외 대식세포 경로라는 [NCBI 교과서 설명](https://www.ncbi.nlm.nih.gov/sites/books/NBK2269/?report=reader)과 세부 기전상 주의점이 있다. 태반 조직의 전문 판독과 원문 의도는 미확정이다. 이것이 후보들의 anti-receptor 선지를 정당화하지는 않는다. 두 사례 모두 공식 라벨은 유지했다.

## 분야별 결과

각 분야30문항. 괄호는 baseline 대비 정답 개수 변화다. 분야별 표본이 작으므로 분야 순위의 일반화는 제한된다. 탐색기의 분야/정오 변화 필터로 모든 해당 문항을 열 수 있다.

'''
subjects=a['metrics']['baseline']['by_subject']
text+=table(['분야','Baseline 정답/30','2epoch 정답(차이)','3epoch 정답(차이)','4epoch 정답(차이)','ERA 완성 개수 2/3/4'],[[sub,subjects[sub]['correct'],*[f"{a['metrics'][m]['by_subject'][sub]['correct']} ({a['metrics'][m]['by_subject'][sub]['correct']-subjects[sub]['correct']:+d})" for m in list(labels)[1:]],'/'.join(str(a['metrics'][m]['by_subject'][sub]['format_success']) for m in list(labels)[1:])] for sub in subjects])
text+='\n각 셀은 baseline 대비 **오답→정답 / 정답→오답** 개수다. 순증감 뒤에 숨은 문항 교체를 함께 확인한다.\n\n'
text+=table(['분야','2epoch 개선/퇴행','3epoch 개선/퇴행','4epoch 개선/퇴행'],[[sub,*[str(a['paired']['baseline_to_'+m]['by_subject'][sub]['wrong_to_right'])+' / '+str(a['paired']['baseline_to_'+m]['by_subject'][sub]['right_to_wrong']) for m in list(labels)[1:]]] for sub in subjects])
text+='''
## 다음 실험에 대한 판단

현재 증거는 **단순한 epoch 연장보다 E/R/A 내용과 연결 관계의 학습 신호를 개선하는 것이 우선**임을 지지한다. 데이터 증강은 해결 수단이 될 수 있지만, 같은 형식·같은 오류를 늘리는 것만으로 해결된다고 볼 수 없다. 이번200문항의 양/분야 범위와 목표 MMMU의 차이, 설명 정답성, 학습 강도를 함께 분리해 실험해야 한다.

1. E에는 이미지의 실제 값·단위·관계와 본문의 결정 조건을, R에는 적용 조건이 포함된 정확한 원리를, A에는 검산 가능한 대입·계산·선지 매핑을 요구하는 검수 기준을 먼저 고정한다. 세 단계 사이의 모순도 검사한다.
2. 표의 층 집계, 다이어그램 관계, 회로/투상, 다단계 계산 등 실패 유형을 보강하되, 학습 정답 설명을 독립 검수한다. 형식 전이용 데이터와 추론 정확성용 데이터의 효과를 구분한다.
3. 같은 문항·같은 예산의 일반 풀이 LoRA-SFT 대조군을 둔다. 이번에는 plain-SFT 대조군이 없어 ERA 방식 자체의 손해인지 SFT/데이터/학습률의 손해인지 식별할 수 없다.
4. 별도 검수 검증셋에서 정답률·완전 ERA율·E/R/A 내용 정확성·모순율을 같이 본다. 이번 MMMU900 오류를 학습 설계에 사용하면 이900개는 더 이상 최종 미관측 검증셋으로 쓰기 어렵다. 일부 checkpoint를 골랐다는 이유로 독립성을 주장하지 않는다.
5. Decoder LoRA 부착 위치가 원인이라는 증거는 아직 없다. 실제 생성되는 추론 텍스트도 decoder가 만든다. '추론 과정용 LoRA'를 다른 곳에 붙이면 해결된다고 가정하기보다, 검증된 데이터와 학습 강도를 먼저 통제한 뒤 attention/MLP/merger 등의 대상을 비교해야 한다. 이번 턴에서는 변경하거나 학습하지 않았다.

100% 형식 출력이 제품 요구라면 형식 제약·검사 후 재시도 같은 별도 장치도 검토할 수 있다. 다만 이는 이번의 동일 baseline 프롬프트 평가와 다른 조건이므로 별도 실험으로 보고해야 하며, 내용 정확성을 보장하지 않는다.

## 파일과 검증 한계

- [전체900 탐색기](explorer.html): 분야, 모델, 정오 변화, 형식 실패, 길이 상한, 반복, 검토30문항 필터. 네 모델 원문·공식 채점 기록·검토 근거를 연결했다. 이미지가 있는31장은 사전 선정 표본이며 나머지870문항 이미지는 이 자료에 포함하지 않았다.
- [900문항 색인](all900_index.json), [원본 전체 응답](evidence/pilot/all900_response_inventory.json), [120응답 내용 점수/인용](semantic_review.json), [상세 내용 검토 문서](semantic_review.md).
- [원래 계획](evidence/plan.json), [범위 변경 기록](evidence/scope_change_20261102.json), [평가 완료](evidence/pilot_completed.json), [원격 검증](evidence/pilot/verification.json), [로컬 독립 검증](local_verification.json), [전송 manifest](evidence/download_manifest.json).
- 원본 JSON/JSONL, Judge 시도 기록, manifest/launcher와 반복 결과는 evidence 폴더에 보존했다. 별도 작성한 내용 판정은 원본 점수에 덮어쓰지 않았다. 모든 분석 도구는 저장된 결과만 읽는 CPU 작업이다.

이번 결과는 한 번의200문항 학습 실행, 단일 추론 seed, 같은900문항에서의 비교다. 반복 일치는 재현성 확인이며 추가 독립 표본이 아니다. 표본30개 내용 점수에는 assistant 판단 오차가 있고 전문가 독립 검수가 필요하다. 출력 텍스트 분석은 모델 내부의 실제 인과적 추론 과정을 직접 관측한 것이 아니다.
'''
(R/'report.md').write_text(text,encoding='utf8')
detail=['# 사전 선정30문항 · 120응답 E/R/A 내용 검토','', '공식 점수는 변경하지 않았다. Assistant 정성검토이며 사람 독립검수가 아니다. 0=핵심 오류/설명 부재, 1=부분 정확, 2=충분 정확, ?=미확정. 긴 응답은 발췌 범위를 표시했다. 인용은 원문에서 추출한 앞부분이며 전체 문맥은 탐색기에서 확인할 수 있다.','']
for row in rows:
    detail+=['## '+row['id'],'',f"[원문·이미지·네 모델 비교](explorer.html#{row['id']})",'']
    for m in labels:
        r=next(x for x in s['annotations'] if x['id']==row['id'] and x['model']==m)
        scores='/'.join('?' if v is None else str(v) for v in r['scores'].values())
        detail+=[f"### {labels[m]} · 공식 {'정답' if r['official_correct'] else '오답'} · E/R/A {scores}",'',f"검토 범위: {'응답 전체' if r['coverage']=='full_response' else '앞/뒤 발췌; 전체 의미 검수 아님'} · {r['raw_response_chars']:,}자",'',r['rationale'],'']
        for q in r['quotes']:detail += [f"**{q['component']} 실제 인용**"+(' (뒤 생략)' if q['truncated'] else ''),'','> '+q['exact_quote'].strip().replace('\n','\n> '),'']
        if r['sources']:detail += ['확인 자료: '+' · '.join(f'[출처{i+1}]({url})' for i,url in enumerate(r['sources'])),'']
(R/'semantic_review.md').write_text('\n'.join(detail),encoding='utf8')
print('Reports written')
