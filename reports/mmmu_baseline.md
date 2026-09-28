# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 두 번 말하게 하지마
- **팀원**: 김진우, 김선재, 안영찬, 윤희성
- **작성일**: 2026-09-28
- **재현 커맨드**:
    1. 환경 설치

    git clone https://github.com/wlsdn66597/MMDL.git   
    cd MMDL   
    python3.10 -m venv .venv   
    source .venv/bin/activate   
    python -m pip install --upgrade pip   
    python -m pip install -r requirements.txt   
    python -c "import mc_parser, eval_output_policy; print('local modules: OK')"   

    2. 실행

    MODEL_SNAPSHOT=""   
    MMMU_SNAPSHOT=""

    bash scripts/run_mmmu_val_baseline.sh \
    --model-path "$MODEL_SNAPSHOT" \
    --data-root "$MMMU_SNAPSHOT" \
    --output-root results/mmmu_val_two_stage4096


---

## 1. 환경 / 재현성

| 항목 | 값 |
| --- | --- |
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | vLLM 0.28.0 |
| 사용 GPU | NVIDIA GeForce RTX 4090 24GB |
| 실측 peak VRAM | 21.54 GiB |
| 총 소요 시간 | 8,331초 |
| 의존성 | [requirements.txt](../requirements.txt) |
| 실행 커맨드 | `bash scripts/run_mmmu_val_baseline.sh --model-path "$MODEL_SNAPSHOT" --data-root "$MMMU_SNAPSHOT" --output-root results/mmmu_val_two_stage4096` |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

객관식 1단계

```
Question: {question}

Choices:
(A) {option_A}
(B) {option_B}
{additional_choice_lines_if_any}

Work out the problem briefly, using the supplied images. Identify the relevant visual evidence and perform only the necessary calculations. Avoid repeating alternatives. This is a working draft; a separate step will select the final option.
```

객관식 2단계

```
user: {stage1_user_content}

assistant: {working_draft}

user: Now select the single best option for the ORIGINAL question using the original images and the working draft above. The draft may be incomplete or incorrect; do not blindly copy it. Output only one valid option letter.
```

주관식 1단계

```
Question: {question}

Work out the problem briefly, using the supplied images. Identify the relevant visual evidence and perform only the necessary calculations. Avoid repeating alternatives. This is a working draft; a separate step will produce the final answer.
```

주관식 2단계

```
user: {stage1_user_content}

assistant: {working_draft}

user: Now answer the ORIGINAL question using the original images and the working draft above. The draft may be incomplete or incorrect; do not blindly copy it. Output only the concise final answer, including units if needed. Do not include an explanation.
```

- stage1_user_content는 해당 문항의 1단계 user 입력 전체로, 이미지 번호·이미지·질문·초안 지시문과 객관식의 경우 선택지까지 포함
- working_draft는 1단계 모델 호출에서 생성한 풀이 초안을 의미
- **출처**: 직접 설계

    방법론적으로는 Kojima et al. (2022)의 *"Large Language Models are Zero-Shot Reasoners"* (arXiv:2205.11916) 논문에서 정립된 Zero-shot-CoT (초안 생성 후 정답 추출의 2단계 프롬프팅) 방식과, Cobbe et al. (2021) 논문의 다중 추론-검증 패러다임을 아카이브에서 직접 리서치하여 파이프라인에 차용.

- **선택 이유**: 시각적 근거 파악 및 다단계 계산이 필수적인 MMMU 벤치마크 특성상, 모델이 한 번의 호출로 정답을 출력하도록 강제하면 중간 논리 전개가 누락되어 환각 및 오답을 유발하기 쉽습니다. 이에 1단계에서 자유로운 시각 증거 파악 및 계산(초안)을 유도하고, 2단계에서 해당 초안을 바탕으로 제한된 정답 형식만 출력하게 유도하는 이중 호출(Two-stage) 구조를 분리 적용하여 파싱 정확도와 추론 성능을 동시에 극대화하고자 했습니다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
| --- | --- |
| `do_sample` | True |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 3407 |
- **출처**: https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction

    두 단계 호출 모두 기본적으로 Qwen 공개 평가 설정의 파라미터 값(temperature, top-p, top-k, penalty)을 따릅니다. 다만, 2단계 객관식 도출 시 유효 선택지 문자만 허용하는 제한 디코딩을 추가 적용해 공식 자유 생성 설정과 출력 분포의 차이를 두었습니다.


### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
| --- | --- |
| 1단계 `max_new_tokens` | 4096 |
| 2단계 `max_new_tokens` (객관식 / 주관식) | 16 / 128 |
| `max_model_len` | 16384 |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels`) | 1,003,520 / 4,014,080 |

**선택 근거**

RTX 4090 (24GB VRAM) 환경 인프라 제약과 시간적 트레이드오프를 고려했습니다. 1단계는 이미지 근거 확인과 계산을 포함한 풀이 초안을 생성하므로 4096토큰을 초기 기준으로 설정했습니다. 이후 동일한 문항에서 8192토큰과 비교했을 때 정확도 증가는 0.22%p(62.78% → 63.00%)였으나 추론 시간은 약 136.5분에서 223분으로 늘었습니다. 이에 이번 실험에서는 4096토큰을 실행 비용을 고려한 기준선으로 유지했습니다.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: 외부 judge API 없이 로컬에서 평가했습니다. 객관식은 vLLM의 선택지 제한 디코딩을 사용하고, 주관식은 저장소에 포함한 MMMU 평가 함수 `parse_open_response`와 `eval_open`을 사용했습니다.
- 동작 방식 요약:
    - **객관식:** 2단계에서 해당 문항의 유효 선택지 문자 하나만 생성하도록 제한. 생성 결과에 `strip()`을 적용한 뒤 유효 문자와 정확히 일치하는지 검증하고, 정답 문자와 비교. 형식이 어긋나면 임의로 답을 추측하거나 오답으로 보정하지 않고 실행을 중단.
    - **주관식:** 1단계 초안은 채점하지 않음. 2단계 최종 답만 MMMU 평가 함수로 파싱·채점. 최종 출력이 비었거나 길이 제한으로 잘렸거나 파싱할 수 없으면 미파싱·오답으로 처리.

## 5. 결과

| No. | Subject | Data Num | Acc |
| --- | --- | --- | --- |
| 1 | Accounting | 30 | 70.0 |
| 2 | Agriculture | 30 | 53.3 |
| 3 | Architecture_and_Engineering | 30 | 53.3 |
| 4 | Art | 30 | 63.3 |
| 5 | Art_Theory | 30 | 80.0 |
| 6 | Basic_Medical_Science | 30 | 73.3 |
| 7 | Biology | 30 | 36.7 |
| 8 | Chemistry | 30 | 46.7 |
| 9 | Clinical_Medicine | 30 | 70.0 |
| 10 | Computer_Science | 30 | 63.3 |
| 11 | Design | 30 | 73.3 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 43.3 |
| 13 | Economics | 30 | 83.3 |
| 14 | Electronics | 30 | 50.0 |
| 15 | Energy_and_Power | 30 | 60.0 |
| 16 | Finance | 30 | 66.7 |
| 17 | Geography | 30 | 43.3 |
| 18 | History | 30 | 76.7 |
| 19 | Literature | 30 | 83.3 |
| 20 | Manage | 30 | 70.0 |
| 21 | Marketing | 30 | 76.7 |
| 22 | Materials | 30 | 60.0 |
| 23 | Math | 30 | 63.3 |
| 24 | Mechanical_Engineering | 30 | 43.3 |
| 25 | Music | 30 | 30.0 |
| 26 | Pharmacy | 30 | 70.0 |
| 27 | Physics | 30 | 60.0 |
| 28 | Psychology | 30 | 73.3 |
| 29 | Public_Health | 30 | 83.3 |
| 30 | Sociology | 30 | 63.3 |
|  | **Overall (macro avg)** | **900** | **62.8** |

계산식: `Overall = mean(30개 과목 accuracy, 소수점 둘째 자리에서 반올림한 결과)`

## 6. 공식 수치와의 비교

|  | Overall (MMMU val) |
| --- | --- |
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 62.8 |
| 차이 (Δ) | -4.6 |

## 7. 격차 분석

본 실험의 정확도는 공개 수치 67.40%보다 4.62%p 낮은 62.78%였습니다. 이에 대한 원인은 크게 세 가지로 추정됩니다.   
첫째, 1단계 초안 출력에서 120문항이 4096토큰 제한에 걸렸고 이 문항들의 정답률은 40.83%로 정상 종료된 문항들의 정답률(66.15%)보다 낮았습니다. 다만 토큰을 8192로 늘려도 전체 정답은 2건만 증가해, 출력 길이만으로 격차를 해소하기는 어렵다고 판단했습니다.   
둘째, 정상 종료한 초안에서도 오답이 264건 발생했습니다. 선별한 오답을 원본 이미지와 대조한 결과, 형광 영상의 특징이나 그래프의 꼭짓점 수를 잘못 읽은 사례(Biology_24, Math_3)와 화학 구조를 말토스 규칙에 잘못 연결하거나 그래프에 포화 수송 규칙을 반대로 적용한 사례(Biology_29, Pharmacy_18)등이 있었습니다. 즉 초안이 끝까지 생성되더라도 시각 근거를 판독하거나 이를 전문 규칙에 연결하는 과정에서 오류가 나면 잘못된 풀이로 이어져 정확도가 낮아지는 것으로 판단했습니다.   
셋째, 2단계는 초안을 다시 입력받아 객관식은 선택지 문자 하나, 주관식은 짧은 답만 생성합니다. 초안을 그대로 복사하지 말라는 지시는 있지만, 근거·계산·결론이 원본 문제와 일치하는지 확인하도록 명시적인 절차를 두지는 않았습니다. 결과적으로 초안에서 선택지 답을 추출할 수 있었던 492건 중 490건에서 최종 답이 같았고 그중 156건은 오답이어서, 잘못된 초안 결론이 최종 답에 남은 것으로 판단했습니다. 주관식의 경우, 오답 34건 중 30건에서 초안은 정상 종료되었고 최종 답 파싱 실패도 없었습니다. 따라서 주관식은 출력 형식보다 초안의 계산과 최종 답 산출에서 오류가 났다고 판단하였습니다.

## 8. 기타 특이사항 / 한계 (Optional)

초기 목표는 학습 전 MMMU-val 베이스라인을 재현 가능한 형태로 확보하는 것이었습니다. 전체 900문항을 고정된 2-stage 설정으로 평가하고 입력·예측 ID와 채점을 점검했지만, 최종 답 미파싱이 없는 상태에서도 공개 수치와 차이가 남았습니다. 정답률만으로는 그 이유를 설명하기 어려워 저장된 초안과 최종 답을 원본 이미지에 대조했습니다. 그 결과 이미지 순서·참조 문제만으로 설명하기 어려운 시각 구조 판독과 전문 규칙 적용 오류, 초안의 잘못된 결론이 최종 답에 남는 사례를 확인했습니다. 이를 바탕으로 베이스라인 재현 이후의 연구 방향을 이미지 배치 변경보다 시각 근거 활용과 초안 검토 과정의 개선으로 구체화했습니다.   
후속 연구에서는 기존 2-stage 흐름을 유지하며, 질문과 관련된 시각 특징을 전달하는 게이트 G와 초안 작성·최종 검토를 각각 맡는 LoRA A·B를 학습할 예정입니다. 같은 평가 조건에서 정확도와 오답 초안의 수정 여부, 추론 비용을 비교할 예정입니다.
