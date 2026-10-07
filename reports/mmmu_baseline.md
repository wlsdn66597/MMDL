# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 두 번 말하게 하지마
- **팀원**: 김진우, 김선재, 안영찬, 윤희성
- **작성일**: 2026-10-08 (KST)
- **평가 프롬프트**: 공식 기본 문구 + 마지막 줄 Final answer 지시
- **재현 커맨드**: `.venv/bin/python src/evaluate.py --config src/config.json --run-name baseline`

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 데이터 | `MMMU/MMMU`, revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, validation 30과목 × 30문항 |
| 추론 백엔드 | vLLM 0.11.0, offline inference |
| 사용 GPU | NVIDIA GeForce RTX 4090 24GB, TP=1 |
| Python·주요 패키지 | Python 3.12, torch 2.8.0 (CUDA 12.8), transformers 4.57.0, qwen-vl-utils 0.0.14, datasets 4.1.1 |
| dtype·양자화 | BF16, 양자화 없음 |
| 실측 peak VRAM | 별도 계측하지 않음 |
| 총 소요 시간 | 900문항별 경과 시간 합 49,513.468초(13.75시간). 엔진 초기화·저지를 포함한 종단간 시간은 별도 미계측 |
| 의존성 | [requirements.txt](../requirements.txt), 실제 기준 환경의 패키지 버전 고정 |

vLLM을 사용해 공식 샘플링과 멀티모달 입력을 같은 백엔드에서 처리했다. 4090 메모리에 맞춰 context=64000, gpu_memory_utilization=0.90, max_num_seqs=1, max_num_batched_tokens=4096, enforce_eager=true를 사용한다. CUDA graph 프로파일링 메모리를 줄이고 한 문항씩 실행한다. 실제 최대 이미지 5장으로 상한을 설정하며 공식 일반 상한 10과 다르다.

```bash
# 최초 준비: 가상환경·고정 패키지·모델·데이터
python3.12 src/prepare.py --model-cache /path/to/model-cache --dataset-cache /path/to/dataset-cache

# 전체 평가: 추론 → 파서 → 필요한 저지 → 최종 표
.venv/bin/python src/evaluate.py --config src/config.json --run-name baseline \
  --model-cache /path/to/model-cache --dataset-cache /path/to/dataset-cache

# 학습 후 같은 평가, 병합된 BF16 체크포인트만 변경
.venv/bin/python src/evaluate.py --config src/config.json --run-name finetuned \
  --checkpoint /path/to/merged-bf16-checkpoint
```

캐시는 표준 Hugging Face 방식으로 재사용한다. 코드에 사용자 서버 절대 경로를 고정하지 않는다. 추론·저지 프로세스를 순차 실행해 평가 모델이 해제된 후 저지 모델을 올린다. 같은 설정으로 재개할 때 완료된 응답을 재생성하지 않는다. 코드·설정·체크포인트·원문이 다른 실행의 혼용을 거부한다.

## 2. 프롬프트

객관식의 실제 텍스트 전문:

```text
{optional_hint}Question: {question}
Options:
{lettered_options}Please select the correct answer from the options above.
On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.
```

주관식:

```text
{optional_hint}Question: {question}
On the last line, write "Final answer: ANSWER", replacing ANSWER with a concise final answer.
```

`optional_hint`는 데이터에 hint가 있을 때만 `Hint: {hint}
`이다. `lettered_options`는 A부터 모든 보기를 `A. {option}
` 형태로 연결한다. 데이터의 `<image n>` 표지와 질문·보기는 그대로 유지한다. 존재하는 image_1→image_7의 이미지를 RGB로 변환해 user 메시지의 텍스트 앞에 배치한다. 별도 system 메시지·CoT·설명 금지 지시는 없다. 모델 processor의 `apply_chat_template(..., add_generation_prompt=True)`를 사용한다.

- **출처**: [공식 build_mmmu_prompt](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)의 기본 문구를 차용하고 마지막 줄 지시를 직접 추가했다. 공식 원본 그대로인 프롬프트라고 주장하지 않는다.
- **선택 이유**: 풀이를 허용하면서 실제 최종 답을 명시하도록 하여 본문 선택지 언급과 최종 답을 구분하고 로컬 저지 의존을 줄인다. 최종 template은 [src/config.json](../src/config.json)에 고정한다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---:|
| do_sample | true (vLLM의 temperature>0) |
| temperature | 0.7 |
| top_p | 0.8 |
| top_k | 20 |
| repetition_penalty | 1.0 |
| presence_penalty | 1.5 |
| seed | 3407 (engine와 모든 문항 요청) |

- **출처**: [공식 MMMU Instruct 실행 설정](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/infer_instruct.sh). 공개 MMMU runner의 engine 기본 seed는 42이며, 요청별 seed를 따로 지정하지 않는다. 본 파이프라인은 중단·재개 시 요청 난수 조건을 유지하도록 engine와 요청에 3407을 지정한다. 이 차이는 문서화하며 하드웨어 간 비트 단위 동일 출력을 보장하지 않는다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---:|
| max_new_tokens / vLLM max_tokens | 32768 |
| min_pixels | 1003520 |
| max_pixels | 4014080 |
| max_model_len | 64000 |
| 이미지 상한 | 5장 |
| 추가 stop 문자열·토큰 | 없음 |

생성·픽셀 예산은 공개 MMMU Instruct 설정을 기준으로 한다. 긴 답의 잘림을 줄이는 대신 추론 시간이 늘어난다. 실제 81개 응답이 32768에서 종료되어 종료 사유와 토큰 수를 보존했다. 종횡비를 유지하고 patch=16, merge=2에 맞춰 32의 배수로 smart_resize한다. qwen-vl-utils로 전처리한 뒤 vLLM에는 do_resize=false를 주어 중복 resize를 막는다. 입력+출력이 context를 넘으면 예산을 임의로 줄이지 않고 오류로 중단한다. 평가 응답의 최대 입력은 5647토큰이다.

4090의 KV cache 메모리 예산에 맞춰 context=64000을 사용한다. 공식 runner 기본 128000과 다르다.

## 4. 채점(파싱) 방식

자체 최종 답 파서와 저지 출력 파서를 사용한다. 공식 문자·보기 문구 추출 함수는 [eval_utils.py](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/eval_utils.py)의 원문을 복사하고 출처 commit과 hash 및 라이선스를 보존했다.

1. `finish_reason=length` 81개는 답 추출 여부와 무관하게 오답. 한도 초과 응답에서 본문 답을 잡아 정답으로 만드는 것을 막는 자체 정책이다.
2. 마지막 `</think>` 이후에서 Markdown 서식을 정리하고 명시적 Final answer 필드를 읽는다. 동일 답 반복은 허용하고 충돌·복수 보기·최종 필드 뒤 실질 텍스트는 저지로 보낸다.
3. 최종 필드가 없을 때만 공식 can_infer를 전체 응답에 적용한다. 주관식 최종 텍스트도 공식 매칭 규칙을 사용하며 추가 숫자 허용오차는 없다.
4. 실패한 응답은 Qwen3-8B(`b968826d9c46dd6066d109eabc6255188de91218`)에 공식 질문·보기·응답 매칭 프롬프트로 전달한다. 입력은 전체 응답의 `</think>` 이후이고 이미지·추가 풀이 정답 생성 지시는 없다.
5. 저지 출력의 마지막 Your output:, Output:, Final answer: 또는 단독 문자를 우선 읽고 공식 추출을 fallback으로 유지한다. Z는 정상 no-match 결정으로 오답 종료한다. 무효 출력은 최대 25회 후 오답이며 무작위 대체는 없다.

저지는 BF16·양자화 없음, temperature=0, top_p=1, top_k=-1, repetition_penalty=1, presence_penalty=0, seed=3407, max_tokens=4096, enable_thinking=false이다. 저지 context=32768이며 입력을 자르지 않는다. 공식 GPT judge를 로컬 모델로 대체했으므로 공식 리포트와 동일 채점이라고 주장하지 않는다.

주관식은 공식 전처리처럼 채점에서만 A=참조 정답, B=Other Answers로 구성한다. 참조가 대문자 한 글자이면 공식 GT 처리 특례를 그대로 유지한다. 실제 주관식 답과 채점용 보기 코드는 구분해서 저장한다. 모든 저지 프롬프트 전문은 config, 시도별 입력·출력은 judge_attempts.jsonl에 있다.

## 5. 결과

| No. | Subject | Data Num | Correct | Acc |
|---:|---|---:|---:|---:|
| 1 | Accounting | 30 | 19 | 63.33% |
| 2 | Agriculture | 30 | 15 | 50.00% |
| 3 | Architecture_and_Engineering | 30 | 11 | 36.67% |
| 4 | Art | 30 | 18 | 60.00% |
| 5 | Art_Theory | 30 | 24 | 80.00% |
| 6 | Basic_Medical_Science | 30 | 23 | 76.67% |
| 7 | Biology | 30 | 17 | 56.67% |
| 8 | Chemistry | 30 | 16 | 53.33% |
| 9 | Clinical_Medicine | 30 | 23 | 76.67% |
| 10 | Computer_Science | 30 | 20 | 66.67% |
| 11 | Design | 30 | 24 | 80.00% |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 10 | 33.33% |
| 13 | Economics | 30 | 26 | 86.67% |
| 14 | Electronics | 30 | 19 | 63.33% |
| 15 | Energy_and_Power | 30 | 14 | 46.67% |
| 16 | Finance | 30 | 23 | 76.67% |
| 17 | Geography | 30 | 18 | 60.00% |
| 18 | History | 30 | 23 | 76.67% |
| 19 | Literature | 30 | 25 | 83.33% |
| 20 | Manage | 30 | 18 | 60.00% |
| 21 | Marketing | 30 | 26 | 86.67% |
| 22 | Materials | 30 | 15 | 50.00% |
| 23 | Math | 30 | 18 | 60.00% |
| 24 | Mechanical_Engineering | 30 | 11 | 36.67% |
| 25 | Music | 30 | 6 | 20.00% |
| 26 | Pharmacy | 30 | 21 | 70.00% |
| 27 | Physics | 30 | 25 | 83.33% |
| 28 | Psychology | 30 | 24 | 80.00% |
| 29 | Public_Health | 30 | 26 | 86.67% |
| 30 | Sociology | 30 | 18 | 60.00% |
| | **Overall (macro avg)** | **900** | **576** | **64.00%** |

계산식: `Overall = mean(30개 과목 accuracy)`. 모든 과목이 30문항이므로 전체 576/900과 동일하다.

| 채점 경로 | 수 | 정답 |
|---|---:|---:|
| explicit_final_parser | 759 | 539 |
| output_limit_incorrect | 81 | 0 |
| local_qwen3_8b_judge_parser | 54 | 33 |
| official_parser_fallback | 6 | 4 |

저지 54개 중 유효 보기 40개(정답 33), Z 13개, 무효 문자 1개이다. 무작위 대체는 0개이다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| 공식 — 과제에서 지정한 Qwen3-VL Technical Report 기준 | 67.40% |
| 우리 저장 응답의 최종 채점 | 64.00% |
| 차이 (Δ) | −3.40%p |

## 7. 격차 분석

현재 점수는 576/900(64.00%)으로 과제 기준 공식 67.4%보다 3.4%p 낮다. 공식 수치의 완전 재현은 아니다. 우리는 고정 HF validation을 사용하며, 사전 대조에서는 공식 TSV와 질문·보기·정답이 다른 11문항이 기록됐다. 공식 프롬프트에 최종 답 형식 지시를 추가했고, seed·4090 실행 설정·로컬 Qwen3-8B 저지도 공식 runner와 다르다.

32768토큰에 도달한 응답은 81개(9.0%)이며 답 추출 여부와 무관하게 전부 오답 처리했다. 토큰 한도가 평가에 영향을 주는 직접 증거지만, 예산을 늘렸을 때 맞힐 문항 수는 알 수 없다. 파서·저지 출력 처리도 점수에 영향을 줬다. 현재도 실제 답과 채점의 불일치 2개 및 정답 집계 중 저지 검토 후보 3개가 남아 있어 점수의 한계로 공개한다.

오답 324개는 내용 오류 패턴 172개, 출력 한도 81개, 근거 부족 48개, 확인된 저지 오류 2개, 입력·참조 검토 21개로 구분했다. 이것은 응답의 관찰 패턴이며 공식과의 격차 원인별 기여도나 신경망 모듈 원인의 증명이 아니다. 현재 자료만으로 3.4%p를 원인별로 배분하지 않는다.

## 8. 기타 특이사항 / 한계

- 입력·채점 대조, 파서 규칙 검사, 소규모 GPU 추론·저지 검증을 통과했다. 전체 900문항 새 추론 및 새 환경 전체 설치는 검증하지 않았다.
- 보고서 점수는 저장된 900문항 응답과 보존된 저지 출력을 채점한 결과다. 재실행은 유효 저지 결정 또는 Z가 나오면 종료하므로 보존 출력과 결과가 달라질 수 있다. 학습 효과 비교에는 베이스라인·학습 모델을 같은 코드·설정으로 실행해야 한다.
- 확인된 저지 오답 판단 2개(Architecture_and_Engineering_24, Electronics_27)와 정답 집계 중 수치 판정 검토 후보 3개(Electronics_6, Electronics_21, Electronics_24)는 현재 점수에 수동 보정하지 않았다.
- 고정 HF 데이터·공식 TSV, 프롬프트, seed, context, 로컬 judge 및 자체 한도 정책이 달라 완전한 공식 리포트 재현이 아니다.
- 가중치·데이터·이미지·대용량 응답은 Git에 넣지 않는다.
