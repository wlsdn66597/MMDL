# AIX-408-06 MMMU Test 평가 계약

2026-11-21 DPO2-E 재현 모델 seed 42 실행에서 사용한 코드와 설정을 기준으로 한다. 다른 모델의 효과를 비교할 때 모델 경로와 실행 이름만 바꾼다.

| 항목 | 실제 적용 값 |
|---|---|
| 데이터 | `MMMU/MMMU` Test, 30과목, 총 10,500개 고유 ID |
| 데이터 revision | `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` |
| 유형 | 객관식 9,873 / 주관식 627 |
| 문항 파일 SHA256 | `f826706492339a5e0c9181e948838fae1aa9bb30f9781973adb61f399f5311d6` |
| 기준 processor | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 학습 모델 | 로컬 BF16 전체 병합 모델, 양자화 없음 |
| 생성 seed | 기본값 **42**. 실행기의 세 번째 인자로 지정한 값을 엔진과 각 요청에 적용 |
| temperature / top-p / top-k | 0.7 / 0.8 / 20 |
| repetition / presence penalty | 1.0 / 1.5 |
| 출력 상한 / 문맥 | 32,768 / 64,000토큰 |
| 동시 처리 / batched tokens | 16 / 4,096 |
| 생성 엔진 | vLLM 0.11.0, BF16, eager, GPU memory utilization 0.9 |
| 이미지 | RGB, min pixels 1,003,520 / max pixels 4,014,080, 최대 7장 |
| 전처리 | qwen-vl-utils 0.0.14 smart_resize, 비율 유지, factor 32, patch size 16; vLLM 재리사이즈 끔 |
| 프롬프트 | 고정 records의 input_prompt와 rendered_chat 그대로 사용; system 메시지 없음 |
| 반복 중단 | 매 4,096토큰, 최근 구간의 12-gram 반복률 ≥0.5가 2회 연속이면 중단, 오답 |
| 보조 Judge | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` |
| Judge 생성 | **seed 3407**, temperature 0, top-p 1, top-k -1, 출력 4,096, thinking 끔 |
| Judge 실행 | 문맥 32,768, GPU memory utilization **0.92**, 나머지는 원본 judge.py와 vLLM 0.11.0 기본값 |
| Judge 횟수 | 필요한 문항당 1회; 설정 템플릿의 max_attempts=25를 사용하지 않음 |
| 평가 순서 | 전체 생성 → 필요한 응답만 Judge → 전체 채점 → 요약 → 저장된 응답 재채점 검증 |

## 문항과 프롬프트

객관식은 선택지 A, B, ...를 포함하며 마지막에 다음 문장을 붙인다.

```text
On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.
```

주관식 마지막 지시는 다음과 같다.

```text
On the last line, write "Final answer: ANSWER", replacing ANSWER with a concise final answer.
```

hint가 있으면 원본 템플릿에 따라 포함한다. 질문과 선택지의 순서, 모든 이미지, chat template를 포함한 완성 문항 파일 전체를 해시로 검증한다. 모델의 tokenizer와 이미지 processor 파일도 기록한다. 평가 대상이 다른 tokenizer나 구조로 변경되면 동일 입력 조건을 보장할 수 없으므로 이 계약의 대상은 같은 Qwen3-VL-4B 계열 모델이다.

## 채점과 결과

마지막 Final answer 필드와 원본 fallback 규칙을 적용한다. 해석하지 못한 응답만 Qwen3-8B에게 보기를 대응시키게 한다. 주관식 역시 원본 문자열·특례 규칙과 Judge를 사용한다. 루프 중단·출력 길이 제한은 오답이다. 이 규칙과 파서는 현재 서버 평가와 동일한 소스 파일로 고정했다.

이 브랜치에서 만든 `<tag>.manifest.json`은 실제 모델 전체 파일 해시, 코드 해시, 문항 해시, 생성·Judge seed, 원본 문항 설정, 설치 패키지, GPU·드라이버를 기록한다. CLI 덮어쓰기 값이 `identity.evaluation`에 따로 있으며, 템플릿의 sampling.seed=3407을 실제 생성 seed로 읽으면 안 된다.

검증기는 GPU를 다시 호출하지 않고 저장된 Judge 결과와 원본 채점기로 결과를 대조한다. 이는 Judge 추론을 반복하거나 모든 답을 사람이 확인하는 검증이 아니다.

## 파일 출처

`eval_settings.json`, `env/requirements_eval.txt`, `scripts/build_records.py`, `scripts/download.py`, `scripts/diag/*.py`는 기존 408-06 Test baseline 평가 묶음에서 복사했다. `scripts/run_eval_dpo2e.sh`는 DPO2-E 실행에 쓰인 원본이며, baseline 실행기의 옵션 앞 누락된 공백 하나를 고친 버전이다. 파일별 SHA256은 `protocol_reference.json`에 기록했다. 원본 주석에 val/진단용이라고 적힌 부분은 파일 보존을 위해 그대로 두었으며, 실제 Test 설정은 위 표와 생성된 records가 정한다.

추가한 prepare_test.sh, cache_test.py, run_test.sh, protocol.py는 배포·검증용이다. 생성 및 채점 로직을 교체하지 않는다. 기존 저장소의 제3자 코드 출처는 [THIRD_PARTY_LICENSES.txt](../../../THIRD_PARTY_LICENSES.txt)를 따른다. 모델과 데이터의 이용 조건은 각 배포처를 따른다.
