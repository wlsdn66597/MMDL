# 공통 MMMU Test 평가 조건

두 브랜치는 같은 프롬프트·생성·채점 코드를 사용한다. 모델 선택과 기본 tag만 다르다.

| 항목 | 적용 값 |
| --- | --- |
| 데이터 | MMMU/MMMU test, 30과목, 10,500문항 |
| 문항 유형 | 객관식 9,873 / 주관식 627 |
| 데이터 revision | 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 |
| 원본 Baseline | Qwen/Qwen3-VL-4B-Instruct @ ebb281ec70b05090aa6165b016eac8ec08e71b17 |
| 생성 seed | 기본 42, 실행 인자로 변경. 과거 동일-seed 비교는 3407 |
| 생성 | BF16, vLLM 0.11.0, 출력 상한 32768, 문맥 64000, 동시 처리 16, batched tokens 4096 |
| 샘플링 | temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.0, presence penalty 1.5 |
| 이미지 | RGB, min/max pixels 1003520 / 4014080, 최대 7장, 비율 유지 |
| 반복 중단 | 4096토큰 간격의 최근 구간, 12-gram 반복률 0.5 이상 2회 연속. 중단은 오답 |
| Judge | Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218 |
| Judge 생성 | seed 3407, temperature 0, 출력 상한 4096, thinking 끔, 필요한 문항당 1회 |
| Judge 실행 | 문맥 32768, GPU memory utilization 0.92, 나머지는 원본 judge.py/vLLM 기본값 |

## 원본 설정 파일을 읽는 방법

`eval_settings.json`은 원본을 그대로 보존해 validation/900, 생성 seed 3407, 동시 처리 1, 최대 이미지 5, Judge max_attempts 25 같은 템플릿 값이 남아 있다.

실제로는 `build_records.py mmmu-test`가 test/10500·최대 이미지 7로 변환하고, 실행기가 생성 seed·동시 처리 16·출력 상한 32768·반복 중단을 명시한다. Judge 코드는 필요한 문항당 한 번 실행한다. 파일 전체의 3407을 42로 일괄 변경하면 Judge seed까지 바뀌므로 그렇게 수정하지 않는다. 실제 적용 값은 실행 manifest의 `identity.evaluation`에 기록된다.

## 프롬프트

객관식은 질문과 보기를 제공하고 마지막 줄을 `Final answer: LETTER`로, 주관식은 `Final answer: ANSWER`로 쓰도록 요청한다. system 메시지는 없다. 생성기는 고정 문항의 `rendered_chat`을 그대로 사용한다.

문항 전체 해시:

```text
f826706492339a5e0c9181e948838fae1aa9bb30f9781973adb61f399f5311d6
```

## 출처와 보존 범위

제공된 `mmmu_eval_base`의 설정·환경·문항 생성·다운로드·diag 코드와 `run_eval_dpo2e.sh`를 바이트 단위로 복사했다. 이 파일 9개의 SHA256은 `protocol_reference.json`에 있다. 원본 `run_eval.sh`에는 모델 경로 옵션 앞 공백 누락이 있어, 실제 DPO-E 실행에 사용했던 수정본 `run_eval_dpo2e.sh`를 두 브랜치 공통으로 사용한다. Baseline은 모델 인자를 비워 원래 고정 저장소/revision 경로를 유지한다.

환경 준비·평가 진입점·실행 기록·검증 래퍼를 추가했다. 파서와 Judge의 알려진 채점 한계도 원본과 동일하다. 이를 고쳐 평가할 경우 두 모델과 모든 seed에 같은 수정 기준을 적용하고 기존 점수와 구분해 보고해야 한다.

## 범위

이 묶음은 평가 코드다. DPO-E/F 학습 코드나 학습 데이터셋 카드가 아니며 모델 가중치·학습 데이터·MMMU 문항 원문·이미지·실제 응답을 배포하지 않는다. 공개 저장소에 올릴 때 원 코드·모델·데이터의 출처와 사용 조건은 그대로 유지한다.
