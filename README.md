# Qwen3-VL-4B MMMU 평가

실험 3을 기준으로 고정한 평가 파이프라인입니다. 준비 후 명령 하나로 900문항 추론, 최종 답 파서, 공식 규칙 fallback, 로컬 Qwen3-8B 저지, 최종 JSON 및 30과목 점수 표까지 실행합니다. 제출 보고서는 [reports/mmmu_baseline.md](reports/mmmu_baseline.md)에 있습니다.

## 준비

Linux x86-64(Ubuntu 또는 NVIDIA GPU 사용이 가능한 WSL2), Python 3.12, BF16을 지원하는 NVIDIA GPU와 드라이버가 필요합니다. 기준 GPU는 RTX 4090 24GB입니다. Python과 드라이버는 사전 설치 조건이며 스크립트가 시스템 드라이버를 설치하지 않습니다. vLLM GPU 추론은 Windows 네이티브 환경을 대상으로 하지 않습니다.

```bash
git clone https://github.com/yunheeseong1/mmdl_test.git
cd mmdl_test
python3.12 src/prepare.py
```

`prepare.py`는 `.venv`를 만들고 `requirements.txt`의 고정 패키지를 설치한 뒤, 지정 revision의 평가 모델·저지 모델·MMMU 30과목 validation을 Hugging Face 캐시에 준비합니다. 모델 파일은 이미 있는 캐시를 재사용하고 가중치 hash 및 준비 정보를 `outputs/preparation.json`에 기록합니다. 패키지 목록은 실제 평가 환경의 `pip freeze`를 고정한 것입니다. 다운로드에는 인터넷과 충분한 저장 공간이 필요합니다.

캐시는 `src/config.json`, 다음 옵션 또는 환경변수로 지정합니다. `MMMU_MODEL_CACHE`와 `MMMU_DATASET_CACHE`를 쓰면 준비와 평가가 같은 캐시를 공유합니다.

```bash
python3.12 src/prepare.py --model-cache /path/to/model-cache --dataset-cache /path/to/dataset-cache

# 이미 설치한 환경에서는 설치를 반복하지 않고 검증·캐시 재사용
/path/to/venv/bin/python src/prepare.py --existing-env --offline \
  --model-cache /path/to/model-cache --dataset-cache /path/to/dataset-cache

# 버전·GPU만 점검, 다운로드·가중치 hash 검사는 생략
.venv/bin/python src/prepare.py --existing-env --check-only
```

## 전체 평가: 한 명령

```bash
.venv/bin/python src/evaluate.py --config src/config.json --run-name baseline
```

평가 모델과 데이터는 준비된 고정 캐시를 오프라인으로 사용합니다. 캐시 위치를 바꾸려면 아래처럼 지정합니다.

```bash
.venv/bin/python src/evaluate.py --config src/config.json --run-name baseline \
  --model-cache /path/to/model-cache --dataset-cache /path/to/dataset-cache
```

추론 프로세스가 종료되어 CUDA 메모리를 해제한 다음 저지 프로세스를 시작합니다. 오류가 발생하면 작업을 중단하고 완료된 응답과 저지 시도를 보존합니다. 같은 명령으로 재개하면 완료된 문항을 건너뜁니다. 같은 실행 폴더에 서로 다른 코드·설정·모델을 섞는 것은 거부하며 동시 쓰기도 차단합니다. 코드 또는 설정을 변경했으면 새 `--run-name`을 사용하십시오.

학습 후에는 동일 설정에서 체크포인트만 바꿉니다.

```bash
.venv/bin/python src/evaluate.py --config src/config.json \
  --checkpoint /path/to/merged-bf16-checkpoint --run-name finetuned
```

체크포인트는 같은 Qwen3-VL 구조의 **병합된 BF16 모델 디렉터리**여야 하며 모델·processor·tokenizer 파일을 포함해야 합니다. 단독 LoRA adapter와 양자화 모델은 이 진입점에서 받지 않습니다. 체크포인트 hash를 기록해 재개 시 다른 모델을 섞지 않습니다. 프롬프트·데이터·생성·이미지·채점 설정은 그대로 사용합니다. 학습 코드와 LoRA 병합 코드는 실제 학습 단계에서 `src`에 추가합니다.

## 확정한 채점

1. `finish_reason=length`인 응답은 전부 오답으로 처리합니다.
2. 최종 답 필드 파서 v2로 추출합니다. 모호하거나 충돌하는 최종 필드는 저지로 보냅니다.
3. 최종 답 필드가 없을 때만 공식 Qwen 규칙 파서를 전체 응답에 적용합니다.
4. 남은 응답은 Qwen3-8B로 공식 문항·보기·응답 매칭 프롬프트를 사용해 판정합니다. 저지에는 이미지를 주지 않습니다.
5. 저지 출력 파서 v3가 마지막 결정 또는 공식 fallback을 읽습니다. `Z`는 오답으로 즉시 종료합니다. 유효하지 않은 결과는 최대 25회까지 시도하고 최종 실패는 오답입니다. 무작위 대체와 수동 정답 보정은 없습니다.

주관식은 공식 구현을 따라 **채점에서만** A=참조 답, B=Other Answers로 변환합니다. 이 과정의 공식 문자 GT 처리 특례와 문자열 매칭을 유지합니다. 숫자 허용오차는 추가하지 않았습니다. 공식 규칙의 원문 함수는 `src/parsers.py`, 실제 프롬프트 전문과 출처 commit/hash는 `src/config.json`에 있습니다. 공식 코드 라이선스는 `THIRD_PARTY_LICENSES.txt`에 보존합니다.

## 결과 파일

`outputs/<run-name>/`에 저장하며 Git에서는 제외합니다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 실제 설정, 코드 hash, Git commit, 모델 식별, 캐시 위치, 환경 |
| `predictions.jsonl` | 질문·보기·참조·실제 입력·모델 응답 원문·토큰·종료 사유 |
| `judge_attempts.jsonl` | 저지 입력·응답·시도 번호·출력 결정 |
| `results.json` | 900문항 최종 채점 및 전체 응답 |
| `summary.json` | 30과목 정확도·macro 평균·채점 경로·종료 통계 |
| `scores.md` | 제출에 사용할 과목별 점수 표 |

이미지를 별도 파일로 저장하지 않고 고정 MMMU 데이터에서 직접 읽습니다. 각 문항의 `id`, `dataset_config`, `dataset_index`, `image_fields`로 원본 이미지를 찾습니다. 가중치·데이터·이미지·대용량 응답은 저장소에 포함하지 않습니다.

## 검증 및 기존 결과 재현

```bash
# GPU·패키지 설치 없이 파서의 실제 실패 사례를 검증
python src/evaluate.py --selfcheck

# 기존 900개 원문과 마지막 저지 출력으로 채점을 재현 (GPU/API 호출 없음)
python src/evaluate.py --config src/config.json --run-name archived-baseline \
  --import-results /path/to/experiment_3_final.json
```

저장된 실험 3 결과의 900문항을 재채점해 **576/900, 64.00%**를 재현했고, 문항별 정오답·추출 답·채점용 정답·채점 경로의 차이는 0개였습니다. 최종 답 파서 759개, 공식 fallback 6개, 출력 한도 81개, 저지 54개입니다. 원본 JSON은 서버에 보존되어 있으며 Git에 올리지 않습니다. 서버 원본 위치와 hash는 제출 보고서에 기록합니다.

이 검증은 저장 응답 재채점입니다. 새 코드로 900문항 전체를 새로 추론한 결과라고 표시하지 않습니다. 과거 저지는 25회 시도의 마지막 출력에 v3를 적용했지만, 새 실행은 v3 규칙으로 유효 결정 또는 Z가 나오면 종료합니다. 학습 효과를 비교할 때는 최종 코드로 베이스라인과 학습 모델을 모두 같은 절차로 실행하십시오. 동일 seed라도 하드웨어·라이브러리 차이에 따라 생성 결과가 달라질 수 있습니다.

추가 검증: 서버의 고정 데이터로 900문항 질문·보기·정답·프롬프트가 기존 기록과 일치함을 확인했습니다. 이미지 5장 문항을 포함한 대표 5문항에서 chat template와 실제 processor 입력 토큰 길이가 일치했습니다. 54개 저지 프롬프트도 공식 함수와 일치했습니다. GPU smoke에서는 평가 모델 1문항의 응답 원문이 기존 기록과 동일했고, 별도 프로세스의 저지 1문항은 첫 시도에 Z로 정상 종료했습니다. 준비 스크립트의 환경 검사, 패키지 가용성 dry-run, 설치 환경의 의존성 검사도 통과했습니다.

## 파일 구성

```text
README.md
requirements.txt
.gitignore
THIRD_PARTY_LICENSES.txt
reports/mmmu_baseline.md
src/prepare.py
src/evaluate.py
src/parsers.py
src/judge.py
src/config.json
```
