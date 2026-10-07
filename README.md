# Qwen3-VL-4B MMMU 평가

**두 번 말하게 하지마** 팀의 Qwen3-VL-4B-Instruct MMMU validation 평가 저장소입니다. 환경 준비와 900문항 평가를 제공하며, 학습한 체크포인트도 같은 설정으로 평가할 수 있습니다. 현재 학습·LoRA 병합 코드는 포함하지 않습니다.

평가 흐름: **모델 추론 → 최종 답 추출 → 공식 규칙 fallback → 필요한 문항의 로컬 Qwen3-8B 저지 → 결과 저장**.

실제 프롬프트, 생성·이미지 설정, 채점 규칙, 점수와 평가 한계는 [베이스라인 평가 보고서](reports/mmmu_baseline.md)에 정리했습니다.

## 저장소 구성

| 파일·폴더 | 역할 |
|---|---|
| `src/prepare.py` | 가상환경·패키지 설치, 고정 모델·데이터 준비와 환경 검사 |
| `src/evaluate.py` | 전체 평가 실행, 중단 후 재개, 체크포인트 지정, 저장 응답 재채점 |
| `src/parsers.py` | 최종 답·공식 규칙·저지 출력 파서 |
| `src/judge.py` | 로컬 Qwen3-8B 저지 실행 |
| `src/config.json` | 모델·데이터 revision, 프롬프트, 생성·이미지·채점 설정 |
| `reports/` | 평가 방법·결과·분석 보고서 |
| `requirements.txt` | 고정 패키지 버전 |
| `THIRD_PARTY_LICENSES.txt` | 차용한 공식 코드의 출처와 라이선스 |
| `outputs/` | 실행 시 생성되는 결과 폴더, Git에서 제외 |

## 빠른 시작

Linux x86-64(Ubuntu 또는 NVIDIA GPU 사용이 가능한 WSL2), Python 3.12, BF16을 지원하는 NVIDIA GPU와 드라이버가 필요합니다. 기준 GPU는 RTX 4090 24GB입니다. Python과 드라이버는 사전 설치해야 하며, vLLM GPU 추론은 Windows 네이티브 환경을 대상으로 하지 않습니다.

```bash
git clone https://github.com/yunheeseong1/mmdl_test.git
cd mmdl_test
python3.12 src/prepare.py
.venv/bin/python src/evaluate.py --config src/config.json --run-name baseline
```

준비 스크립트는 `.venv`를 만들고 고정 패키지를 설치한 뒤 평가 모델·저지 모델·MMMU 30과목 validation을 Hugging Face 캐시에 준비합니다. 인터넷과 충분한 저장 공간이 필요합니다. 평가는 준비된 캐시를 오프라인으로 사용하며, 모델 추론이 종료된 다음 저지를 실행합니다.

## 사용 옵션

### 캐시 위치 지정

```bash
python3.12 src/prepare.py \
  --model-cache /path/to/model-cache --dataset-cache /path/to/dataset-cache
.venv/bin/python src/evaluate.py --run-name baseline \
  --model-cache /path/to/model-cache --dataset-cache /path/to/dataset-cache
```

캐시 위치는 명령줄 인자, `MMMU_MODEL_CACHE`·`MMMU_DATASET_CACHE` 환경변수, `src/config.json` 순으로 적용됩니다. `--model-cache`는 모델과 저지의 Hugging Face 캐시 루트, `--dataset-cache`는 고정 MMMU 데이터의 캐시 위치입니다.

### 기존 환경·캐시 사용

```bash
/path/to/venv/bin/python src/prepare.py --existing-env --offline
# 패키지 버전·GPU만 점검
/path/to/venv/bin/python src/prepare.py --existing-env --check-only
```

### 중단 후 재개

같은 명령과 `--run-name`으로 다시 실행하면 완료된 응답·저지 시도를 재사용합니다. 같은 실행 폴더의 동시 쓰기와 코드·설정·모델 혼용을 차단하므로, 변경 후에는 새 `--run-name`을 사용하십시오.

### 체크포인트 변경

```bash
.venv/bin/python src/evaluate.py --config src/config.json \
  --checkpoint /path/to/merged-bf16-checkpoint --run-name finetuned
```

`--checkpoint`를 생략하면 고정 공식 모델을 사용합니다. 지정할 때는 같은 Qwen3-VL 구조의 BF16 전체 모델 디렉터리와 processor·tokenizer 파일이 필요합니다. 단독 LoRA adapter와 양자화 모델은 받지 않습니다. 준비된 공식 모델·저지·데이터 캐시도 필요하며, 평가 설정은 그대로 적용합니다.

### 파서 검사·저장 응답 재채점

```bash
python src/evaluate.py --selfcheck
python src/evaluate.py --run-name saved-baseline \
  --import-results /path/to/baseline_results.json
```

이 두 명령은 GPU/API 호출 없이 실행합니다. 재채점 입력에는 `settings`의 모델·데이터 revision, sampling·vision 설정과 `records`의 문항·프롬프트·응답·최종 채점이 필요합니다. 저지가 필요한 문항은 `grading.local_qwen3_8b_judge`에 `attempt_count`·`final_raw_output`도 포함해야 합니다. 이 보존 입력 형식은 아래의 실행 결과 `results.json`과 다릅니다.

## 결과 확인

`outputs/<run-name>/`에 저장합니다.

| 확인할 내용 | 파일 |
|---|---|
| 실제 설정·코드 hash·Git commit·모델·환경 | `manifest.json` |
| 문항·입력 프롬프트·모델 응답 원문·토큰·종료 사유 | `predictions.jsonl` |
| 저지의 시도별 입력·출력·결정 | `judge_attempts.jsonl` |
| 문항별 최종 정오답과 전체 응답 | `results.json` |
| 전체·과목별 정확도와 채점 경로·종료 통계 | `summary.json` |
| 제출용 과목별 점수 표 | `scores.md` |

가중치·데이터·이미지·대용량 응답은 Git에 포함하지 않습니다. 이미지는 별도 폴더로 내보내지 않으며 문항의 `id`, `dataset_config`, `dataset_index`, `image_fields`로 원본 MMMU 데이터에서 확인합니다.

## 평가 보고서·라이선스

- [베이스라인 평가 보고서](reports/mmmu_baseline.md): 제출 템플릿에 따른 평가 조건·결과·공식 수치 비교·한계
- [고정 설정](src/config.json): 실행에 적용되는 설정과 공식 코드 출처
- [공식 코드 라이선스](THIRD_PARTY_LICENSES.txt)
