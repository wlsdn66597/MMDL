# MMDL · Qwen3-VL-4B 학습과 평가

**두 번 말하게 하지마** 팀의 Qwen3-VL-4B-Instruct 실험 저장소입니다. 원문 풀이와 ERA(Evidence, Rule, Application)를 결합한 SFT, DPO2-E·DPO2F-CD 학습, MMMU validation·Test·Pro 평가를 재현할 수 있도록 코드와 결과를 정리했습니다.

## 저장소 구성

| 위치 | 내용 |
| --- | --- |
| [`src/train/sft/`](src/train/sft/README.md) | 원문 풀이 + ERA 2,000문항 SFT, epoch별 LoRA 저장과 validation 평가 |
| [`src/train/dpo/`](src/train/dpo/README.md) | DPO2-E 선호 쌍 생성·학습·병합, DPO2F-CD 자기증류 SFT와 DPO |
| [`src/eval/mmmu_validation/`](src/eval/mmmu_validation/README.md) | MMMU validation 900문항 평가와 저장 응답 재채점 |
| [`src/eval/mmmu_test/`](src/eval/mmmu_test/README.md) | MMMU Test 10,500문항 준비·추론·채점·완료 검증 |
| [`src/eval/mmmu_pro_std10/`](src/eval/mmmu_pro_std10/README.md) | MMMU-Pro 표준 10지선다 Test 1,730문항 평가 |
| [`reports/`](reports/) | 모델별 점수, 과목별 비교, 학습 로그와 평가 분석 |
| [`assignment/`](assignment/) | 제출용 베이스라인 문서 |

## 주요 평가 결과

| 평가 | 문항 수 | 기본 모델 | ERA SFT | DPO2-E 재현 | DPO2F-CD | 상세 결과 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MMMU validation | 900 | 64.00% | 57.56% (5 epoch) | 66.89% | 66.11% | [SFT epoch별 결과](reports/mmmu.md) · [DPO 학습 seed별 결과](reports/dpo2/README.md) |
| MMMU Test | 10,500 | 61.61% | — | 65.89% | 65.04% | [Test 비교](reports/mmmu_test/comparison.md) |
| MMMU-Pro 표준 10지선다 Test | 1,730 | 52.60% | — | 54.68% | — | [Pro 비교](reports/mmmu_pro_std10/comparison.md) |

표는 생성 seed `3407` 기준이며 validation의 DPO 점수는 학습 seed `3407` 실행입니다. 생성 seed `42`의 DPO2-E Test 결과와 채점 한계는 Test 비교 문서에서 확인할 수 있습니다.

## 평가 실행

Linux x86-64, Python 3.12, BF16을 지원하는 NVIDIA GPU를 사용합니다. 기준 GPU는 RTX 4090 24GB입니다. 준비 스크립트는 고정 버전의 패키지와 모델·데이터를 받습니다. 학습 모델을 평가하려면 tokenizer와 processor를 포함한 전체 병합 체크포인트 경로를 지정합니다.

```bash
git clone --branch mmdl-test-main --single-branch https://github.com/wlsdn66597/MMDL.git
cd MMDL
```

### MMMU validation

```bash
python3.12 src/eval/mmmu_validation/prepare.py
.venv/bin/python src/eval/mmmu_validation/evaluate.py --run-name baseline
```

병합 모델에는 `--checkpoint /path/to/merged`와 새 `--run-name`을 사용합니다. 기본 요약은 `outputs/<run-name>/summary.json`에 저장됩니다.

### MMMU Test

저장소 루트에서 아래 디렉터리로 이동합니다. 기본 모델과 병합 모델을 같은 생성 seed로 평가하는 예입니다. `/path/to/merged`를 실제 모델 경로로 바꿉니다.

```bash
cd src/eval/mmmu_test
bash scripts/prepare_test.sh
bash scripts/run_test.sh baseline base_test_seed3407 3407
bash scripts/run_test.sh /path/to/merged trained_test_seed3407 3407
```

요약과 완료 검증은 이 폴더의 `outputs/<tag>.summary.txt`와 `outputs/<tag>.verification.json`에서 확인합니다.

### MMMU-Pro 10지선다

[Pro 실행 안내](src/eval/mmmu_pro_std10/README.md)에 따라 루트의 `.venv`를 준비한 뒤 저장소 루트에서 실행합니다.

```bash
.venv/bin/python src/eval/mmmu_pro_std10/run_mmmu_pro_std10.py
```

병합 모델에는 `--checkpoint /path/to/merged --run-name trained`를 추가합니다. 결과는 `src/eval/mmmu_pro_std10/outputs/`의 실행 폴더에 저장됩니다.

## 학습 데이터와 제출 문서

- ERA 2,000문항은 [Hugging Face ZIP](https://huggingface.co/datasets/SSU-MMDL/MMMU_ERA_Dataset/blob/main/vic_era_2000.zip)에서 받습니다. [SFT 실행 안내](src/train/sft/README.md)에 데이터 준비, `--check-data` 검증, 실제 학습 명령이 있습니다. 이전 SFT 체크포인트는 실습실 저장공간 초기화로 소실됐으며 학습 코드와 평가 기록을 보관합니다.
- DPO2-E와 DPO2F-CD는 DPO 단계에서 같은 `pairs_E.jsonl`을 사용합니다. F-CD는 그 전에 자기증류 SFT를 진행합니다. 학습에 필요한 입력과 실행 방법은 [DPO 실행 안내](src/train/dpo/README.md)를 참고합니다.
- [제출용 베이스라인 보고서](assignment/mmmu_baseline.md)에는 평가 설정·채점 규칙·점수가 있습니다. [제3자 코드 출처 및 라이선스](THIRD_PARTY_LICENSES.txt)도 함께 제공합니다.
