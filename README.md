# Baseline 모델의 MMMU Test 평가

이 브랜치는 **학습 전 기본 모델 `Qwen/Qwen3-VL-4B-Instruct`를 MMMU Test 10,500문항으로 평가한 실행을 정리하고, 같은 평가 절차를 다시 실행하기 위한 코드**를 담는다.

평가 대상은 고정 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`의 공식 Baseline 모델이다. 별도 학습 가중치 경로를 받지 않는다.

## 기존 완료 결과

| 실행 이름 | 생성 seed | 정답 / 전체 | 정확도 | 상태 |
| --- | ---: | ---: | ---: | --- |
| `base_test` | 3407 | 6,469 / 10,500 | **61.61%** | 완료 |

제공받은 원본 결과에서 10,500개 고유 ID와 모든 응답의 `seed=3407`, `model_path=null`을 확인했다. `model_path=null`은 생성기가 JSON 설정의 공식 모델 저장소와 고정 revision을 사용했다는 뜻이다.

별도의 `base_test_seed42` 파일은 확인 당시 5,038문항까지만 있었고 최종 채점이 없었다. 이를 완료 결과나 seed 42 정확도로 사용하지 않는다.

## 기존 seed 3407 조건으로 실행

Linux GPU 환경에서 다음과 같이 실행한다. 설치·GPU 요건과 학교 서버 예시는 [상세 안내](mmmu_test_eval/README.md)를 따른다.

```bash
git clone --single-branch --branch baseline_MMMu_test https://github.com/wlsdn66597/MMDL.git MMDL-baseline
cd MMDL-baseline/mmmu_test_eval
bash scripts/prepare_test.sh
bash scripts/run_test.sh 3407 base_test
```

`run_test.sh`는 모델 필드를 비워 공통 실행기에 전달한다. 생성기는 `--model-path`가 없으면 공식 Baseline 모델을 선택한다. `records/mmmu_test.json`의 렌더링된 프롬프트와 test 이미지를 입력한다.

패키지의 생략 시 기본 생성 seed는 42다. **위 명령은 기존 완료 결과의 seed 3407을 명시적으로 지정**한다. 새 seed 42 실행은 다음과 같다.

```bash
bash scripts/run_test.sh 42 base_test_seed42
```

## 실제 평가 조건

- 데이터: `MMMU/MMMU`, **test split, 30과목, 10,500문항**. 객관식 9,873개와 주관식 627개다.
- 두 모델은 같은 문항·이미지 전처리·렌더링된 프롬프트·파서·Judge 규칙을 사용했다.
- 기존 완료 결과를 비교하는 생성 seed는 **3407**이다. Judge seed도 3407이다.
- 출력 상한 32,768토큰, 문맥 64,000토큰, 동시 요청 16, 반복 생성 감지 시 중단을 사용했다.
- 기준 환경: Python 3.12, vLLM 0.11.0, torch 2.8.0, transformers 4.57.0, RTX 4090 24GB.
- 학습 seed와 평가의 생성 seed는 별개다. 아래 seed는 **평가 응답 생성 seed**를 뜻한다.

이 점수는 팀의 **로컬 채점 기준**이다. 주관식 문자열 포함 판정과 보조 Judge에는 알려진 오판정 사례가 있어 공식 서버 점수와 동일하다고 해석하지 않는다. 코드 업로드 검증에서 GPU 추론을 새로 실행한 것은 아니다.

## 이 브랜치에서 사용할 코드

평가 실행 위치는 저장소 루트의 **`mmmu_test_eval/`**이다.

| 파일 | 역할 |
| --- | --- |
| `mmmu_test_eval/scripts/prepare_test.sh` | 고정 환경·모델·데이터를 준비하고 MMMU Test 문항을 생성 |
| `mmmu_test_eval/scripts/run_test.sh` | 이 브랜치의 모델 선택, 생성 seed 지정, 평가·완료 검사 |
| `mmmu_test_eval/scripts/run_eval_dpo2e.sh` | 두 브랜치에서 함께 사용하는 생성 → Judge → 채점 실행기 |
| `mmmu_test_eval/scripts/diag/run_generate.py` | 실제 추론. `--model-path` 유무에 따라 공식 모델 또는 지정 모델 선택 |
| `mmmu_test_eval/scripts/protocol.py` | 코드·문항 해시, 모델·seed 혼용, 누락·중복·재채점 일치 검사 |

기존 `src/prepare.py`, `src/evaluate.py` 및 `src/eval/mmmu_test/`는 기준 브랜치에서 상속된 코드다. 이 브랜치의 Baseline/DPO-E 구분은 **위 `mmmu_test_eval/scripts/run_test.sh`**에 구현되어 있다.

### 코드에 남은 명칭을 읽는 방법

- `eval_settings.json`의 `validation`은 보존한 원본 설정값이다. `prepare_test.sh`가 `build_records.py mmmu-test`를 호출해 **split=test**로 바꾸고, Test 문항 해시·10,500개 고유 ID를 확인한다.
- 생성기에 전달하는 `--baseline records/mmmu_test.json`은 **평가 문항·프롬프트·설정 JSON의 경로**다. 이 옵션 이름이 모델 종류를 정하지 않는다.
- `run_generate.py`의 과거 `val` 설명과 `run_eval_dpo2e.sh`의 파일명은 공통 원본 코드에 남아 있다. 실제 모델은 `--model-path`, 실제 데이터는 입력 JSON의 `settings.dataset.split`로 정해진다.
- 두 브랜치의 파서·Judge·문항 생성 코드는 같은 원본이다. 원본 파일은 SHA256 검증을 위해 바이트를 유지했다.

## 결과 파일과 추가 안내

실행 후 `mmmu_test_eval/outputs/<tag>.jsonl`, `.judge.jsonl`, `.graded.jsonl`, `.summary.txt`, `.manifest.json`, `.verification.json`이 생성된다. 마지막 검증 파일의 `status=PASS`, 예측·채점 각각 10,500개, 누락·중복·재채점 불일치 0건을 확인한다.

- [환경 준비·학교 컨테이너·재개·결과 회수](mmmu_test_eval/README.md)
- [정확한 실행 조건 및 채점 한계](mmmu_test_eval/PROTOCOL.md)
- [배포 전 검증 범위](mmmu_test_eval/VALIDATION.md)
- [원 코드 출처와 라이선스](THIRD_PARTY_LICENSES.txt)

이 브랜치는 평가 코드와 설명을 보관한다. 가중치·이미지 캐시·실제 문항별 응답은 포함하지 않는다.
