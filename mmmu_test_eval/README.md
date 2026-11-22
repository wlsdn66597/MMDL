# Baseline Qwen3-VL-4B-Instruct · MMMU Test 평가

이 브랜치는 공식 Baseline을 **MMMU Test 10,500문항으로 평가한 코드**다. 기존 완료 실행은 생성 seed **3407**, 정답 **6,469/10,500 (61.61%)**였다. 기존 조건 재실행은 `bash scripts/run_test.sh 3407 base_test`를 사용한다. seed 42는 별도 실행이며, 제공된 기존 seed 42 파일은 5,038문항으로 미완료였다.

브랜치: `baseline_MMMu_test` · 용도: **MMMU Test 10,500문항 평가**

공식 기본 모델 Qwen/Qwen3-VL-4B-Instruct의 고정 revision을 평가한다. 별도 병합 모델 경로를 받지 않는다.

기본 생성 seed는 **42**이고, 인자로 **3407** 등 다른 seed를 선택할 수 있다. Baseline과 DPO-E는 같은 seed로 비교한다. **Judge seed는 생성 seed와 별개로 3407**이다.

## 구성

| 경로 | 역할 |
| --- | --- |
| `scripts/prepare_test.sh` | 고정 환경 설치, 모델·데이터 캐시, Test 문항 생성·해시 확인 |
| `scripts/run_test.sh` | 이 브랜치의 모델 선택, 생성·채점·완료 검증 |
| `scripts/run_eval_dpo2e.sh` | 실제 사용했던 생성 → Judge → 채점 → 요약 실행기 |
| `scripts/diag/` | 실제 사용했던 생성·파서·채점·Judge 코드 |
| `scripts/protocol.py` | 코드·문항·모델 해시 기록, seed 혼용 방지, 재개·완료 검증 |
| `eval_settings.json`, `protocol_reference.json` | 원본 설정과 실제 Test 실행 조건·고정 해시 |
| `env/requirements_eval.txt` | 실제 환경의 고정 패키지 버전 |
| `PROTOCOL.md`, `VALIDATION.md` | 설정·채점 한계와 배포 전 검증 범위 |

실제 응답·정답 데이터, 모델 가중치, 이미지 캐시, 가상환경은 포함하지 않는다. `prepare_test.sh`가 고정 revision의 자료를 받아 문항을 생성한다.

## 1. 일반 Linux GPU 환경에서 준비

Python **3.12**, BF16을 지원하는 NVIDIA GPU와 `flock`(util-linux)이 있는 Linux 환경이 필요하다. 실제 평가 환경은 RTX 4090 24GB 한 장, vLLM 0.11.0, torch 2.8.0, transformers 4.57.0이었다. 모델·Judge·데이터·가상환경을 위한 수십 GB의 여유 공간이 필요하다.

이 브랜치를 받은 뒤 `mmmu_test_eval/` 안에서 실행한다.

```bash
bash scripts/prepare_test.sh
```

준비가 끝나면 `PASS records-check`와 `Ready:`가 출력된다. 프롬프트·보기·이미지 참조가 들어 있는 `records/mmmu_test.json`의 SHA256이 기존 평가와 일치해야 진행된다. Test Arrow 이미지 캐시도 30과목 전체를 준비한다.

캐시를 공유하려면 **준비와 실행 모두** 같은 `HF_HOME`을 지정한다. 예: `export HF_HOME=/shared_dir/mmmu_cache/hf`. 지정하지 않으면 이 패키지의 `.lab/hf`를 쓴다.

## 2. 평가 실행

준비 과정에서 공식 Baseline을 캐시에 받는다. 실행 시 해당 고정 revision의 가중치 해시를 기록한다.

```bash
bash scripts/run_test.sh 42 base_test_seed42
```

기존 seed 3407 조건으로 실행하려면:

```bash
bash scripts/run_test.sh 3407 base_test_seed3407
```

다른 모델·seed는 새 실행 이름(tag)을 사용한다. 같은 tag로 다시 실행하면 동일 모델·설정·환경인지 검사하고 이미 저장한 문항을 건너뛴다. 기존 서버 결과를 이 패키지의 `outputs/`에 복사해 재개하지 않는다. 실행 기록(manifest)이 없는 결과와 새 실행을 섞지 않도록 검사한다.

## 3. 학교 408 서버에서 실행

호스트의 `~/shared_dir`가 컨테이너의 `/shared_dir`에 연결되는 기존 환경을 기준으로 한다. 아래는 브랜치 체크아웃을 `~/shared_dir/baseline_MMMu_test/`에 둔 예다. Python 준비·추론은 컨테이너 안에서 수행한다.

먼저 필요한 GPU 작업이 없는 상태에서 준비한다.

```bash
bash "$HOME/docker/run.sh" bash -c '
  cd /shared_dir/baseline_MMMu_test/mmmu_test_eval &&
  bash scripts/prepare_test.sh
'
```

호스트 터미널에서 백그라운드 실행:

```bash
nohup bash "$HOME/docker/run.sh" bash -c '
  cd /shared_dir/baseline_MMMu_test/mmmu_test_eval &&
  exec bash scripts/run_test.sh 42 base_test_seed42
' </dev/null >> "$HOME/shared_dir/baseline_MMMu_test/mmmu_test_eval/base_test_seed42.launch.log" 2>&1 &
```

학교 `run.sh`가 동일 이름 컨테이너를 교체할 수 있으므로 다른 평가가 실행 중이면 위 명령을 동시에 실행하지 않는다. 일반 Linux에서 백그라운드 실행할 때도 위 평가 명령 앞에 `nohup`을 붙이고 stdin을 `/dev/null`, 로그를 `outputs/` 바깥 파일로 연결한다.

## 4. 진행과 결과 확인

패키지 폴더 안에서:

```bash
tail -f outputs/base_test_seed42.gen.log
cat outputs/base_test_seed42.summary.txt
cat outputs/base_test_seed42.verification.json
```

`Ctrl+C`는 `tail -f` 표시만 종료한다. 생성 완료 후 Judge와 최종 채점이 이어진다. 완료 기준은 `verification.json`의 **status=PASS, unique_predictions=10500, unique_graded=10500, missing=0, duplicates=0, regrade_mismatches=0**이다.

| 파일 | 내용 |
| --- | --- |
| `outputs/<tag>.jsonl` | 문항별 실제 응답 |
| `outputs/<tag>.judge.jsonl` | 보조 채점 결과 |
| `outputs/<tag>.graded.jsonl` | 문항별 판정 |
| `outputs/<tag>.summary.txt` | 정확도·과목별 요약 |
| `outputs/<tag>.manifest.json` | 모델·코드·문항 해시와 실제 seed·환경 |
| `outputs/<tag>.verification.json` | 누락·중복·저장 Judge 재채점 검증 |

완료 후 전달할 결과를 묶는 예:

```bash
tar -czf ../base_test_seed42_results.tar.gz \
  outputs/base_test_seed42.* records/mmmu_test.json \
  eval_settings.json protocol_reference.json scripts env
```

최상위 `base_test_seed42.launch.log`도 생성했다면 함께 전달한다. 모델·이미지 캐시는 결과 묶음에서 제외한다.

## 평가 해석

이 코드는 기존 점수와 비교하기 위해 **원래 파서와 Judge 규칙을 그대로 보존**했다. 주관식은 정답 문자열 포함 여부를 사용해 `10`과 `100`을 잘못 같게 취급할 수 있고, Judge가 서로 다른 숫자를 유사하다는 이유로 정답 처리한 사례도 있다. 결과는 **동일 로컬 채점 기준의 점수**이며 공식 서버 점수나 모든 판정의 의미적 정확성을 보증하지 않는다.

새 래퍼의 완료 검증은 저장된 Judge 판정을 재사용한다. 같은 seed라도 GPU·드라이버·비동기 배치에 따라 응답이 완전히 같다고 보장하지 않는다. 새 머신의 전체 설치·GPU 추론은 이번 파일 구성 과정에서 실행하지 않았다.
