# MMMU Test 10,500문항 평가

AIX-408-06에서 2026-11-21에 시작한 **DPO2-E 재현 모델 · MMMU Test 10,500문항 · 생성 seed 42** 평가와 같은 방법이다. 다른 ERA 강화 모델도 **병합 모델 경로와 실행 이름만 바꿔서** 평가한다. 아래 명령은 이 폴더를 기준으로 한다.

현재 서버에서 사용한 생성·채점 코드와 `run_eval_dpo2e.sh`를 원본 바이트 그대로 보존했다. 설치, 파일 검증, 실행 기록, 완료 검증만 바깥에 추가했다. 저장소의 `src/evaluate.py`는 기존 **validation 900문항** 평가이므로 이번 Test 실행에는 사용하지 않는다.

## 1. 필요한 것

- Linux x86-64, Python **3.12**, NVIDIA GPU. 기준 환경은 **RTX 4090 24GB**, vLLM **0.11.0**, torch **2.8.0**, transformers **4.57.0**이다. 패키지 전체 버전은 `env/requirements_eval.txt`에 고정했다.
- 평가할 **Qwen3-VL-4B의 BF16 전체 병합 체크포인트**. `config.json`, `model*.safetensors`, shard index, tokenizer/processor 파일이 들어 있는 `merged/` 디렉터리가 필요하다. 양자화 모델이나 LoRA adapter 디렉터리를 지정하지 않는다. ERA/DeepStack gate 등 추가 학습 파라미터도 제작자가 최종 가중치에 병합해야 한다.
- 처음 한 번 모델·데이터를 받을 인터넷과 수십 GB 이상의 디스크 여유. 공식 4B 모델 캐시 약 8.3GB, Judge 약 16GB, MMMU 원본 약 3.5GB에 별도 학습 모델·Arrow 캐시·Python 환경 공간이 추가된다.

**모델 가중치는 이 Git 브랜치에 없다.** 새 모델 제작자에게 전체 `merged/` 폴더 또는 그 압축을 받아 아래 경로에 먼저 풀어 둔다. 정확한 DPO2-E 가중치를 다시 평가하려면 `DPO2E_repro_merged.zip`을 별도로 받아야 하며, 기존 압축 SHA256은 다음과 같다.

```text
0f1fd00162da1519a6596d6c7ee5b01079cf7a4d3c7ff2bb33c4a1546657bf3d
```

새 ERA 모델은 이 DPO2-E 압축과 다른 파일이므로 이 해시와 일치할 필요가 없다. 실행 시 실제 가중치 전체의 SHA256을 별도로 기록한다.

## 2. 학교 GPU 서버에서 준비하기

### 호스트 터미널 (`gpu@AIX-...:~$`)

호스트의 `~/shared_dir`가 학교 컨테이너의 `/shared_dir`에 연결된다. Python 설치와 추론은 컨테이너 안에서 수행한다. 아래 예시는 학교의 `~/docker/run.sh`가 있는 서버용이다.

```bash
cd ~/shared_dir
git clone --branch mmdl-test-main --single-branch \
  https://github.com/wlsdn66597/MMDL.git MMDL
```

모델을 다음 위치에 넣는다. 예시 이름 `era_plus`는 받은 모델에 맞게 바꿔도 된다.

```text
호스트:   ~/shared_dir/models/era_plus/merged/config.json
컨테이너: /shared_dir/models/era_plus/merged/config.json
```

기존 GPU 평가가 없는 상태에서 환경·모델 캐시·Test 문항을 준비한다. `run.sh`는 멈춘 동일 이름 컨테이너를 교체하므로, 준비와 실행 사이에도 필요한 파일은 모두 공유 폴더에 둔다.

```bash
bash ~/docker/run.sh bash -c '
  cd /shared_dir/MMDL/src/eval/mmmu_test &&
  bash scripts/prepare_test.sh
'
```

준비 과정은 다음을 수행한다.

1. 공유 폴더 안 `.venv`에 고정 패키지를 설치하고 Python 버전·GPU를 검사한다.
2. `.lab/hf`에 고정 revision의 공식 모델, Judge, MMMU를 받는다.
3. **30과목의 test Arrow 캐시**를 준비한다. Hub snapshot 다운로드만 하고 이 단계를 생략하면 오프라인 이미지 로딩이 실패할 수 있다.
4. 원본 프롬프트 생성기로 `records/mmmu_test.json`을 만들고 **현재 평가 서버의 파일 SHA256과 일치해야** 완료한다.

마지막에 `PASS records-check`와 `Ready:`가 나오는지 확인한다. 해시나 버전이 다르면 그대로 평가를 시작하지 않는다.

## 3. 새 모델 평가 시작 — 호스트 터미널

**모델 경로와 실행 이름**을 정한다. 아래 예시에서는 `era_plus_test_seed42`를 쓴다. 모델이 바뀔 때마다 새 이름을 사용한다. 생성 seed 기본값은 **42**이며, 세 번째 인자로 바꿀 수 있다. Judge seed는 **3407**이다.

```bash
nohup bash "$HOME/docker/run.sh" bash -c '
  cd /shared_dir/MMDL/src/eval/mmmu_test &&
  exec bash scripts/run_test.sh \
    /shared_dir/models/era_plus/merged \
    era_plus_test_seed42
' </dev/null >> "$HOME/shared_dir/MMDL/src/eval/mmmu_test/era_plus_test_seed42.launch.log" 2>&1 &
```

같은 생성 seed 3407 비교는 다음 명령으로 시작한다. 기본 모델 평가에도 고정 공식 모델의 스냅샷 경로를 같은 위치에 전달한다.

```bash
bash scripts/run_test.sh /absolute/path/to/merged trained_test_seed3407 3407
```

처음에 가중치 해시를 계산한 뒤 `Protocol verified: ... generation seed=<선택한 값>, judge seed=3407`이 나온다. 이어서 생성 → Qwen3-8B 보조 채점 → 원본 채점 → 요약 → 10,500개 ID·독립 재채점 검증 순으로 실행한다. 한 GPU에서는 평가 한 개씩 실행한다.

학교 시스템이 같은 컨테이너를 중단했다가 `docker start`로 재시작하면 위 명령이 다시 실행되고 이미 저장된 응답 ID를 건너뛴다. 모델·환경이 같아야 재개를 허용한다. 컨테이너가 내려가 있고 자동으로 돌아오지 않으면 위 명령을 다시 실행한다.

### 현재 DPO2-E seed 42 실행과 같은 모델을 쓰는 예

이전 서버에 이미 있는 모델 위치를 쓰려면 실행 인자는 다음과 같다. 친구 서버에서는 실제로 전달받아 풀어 놓은 컨테이너 경로로 바꾼다.

```bash
bash scripts/run_test.sh \
  /shared_dir/dpo2e_eval_20261119/checkpoint/merged \
  dpo2e_repro_test_seed42
```

이 명령은 **컨테이너 안에서 `src/eval/mmmu_test` 폴더에 있는 경우**의 예이다. 기존 `mmmu_eval_base/outputs` 응답은 새 패키지에 복사하지 않는다. 새 패키지의 실행 기록(manifest)이 없는 결과와 혼용하지 않도록 검사한다.

## 4. 진행 확인·중단·결과 회수 — 호스트 터미널

```bash
# 초기화, 파일 검사, 현재 단계, 최종 검증
tail -n 40 ~/shared_dir/MMDL/src/eval/mmmu_test/era_plus_test_seed42.launch.log

# 생성 진행 (초기화가 시작된 뒤 파일 생성)
tail -f ~/shared_dir/MMDL/src/eval/mmmu_test/outputs/era_plus_test_seed42.gen.log

# 완료 후 정확도
cat ~/shared_dir/MMDL/src/eval/mmmu_test/outputs/era_plus_test_seed42.summary.txt

# 최종 완료 검증
cat ~/shared_dir/MMDL/src/eval/mmmu_test/outputs/era_plus_test_seed42.verification.json
```

`Ctrl+C`는 `tail -f` 표시만 종료한다. 평가 전체를 중단하려면 해당 컨테이너 이름을 확인한 뒤 중단한다. 학교 기본 이름은 `gpu-cuda`이며 다른 작업도 같은 컨테이너 안에 있으면 함께 중단된다.

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
docker stop -t 30 gpu-cuda
```

정상 완료 기준은 **응답·최종 채점 각각 고유 ID 10,500개, 누락·중복·재채점 불일치 0개**이다. 단순 줄 수나 `summary.txt` 존재만으로 완료를 판단하지 않는다. `verification.json`이 마지막 단계에서 생성된다.

친구가 돌려준 결과는 다음처럼 묶어서 전달한다. `records`, 원본 응답, Judge 결과, 코드·가중치·실행 환경 해시를 함께 받아야 비교를 검증할 수 있다. 이미지 캐시·모델 가중치는 결과 압축에 넣지 않는다.

```bash
cd ~/shared_dir/MMDL/src/eval/mmmu_test
tar -czf ../era_plus_test_seed42_results.tar.gz \
  outputs/era_plus_test_seed42.* \
  records/mmmu_test.json protocol_reference.json eval_settings.json \
  era_plus_test_seed42.launch.log
```

부분 결과도 지우지 않는다. 중단 때문에 JSONL 마지막 줄이 깨졌거나 `.graded.jsonl`이 일부만 저장되었으면 자동 재개가 중단된다. 원본을 먼저 보관하고 오류를 확인해야 한다. 생성 응답이 온전하고 채점 파일만 일부인 경우, 그 `.graded.jsonl`을 다른 이름으로 옮긴 뒤 같은 명령을 재실행하면 원본 채점 절차가 다시 실행된다. seed나 모델이 다른 응답을 합치면 안 된다.

## 일반 Linux GPU 머신에서 실행하기

학교 컨테이너를 쓰지 않는 경우에도 위와 같이 같은 브랜치를 `MMDL` 폴더에 clone하고 이 폴더에서 실행한다.

```bash
cd MMDL/src/eval/mmmu_test
bash scripts/prepare_test.sh
nohup bash scripts/run_test.sh /absolute/path/to/merged era_plus_test_seed42 \
  </dev/null >> era_plus_test_seed42.launch.log 2>&1 &
```

캐시 위치를 바꾸려면 **준비와 실행 모두** 같은 `HF_HOME=/absolute/cache`를 지정한다. 준비된 가상환경을 사용할 때는 두 명령 모두 `PY=/absolute/venv/bin/python`을 지정한다. 준비 단계는 해당 환경에 고정 패키지를 설치한다. 출력은 항상 이 패키지의 `outputs/`에 생긴다.

## 평가 조건과 해석

배포 전 검사 결과와 새 GPU에서 아직 실행하지 않은 범위는 [VALIDATION.md](VALIDATION.md)에 기록했다.

전체 설정은 [PROTOCOL.md](PROTOCOL.md), 고정 파일 해시는 [protocol_reference.json](protocol_reference.json)에 있다. `eval_settings.json`에 남은 validation/900·동시 처리 1·이미지 5 값은 원본 템플릿이다. 문항 생성과 실행 인자로 **test/10,500·동시 처리 16·이미지 최대 7·선택한 생성 seed**가 실제 적용된다. 설정 JSON 전체의 seed를 42로 일괄 변경하지 않는다.

기존 DPO2-E seed 3407 결과는 **6,918/10,500 = 65.89%**이다. seed 42 또는 새 모델 결과는 실행 완료 후 비교한다. 이 점수는 팀의 **제공된 로컬 채점기 기준**이며 공식 서버 제출 점수가 아니다. 같은 방법·seed라도 비동기 배치, GPU, 드라이버 등에 따라 응답이 완전히 같다고 보장할 수 없다. 가능한 한 같은 GPU와 고정 환경을 사용하고, 다른 환경이면 결과와 함께 명시한다.

이 브랜치에는 모델 가중치, 데이터 원문·정답, 학습 데이터, 실제 응답, 계정 정보가 포함되지 않는다. 데이터는 고정 Hugging Face revision에서 준비한다. `scripts/diag/`와 원본 실행기의 채점 규칙은 현재 실행과의 비교를 위해 수정하지 않았다.

[기본 모델·DPO2-E·DPO2F-CD 결과 비교](../../../reports/mmmu_test/comparison.md)
