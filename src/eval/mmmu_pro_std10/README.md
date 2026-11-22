# MMMU-Pro 표준 10지선다 평가

AIX-408-27에서 실행한 Qwen3-VL-4B-Instruct 평가를 재현한다. 기본 모델의 참고 결과는 **910/1,730 (52.60%)**이며, 반복 생성 조기 중단 175개와 출력 한도에 도달한 2개를 오답으로 포함했다. 기본 모델과 학습 모델에 아래 평가 조건을 적용한다.

| 항목 | 설정 |
|---|---|
| 데이터 | `MMMU/MMMU_Pro` · `standard (10 options)` · `test` 1,730문항 · `563f3e84bb3b90893083a1f039cfa13077f2302b` |
| 기본 모델 | `Qwen/Qwen3-VL-4B-Instruct` · `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 생성 | vLLM 0.11.0, BF16, eager, GPU 메모리 비율 0.90, 동시 요청 16, 배치 토큰 4,096 |
| 샘플링 | seed 3407, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0, presence penalty 1.5 |
| 길이 | 문맥 64,000토큰, 생성 최대 32,768토큰 |
| 이미지 | RGB, 이미지 번호 순서, 1,003,520~4,014,080픽셀, 최대 7장, Qwen 전처리 후 vLLM 재리사이즈 끔 |
| 프롬프트 | 평가 모델의 processor와 chat template을 사용하고 `Final answer: LETTER`를 마지막 줄에 요청 |
| 반복 중단 | 4,096토큰마다 검사. 최근 구간의 12단어 묶음 반복률이 50% 이상인 검사가 연속 두 번이면 중단 |
| 채점 | 최종 보기 문자와 규칙 기반 추출, 필요시 Qwen3-8B 저지 1회. 반복 중단과 출력 한도 도달은 오답 |
| 저지 | `Qwen/Qwen3-8B` · `b968826d9c46dd6066d109eabc6255188de91218` · BF16 · seed 3407 · temperature 0 · 최대 출력 4,096 · thinking 끔 |

스크립트가 원래 실행에서 저장한 1,730개 문항 레코드와 SHA-256을 대조한다. 데이터가 준비되면 로그에 `Verified reference records SHA-256:`이 찍힌다. 학습 모델로 비교할 때는 기본 모델과 같은 Qwen3-VL 구조, tokenizer, processor, chat template을 보존한 BF16 병합 모델을 사용한다.

## 실행 환경

Linux x86-64, Python 3.12, BF16을 지원하는 NVIDIA GPU가 필요하다. 기준은 RTX 4090 24GB다. 저장소 루트의 [`requirements.txt`](../../../requirements.txt)가 vLLM 0.11.0, PyTorch 2.8.0, Transformers 4.57.0을 포함해 157개 버전을 고정한다. 같은 패키지 버전으로 준비한 기존 가상환경을 사용할 수 있다. vLLM의 비동기 배치와 GPU·드라이버 차이는 같은 모델과 seed에서도 생성 결과에 영향을 줄 수 있다.

```bash
git clone --branch mmdl-test-main --single-branch \
  https://github.com/wlsdn66597/MMDL.git MMDL
cd MMDL
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

실행할 때 데이터 Parquet 두 개와 필요한 고정 리비전의 모델을 캐시에 준비한다. 기본 캐시는 `src/eval/mmmu_pro_std10/.cache/huggingface`다. 기존 캐시는 명령에 `HF_HOME`을 지정해 재사용한다.

## 기본 모델과 학습 모델 실행

```bash
# 기본 모델
.venv/bin/python -u src/eval/mmmu_pro_std10/run_mmmu_pro_std10.py

# 학습 모델
.venv/bin/python -u src/eval/mmmu_pro_std10/run_mmmu_pro_std10.py \
  --checkpoint /absolute/path/to/merged --run-name trained_model
```

`--checkpoint`는 `config.json`, tokenizer·processor 설정, 전체 `safetensors` 가중치가 있는 병합 모델 폴더다. 두 명령은 GPU 한 장에서 순서대로 실행한다. `--smoke`는 첫 8문항을 별도 경로에 저장하는 설치 확인용 실행이다.

학교 실습실에서는 저장소를 호스트의 `~/shared_dir/MMDL`에 받은 뒤 `~/docker/run.sh`로 실행한다. 아래 예시는 기존 `qwen_baseline/mmdl_test/.venv`와 Hugging Face 캐시를 사용한다. 팀원 환경에 맞게 Python과 캐시 경로를 지정한다.

```bash
cd ~/docker
NAME=mmmu-pro-base ./run.sh bash -c '
  cd /shared_dir/MMDL &&
  export HF_HOME=/shared_dir/qwen_baseline/cache/huggingface &&
  exec /shared_dir/qwen_baseline/mmdl_test/.venv/bin/python -u \
    src/eval/mmmu_pro_std10/run_mmmu_pro_std10.py
'
```

학습 모델은 컨테이너 이름을 `mmmu-pro-trained`로 지정하고 Python 명령 끝에 `--checkpoint /shared_dir/models/merged --run-name trained_model`을 추가한다. 같은 실행을 중단 후 재개할 때는 모델 경로와 실행 이름을 그대로 사용한다.

## 결과 확인

출력은 `src/eval/mmmu_pro_std10/outputs/`의 `base_eval` 또는 `finetuned_trained_model_eval` 폴더에 저장된다. `summary.txt`에서 1,730문항 채점 결과를 확인한다. 문항별 비교에는 `records.json`, `predictions.jsonl`, `graded.jsonl`, `judge.jsonl`을 사용한다. `runtime.json`은 실행 시의 패키지 버전, GPU·드라이버와 코드 해시를 담고, 학습 모델 실행의 `checkpoint.json`은 가중치 해시를 담는다.

[기본 모델과 DPO2-E 결과 비교](../../../reports/mmmu_pro_std10/comparison.md)
