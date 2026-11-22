# ERA SFT

이미지·질문·보기로 구성한 입력에 대해 원문 풀이와 ERA 요약을 답변으로 학습한다. 학습 데이터는 이미지가 한 장씩 있는 2,000문항이며, LoRA를 언어 모델 attention에 적용한다. 매 epoch 체크포인트를 병합해 MMMU validation 900문항으로 평가한다.

## 학습 답변 형식

입력은 기존 MMMU 베이스라인과 같은 형식으로 구성한다. 학습 코드는 `explanation`의 원문 풀이에 `explanation_era`의 요약을 이어 붙이고 마지막 줄에 최종 답을 한 번 넣는다. 손실은 답변 토큰에만 적용하며 시각 인코더와 이미지 연결부를 고정한다.

```text
[원문 풀이]

To summarize the reasoning in Evidence, Rule, and Application form:
Evidence: [이미지와 문항에서 확인한 근거]
Rule: [적용할 규칙]
Application: [근거에 규칙을 적용한 풀이]
Final answer: [최종 답]
```

## 실습실 경로와 실행

저장소 루트에서 평가 환경을 준비하고, 학습용 가상환경에 PEFT를 설치한다. `QWEN_BASELINE_ROOT`의 기본값은 `/shared_dir/qwen_baseline`이다. 아래 명령은 학습 코드가 사용하는 위치에 기반 모델과 평가 캐시를 준비한다.

```bash
export QWEN_BASELINE_ROOT=/shared_dir/qwen_baseline
python3.12 src/eval/mmmu_validation/prepare.py \
  --model-cache "$QWEN_BASELINE_ROOT/cache/huggingface/hub" \
  --dataset-cache "$QWEN_BASELINE_ROOT/cache/huggingface/datasets"
.venv/bin/python -m pip install -c requirements.txt peft
```

Hugging Face에서 [vic_era_2000.zip](https://huggingface.co/datasets/SSU-MMDL/MMMU_ERA_Dataset/blob/main/vic_era_2000.zip)을 다운로드한다. 데이터 공개 범위가 비공개로 표시되면 조직 권한이 있는 계정으로 로그인한다. 다운로드한 ZIP의 경로를 지정하고, 압축을 풀어 `data.json`과 `images/`를 준비한다.

```bash
DATA_DIR="$QWEN_BASELINE_ROOT/data/era_2000"
mkdir -p "$DATA_DIR"
.venv/bin/python -m zipfile -e /path/to/vic_era_2000.zip "$DATA_DIR"
.venv/bin/python src/train/sft/train_qwen_era_llm.py --check-data --data-dir "$DATA_DIR"
```

`--check-data`는 모델을 로드하기 전에 원문 풀이와 ERA 답변 형식, 2,000개 문항과 연결 이미지를 확인한다. 로컬 보관본 검증에서 출력된 문항·이미지 지문은 `d95a7de6c673ff9d0fced6097480b72cc5f679fb05464c34bc6109076a03bcd5`이다.

실제 실험은 **5 epoch, seed `3407`, 최대 전체 시퀀스 길이 `6144`토큰**으로 진행했다. 같은 셸에서 아래 명령을 실행한다.

```bash
.venv/bin/python src/train/sft/train_qwen_era_llm.py \
  --data-dir "$DATA_DIR" \
  --run-dir "$QWEN_BASELINE_ROOT/training/original_plus_era_llm" \
  --max-tokens 6144
```

학습 코드는 매 epoch의 LoRA와 재개 상태를 저장하고, 병합 모델로 MMMU validation을 평가한다. 같은 코드·설정·실행 경로를 지정하면 저장 상태에서 이어서 진행한다. validation 코드 이동 전에 시작한 실행은 당시 커밋에서 재개한다.

이전 SFT 체크포인트는 실습실 저장공간 초기화로 소실됐다. [학습 조건과 epoch별 평가 결과](../../../reports/mmmu.md)를 보관한다.
