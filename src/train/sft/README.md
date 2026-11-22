# ERA SFT

이미지·질문·보기로 구성한 입력에 대해 원문 풀이와 ERA 요약을 답변으로 학습한다. 학습 데이터는 이미지가 한 장씩 있는 2,000문항이며, LoRA를 언어 모델 attention에 적용한다. 매 epoch 체크포인트를 병합해 MMMU validation 900문항으로 평가한다.

## 실습실 경로와 실행

저장소 루트에서 평가 환경을 준비하고, 학습용 가상환경에 PEFT를 설치한다. `QWEN_BASELINE_ROOT`의 기본값은 `/shared_dir/qwen_baseline`이다. 그 경로의 `cache/huggingface`에 기반 모델과 평가 캐시를 준비하고, ERA 2,000문항의 `data.json`과 이미지가 있는 디렉터리를 `--data-dir`로 지정한다.

```bash
.venv/bin/python src/train/sft/train_qwen_era_llm.py \
  --data-dir /shared_dir/qwen_baseline/data/era_2000 \
  --run-dir /shared_dir/qwen_baseline/training/original_plus_era_llm \
  --max-tokens 6144
```

`--check-data`는 같은 데이터 경로에서 문항과 이미지를 점검한다. 같은 실행 경로를 지정하면 저장 상태에서 이어서 진행한다.

이전 SFT 체크포인트는 실습실 저장공간 초기화로 소실됐다. [학습 조건과 epoch별 평가 결과](../../../reports/mmmu.md)를 보관한다.
