# Ablation

DPO-ERA에서 요소 하나만 바꾸고 나머지(데이터 순서, seed 3407, 학습률, 평가 설정)는 그대로 둔 채 다시 학습한다. 평가는 MMMU-val 900문항, 같은 harness.

| 변형 | 바꾼 것 | 확인하려는 것 | 실행 |
|---|---|---|---|
| (a) Prefix scoring | 루프 응답을 처음부터 비선호로 계산 (512 window 제거) | 루프 구간으로 한정하는 게 필요한가 | 김진우 |
| (b) − Visual components | connector LoRA · gate · 이미지 조건 손실 제거 | 시각 연결부 학습이 기여하는가 | `run_ablation_b_no_visual.sh` (안영찬) |
| (c) Main only | 루프 쌍 × 4 + NLL만 (ERA 쌍 · 이미지 손실 · connector · replay 제거) | 보조 요소 전체가 추가 이득을 주는가 | 김진우 |

(b)는 `train_dpo_d.py`에서 `--vision-lora --gates --img-pref-types V,I` 세 옵션만 뺀 것이다. 학습 파라미터는 3,398만 → 3,303만 개(LLM LoRA만)가 된다.

결과는 [`README`의 Ablation](../../../README.md#5-ablation) 표에 정리한다.
