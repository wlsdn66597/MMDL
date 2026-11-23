# DPO-ERA (최종 모델, 코드명 DPO2-E)

Loop-only의 루프 쌍에 ERA 형식 선호 쌍과 보존 데이터를 섞고, 시각 연결부 LoRA·gate까지 **한 번의 DPO 학습**으로 함께 학습한다. 결과는 [`reports/dpo_era/`](../../../reports/dpo_era/README.md).

## 구성

| 폴더 | 내용 |
|---|---|
| `scripts/` | 문항 준비(`prepare_sft_general.py`), 베이스 샘플링(`dpo_sample.py`), 채점(`dpo_grade.py`), 쌍 만들기(`dpo_pairs2.py` → `dpo_pairs_d.py` → `dpo_pairs_e.py`), 학습(`train_dpo_d.py`, 공통 함수 `train_sft.py`), 평가 요약(`summarize_eval.py`, `eval_extra.py`) |
| `pipeline/` | 처음부터 끝까지 한 번에 돌리는 `run_repro_e.sh`, 학습 seed만 바꾸는 `run_seed_e.sh`, 경로 설정 `env.sh`, 고정 하이퍼파라미터 `hparams.json`, 쌍 통계 `data_report/` |

스크립트 경로는 실습실 노드 기준(`/shared_dir/...`)이다. 다른 환경에서는 `pipeline/env.sh`만 바꾸고, `scripts/`의 파일을 `$ROOT/scripts/`에 두면 된다.

## 데이터 (pairs_E.jsonl, 2,307행)

| 종류 | 행 | chosen (선호) | rejected (비선호) |
|---|---:|---|---|
| 루프 쌍 | 1,304 (326 × 4) | 같은 문항의 정상 종료 정답 | 루프 응답, **루프 시작 후 512 tokens만** 계산 |
| ERA 쌍 | 703 (V 183 · I 205 · C 315) | ERA 지시를 붙여 베이스가 쓴 풀이 중 형식을 지키고 맞히고 정상 종료한 것 | 지시 없이 베이스가 쓴 오답 (전체) |
| 보존 | 300 | 8번 모두 맞힌 문항의 풀이 (ERA 297 · 일반 3) | 없음, NLL만 |

- ERA 문항: VIC-ERA 2,000 (MMFineReason 1,456 · OpenMedReason 544, V 400 · I 600 · C 1,000). MMMU 11,550문항과 문장·이미지 중복 0건.
- ERA 지시("Evidence / Rule / Application으로 나눠 추론하라")는 **응답을 만들 때만** 썼다. 학습 입력과 평가 프롬프트는 원래 질문 그대로다.
- 채점은 평가와 같은 파서, 실패하면 Qwen3-8B judge.

## 손실

행마다 종류에 맞는 손실을 계산하고 16행씩 모아 한 번 업데이트한다.

- 루프 쌍: `-log σ(β[(logπθ(y+) − logπref(y+)) − Σ_{t∈W}(logπθ(y−_t) − logπref(y−_t))]) + λ·NLL(y+)`, W = [t₀, t₀+512)
- ERA 쌍: 같은 DPO + NLL, W = 응답 전체. V·I 쌍은 원본 이미지 ≻ 50% 가린 이미지 조건 손실 추가 (mDPO)
- 보존: λ·NLL만
- β 0.1, λ 1.0, πref = LoRA를 끈 베이스

## 학습 설정

LoRA r16/α32/dropout 0.05 (LLM q·k·v·o·gate·up·down + `merger`, `deepstack_merger_list.*`) · DeepStack gate g = 1 + tanh(w) 3개 (w = 0 시작) · AdamW lr 1e-5 (gate 1e-3), weight decay 0.01, clip 1.0 · warmup 3% → cosine · 1 epoch 145 step · max tokens 8,400 · bf16 · RTX 4090 1장 (최대 16.6GB, 약 2시간)

학습 후 LoRA는 W에 병합하고 gate는 `deepstack_merger.linear_fc2`에 곱해 흡수한다. 평가 모델 구조는 베이스와 같다.

```bash
SEED=3407 bash pipeline/run_seed_e.sh      # 샘플링 → 채점 → 쌍 → 학습 → MMMU-val 평가
```
```bash
# 쌍과 이미지가 준비된 경우 학습만
python scripts/train_dpo_d.py --pairs pairs_E.jsonl --images <vic_era_2000> --start-model <Qwen3-VL-4B-Instruct> \
  --vision-config <training_settings_only.json> --diag scripts --out out/dpo_era --beta 0.1 --lr 1e-5 --epochs 1 \
  --grad-accum 16 --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 --lora-r 16 --lora-alpha 32 \
  --lora-dropout 0.05 --max-tokens 8400 --seed 3407 --workers 2 --rpo-alpha 1.0 \
  --vision-lora --gates --gate-lr 1e-3 --img-pref-types V,I --replay-weight 1.0
```
