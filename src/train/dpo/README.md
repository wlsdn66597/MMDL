# DPO 학습 코드: DPO2-E, DPO2-F

결과는 [`reports/dpo2/README.md`](../../../reports/dpo2/README.md).

## 구성
| 폴더 | 내용 |
|---|---|
| `common/` | 공통 코드. 베이스 샘플링(`dpo_sample.py`), 채점(`dpo_grade.py`), 쌍 만들기(`dpo_pairs2.py` B, `dpo_pairs_d.py` D, `dpo_pairs_e.py` E), DPO 학습(`train_dpo_d.py`), SFT(`train_sft.py`), 평가 요약(`summarize_eval.py`, `eval_extra.py`) |
| `dpo2_E/` | E 파이프라인. 원래 실행 스크립트(`run_dpo2*.sh`)와 실습실 노드 재현 스크립트(`run_repro_e.sh`, seed 반복 `run_seed_e.sh`, 경로 설정 `env.sh`), 하이퍼파라미터(`hparams.json`), 쌍 통계(`data_report/`) |
| `dpo2_F/` | F 파이프라인(`run_F_cd.sh`, seed 반복 `run_F_seeds.sh`), 자기증류 데이터 통계 |

스크립트의 경로는 실습실 노드 기준(`/shared_dir/...`)이다. 다른 환경에서는 `dpo2_E/env.sh`의 경로만 바꾸면 된다. 코드는 `common/`의 파일들이 `$ROOT/scripts/`에 있다고 가정한다.

## DPO2-E (최종 모델)
베이스 Qwen3-VL-4B-Instruct에서 DPO 1 epoch (145 step).
- 데이터 `pairs_E.jsonl` 2,307행 (데이터는 Hugging Face 데이터셋에 별도 업로드)
  - 루프 쌍 326 × 4 = 1,304: 같은 문항의 정상 종료 정답 풀이 ≻ 반복 루프 풀이. rejected는 루프 시작(`rej_span_char`)부터 512토큰(`rej_span_tokens`)만 로그확률에 포함
  - ERA 정답 ≻ 베이스 오답 703 (V 183 / I 205 / C 315)
  - 보존(replay) 300 (`kind: replay`, rejected 없음 → NLL만)
- 손실: `-log σ(β[(logπ(c)-logπref(c)) - (logπ(r)-logπref(r))])` + 1.0 × chosen NLL + V·I 쌍 이미지 조건 손실(원본 ≻ 50% 블록 가림) + 1.0 × replay NLL. β 0.1, π_ref = LoRA를 끈 시작 모델
- 학습: LoRA r16/α32/dropout 0.05 (LLM q·k·v·o·gate·up·down + `merger`, `deepstack_merger_list.*`의 fc1·fc2), DeepStack 게이트 g = 1 + tanh(w) 3개 (w = 0 시작, lr 1e-3), lr 1e-5 cosine, warmup 3%, grad accum 16, clip 1.0, max tokens 8,400
- 병합: LoRA는 W + (α/r)BA, 게이트는 `deepstack_merger.linear_fc2`의 weight·bias에 곱해 흡수 → 표준 Qwen3-VL 구조

```bash
SEED=3407 bash dpo2_E/run_seed_e.sh     # 학습 + MMMU-val 평가, 결과 runs/dpo2/E_s3407
```
```bash
# 학습만 (pairs_E.jsonl 과 이미지가 준비된 경우)
python common/train_dpo_d.py --pairs pairs_E.jsonl --images <vic_era_2000 폴더> --start-model <Qwen3-VL-4B-Instruct 경로> \
  --vision-config <training_settings_only.json> --diag common --out out/E --beta 0.1 --lr 1e-5 --epochs 1 \
  --grad-accum 16 --warmup-ratio 0.03 --weight-decay 0.01 --max-grad-norm 1.0 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --max-tokens 8400 --seed 3407 --workers 2 --rpo-alpha 1.0 --vision-lora --gates --gate-lr 1e-3 --img-pref-types V,I --replay-weight 1.0
```

## DPO2-F (자기증류 → DPO, 가설 검증용)
1. 자기증류 SFT: 베이스가 ERA 지시("Evidence / Rule / Application")를 받고 쓴 풀이 중 형식을 지키고, 정답이며, 정상 종료한 것(문항당 최대 2개, 6,144토큰 이하) 3,040개를 **지시를 뺀 원래 질문**에 붙여 학습 (LoRA, lr 2e-5, 1 epoch, 190 step) → 병합
2. 1의 모델에서 시작해(기준 모델도 1) E와 같은 데이터·설정으로 DPO

```bash
SEED=3407 bash dpo2_F/run_F_cd.sh      # SFT → DPO → MMMU-val 평가, 결과 runs/dpo2_F_cd/s3407
```
결과: 답의 71~73%가 ERA 형식으로 바뀌었지만 3 seed 평균 601.7 ± 8.3으로 E(607.3 ± 5.0)보다 높지 않다.
