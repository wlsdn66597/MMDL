# Loop-pair mining과 Loop-only DPO

베이스 모델이 스스로 풀다가 반복 루프에 빠진 응답을 모아, 같은 문항의 정상 정답 응답과 쌍으로 만든다. 이 쌍으로 Loop-Segment DPO를 학습한 모델이 발표의 **Loop-only**다. 최종 모델 DPO-ERA도 이 326쌍을 그대로 쓴다.

작성·실행: 김진우. 스크립트의 기본 경로는 작성자의 서버 기준 `~/mmdl/...`이다. 다른 환경에서는 경로만 바꾸면 된다.

## 흐름

```
공개 문항 6,129개 (MMMU 문항 제외)
 → 베이스 모델로 문항당 8번 풀기 (출력 상한 8,192)        49,032 응답
 → 8,192 tokens에서 잘린 응답                            4,171
 → 루프 확인 + 8번 중 2번 이상 정답인 문항                    366
 → 학습 326쌍 / 검증 40쌍
```

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 문제 은행 | `build_bank2.py`, `build_bank3.py`, `build_bank3b.py`, `merge_bank_nonmath.py`, `build_bank_r2.py` | 수학(MMFineReason 계열)과 비수학(의료 영상, 논문 도표, 미술, 악보) 문항 모으기. MMMU와의 문장·이미지 중복은 이 단계에서 걸러 냄 (`mmmu_overlap.py`) |
| 샘플링 | `sample_bank.py` | 베이스라인과 같은 프롬프트·샘플링 설정으로 문항당 8개 생성, 정오·종료 상태 기록 |
| 쌍 만들기 | `build_dpo_pairs.py` | chosen = 정상 종료한 정답 중 가장 짧은 것, rejected = 루프가 가장 빨리 시작된 응답. 루프 시작 위치 `rej_loop_start`를 함께 저장 |
| 학습 | `train_dpo.py` (`train_sft.py`의 공통 함수 사용) | DPO + 정답 NLL, rejected는 루프 시작 후 512 tokens만 비선호 |
| 실행 묶음 | `queue_r3.sh`, `queue_eval.sh` | 학습 → 병합 → MMMU 평가, 다음 라운드 샘플링 |
| 보조 | `arxivqa_prep.py`, `arxivqa_sft.py` | 문제 은행 스크립트가 쓰는 프롬프트·중복 검사 함수 |

루프 판정: 12토큰 구가 앞에 이미 나온 적 있는 위치의 비율이 256토큰 창에서 0.8 이상이고, 그 뒤 전체에서 0.5 이상인 첫 위치를 루프 시작으로 본다. 잘렸지만 루프가 아닌 긴 풀이는 쓰지 않는다.

## 데이터 통계 (`dpo_r4_loop_report.json`)

- 학습 326쌍: 수학 277 · 비수학 49
- 출처: MMR1 153, Geometry3K 40, WaltonColdStart 30, BMMR 26, MedX 20, arXivQA 15, VisualWebInstruct 13, WikiArt 7, LLaVA-CoT 7 외
- 8번 중 정답 횟수: 2회 85 · 3회 66 · 4회 66 · 5회 47 · 6회 36 · 7회 26
- chosen 길이 중앙값 1,663 tokens, 루프 시작 위치 중앙값 3,470 tokens

## 학습 설정 (Loop-only)

β 0.1 · NLL 가중치 1.0 · lr 1e-5 · 4 epoch(84 step) · grad accum 16 · LoRA r16/α32/dropout 0.05 (LLM q·k·v·o·gate·up·down) · seed 3407

```bash
python train_dpo.py --data <dpo_r4_loop>/train.jsonl --val <dpo_r4_loop>/val.jsonl \
  --baseline <baseline_results.json> --diag <평가 코드 diag> --out <출력 폴더> \
  --beta 0.1 --w-gen 1.0 --lr 1e-5 --epochs 4 --grad-accum 16
python train_dpo.py --merge-only --adapter-name adapter --data x \
  --baseline <baseline_results.json> --diag <평가 코드 diag> --out <출력 폴더>
```

쌍 데이터와 이미지는 레포에 넣지 않았다. [SSU-MMDL Hugging Face](https://huggingface.co/SSU-MMDL)를 참고한다.
