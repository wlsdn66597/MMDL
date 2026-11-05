# Qwen3-VL decoder LoRA 학습

일반 풀이와 ERA 풀이의 차이를 비교하는 200문항 파일럿이다. 두 브랜치는 동일한 모델에서 시작하고, 같은 순서의 문제와 이미지에 서로 다른 assistant 정답 설명을 학습한다. `era_lora/`는 실제 4epoch 학습의 Python 소스 7개다. 파일의 SHA256은 `snapshot.sha256`에 기록했다. 브랜치별 실행 명령은 [TRAIN.md](TRAIN.md)에 있다.

## 환경과 데이터

Linux, Python 3.12, BF16을 지원하는 단일 CUDA GPU를 사용한다. 아래 명령은 이 디렉터리에서 실행한다.

```bash
cd code/training/era_decoder_lora
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
python -m pip install -r requirements-training.txt
sha256sum -c snapshot.sha256
```

권한이 있는 사용자는 기존 200문항 패키지를 `data/pilot200/`에 둔다. 두 브랜치 모두 `exports/train_plain.jsonl`, `exports/train_era.jsonl`, `validation_status.json`과 JSONL의 `image_path`가 가리키는 이미지를 요구한다. 실행 시 두 파일의 문항 ID·순서·문제·이미지 해시·최종 정답을 대조한다.

공통 설정은 `configs/paired_4epoch.json`이다. 모델은 Qwen3-VL-4B-Instruct의 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`로 고정했다. 기본값은 로컬 Hugging Face 캐시를 사용한다. 다른 경로를 쓰려면 설정을 `configs/paired.local.json`으로 복사해 `model.local_path`를 해당 revision의 snapshot 디렉터리로, `dataset.root`를 데이터 패키지 위치로 지정한다. 상대 경로는 config 디렉터리 기준이다. 두 브랜치에서 동일한 설정 파일과 데이터 경로를 사용한다.

## 비교 조건

| 항목 | 설정 |
|---|---|
| 학습 대상 | decoder의 q/k/v/o projection LoRA, assistant 응답 토큰만 loss 계산 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| 정밀도 | BF16, 양자화 없음 |
| 학습률·스케줄 | AdamW, 1e-4, weight decay 0.01, cosine, warmup 3% |
| 배치 | micro batch 1, gradient accumulation 16 |
| 길이와 이미지 | 이미지 포함 최대 4,096 토큰, 문제당 1장, 262,144~1,048,576 pixels |
| 학습 | 4epoch, seed 3407, gradient checkpointing 사용, packing 없음 |

원 실험은 총 52 optimizer step이며 2·3·4epoch 체크포인트는 각각 26·39·52 step이다. 팀원이 사용할 수 있도록 공개 설정의 머신별 경로를 바꿨다. 따라서 원 서버의 config 해시와 학습 run identity는 달라지며, 새 실행은 별도의 출력 폴더에서 시작한다. 학습 코드가 공유하는 평가·비교 모듈도 스냅샷에 포함한다. 이 배포의 실행 범위는 LoRA 학습이다.
