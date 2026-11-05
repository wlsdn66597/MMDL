# ERA 풀이 학습

이 브랜치는 200개 문제의 Evidence, Rule, Application으로 구성한 ERA 풀이를 assistant target으로 사용한다. 입력 질문과 이미지는 비교 브랜치와 동일하다. 환경과 데이터 준비는 [공통 안내](README.md)를 참고한다.

이 디렉터리에서 먼저 데이터 쌍과 학습 계획을 확인한다. 출력에서 arm이 `era`, 항목 수가 200, 유효 배치가 16, epoch가 4인지 확인한다.

```bash
python train_arm.py
```

실제 GPU 학습은 다음 명령으로 시작한다. 모델을 로컬에 처음 받는 경우에는 `--allow-download`를 추가한다. 사용자 설정을 쓸 때는 두 명령 모두에 `--config configs/paired.local.json`을 추가한다.

```bash
python train_arm.py --execute
```
