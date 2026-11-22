# 배포 전 검증 기록

검증일: 2026-11-21. 이 문서는 패키지의 로컬 검증 범위를 기록한다.

| 검사 | 결과 |
|---|---|
| 원본 평가 파일 9개 SHA256 | 보존된 소스와 일치 |
| 원본 MMMU Test 문항 파일 | SHA256 일치, 고유 ID 10,500개, 객관식 9,873 / 주관식 627 |
| 이전 DPO2-E seed 3407 저장 응답 재채점 | 10,500개 전부 비교, 정답 6,918개, 판정 불일치 0개 |
| CPU 단위 검사 | 14개 통과: seed·모델 혼용, 변경된 가중치, 중복·누락 ID, 다른 split, 깨진 JSONL, 불완전 채점, 잘못된 tag 등 |
| Python 파일 컴파일 검사 | 통과 |
| Bash 구문 검사 | prepare_test.sh, run_test.sh, 원본 run_eval_dpo2e.sh 통과 |

재채점은 원본 응답과 저장된 Judge 판정을 새 패키지의 보존된 채점기로 다시 계산한 것이다. Judge 추론을 다시 수행한 검증은 아니다.

**새 GPU 머신에서의 전체 설치·모델 생성은 이 패키지를 준비하는 과정에서 실행하지 않았다.** GPU 추론 코드 자체는 현재 408-06 실행에 쓰인 소스이고, 새 배포용 준비·검증 wrapper는 CPU 검사 범위에서 확인했다. 친구 서버에서 `prepare_test.sh`의 고정 버전·GPU 검사와 문항 해시 검사를 통과해야 평가가 시작된다.

로컬 기본 검사:

```bash
cd src/eval/mmmu_test
python -m unittest discover -s tests -v
python scripts/protocol.py source-check
python -m compileall -q scripts
bash -n scripts/prepare_test.sh
bash -n scripts/run_test.sh
bash -n scripts/run_eval_dpo2e.sh
```

데이터 준비 후 문항 검증 및 평가 완료 후 독립 재채점:

```bash
.venv/bin/python scripts/protocol.py records-check records/mmmu_test.json
.venv/bin/python scripts/protocol.py verify era_plus_test_seed42
```
