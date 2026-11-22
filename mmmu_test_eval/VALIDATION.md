# 배포 전 검증

검증일: 2026-11-22. GPU 평가를 새로 실행한 결과가 아니라 **배포 파일의 로컬 검증 기록**이다.

| 검사 | 결과 |
| --- | --- |
| 수신한 원본 평가 파일 9개 SHA256 | 브랜치 패키지와 바이트 단위 일치 |
| 기준 Test 문항 파일 | 기존 해시 일치, 10,500개 고유 ID |
| CPU 검증 테스트 | 각 브랜치 22개 통과 |
| Python 문법 | 모든 Python 파일 컴파일 통과 |
| Bash 구문 | prepare_test.sh, run_test.sh, 원본 실행기 통과 |
| README 실행 예시 | 모든 Bash 코드 블록 구문 통과 |
| Baseline seed 3407 재채점 | 10,500개 모든 채점 필드 일치, 정답 6,469 |
| DPO2-E seed 42 재채점 | 10,500개 모든 채점 필드 일치, 정답 6,929 |
| 배포물 검사 | 모델 가중치·캐시·실제 응답·문항·정답 데이터 미포함 |

두 브랜치의 생성·파서·채점 코드 9개는 동일하다. 실제 결과 재채점은 이 공통 채점 코드로 수행했다. 저장된 Judge 판정을 재사용했으며 Judge GPU 추론이나 답안의 의미적 정답 여부를 새로 검증한 것은 아니다.

새 머신에서의 환경 전체 설치, 모델 다운로드, 10,500문항 GPU 추론, 실제 로그아웃 후 지속 실행은 이번 배포 준비에서 실행하지 않았다. 로컬 검사만으로 다른 GPU의 완전 동일 응답을 보장하지 않는다.

```bash
python -B -m unittest discover -s tests -v
python scripts/protocol.py source-check
bash -n scripts/prepare_test.sh
bash -n scripts/run_test.sh
bash -n scripts/run_eval_dpo2e.sh
```

실제 서버에서는 `prepare_test.sh`가 Python 3.12·고정 패키지·GPU·문항 해시를 확인하고, 실행 시작 시 모델 전체 해시를 기록한다. 결과는 마지막 `verification.json`으로 확인한다.
