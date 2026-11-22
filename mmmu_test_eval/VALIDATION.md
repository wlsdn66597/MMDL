# 배포 및 업로드 후 검증

검증일: 2026-11-22. 로컬 CPU 검사와 저장 결과 재채점 기록이다. GPU 추론을 새로 실행하지 않았다.

| 검사 | 결과 |
| --- | --- |
| 원본 생성·문항·채점·환경 파일 | 9개 SHA256 원본과 일치 |
| Git 저장 시 줄바꿈 | core.autocrlf=true에서도 9개 원본 바이트 보존 |
| 기준 MMMU Test 문항 | 해시 일치, 고유 ID 10,500개 |
| CPU 테스트 | 각 브랜치 **30개 통과**, 건너뛴 테스트 없음 |
| 실제 생성기 모델 선택 | 공식 repo+revision / 지정 병합 경로 분기 확인 |
| 실제 생성기 입력 | Test split·렌더링된 프롬프트·이미지 설정·생성 seed 전달 확인 |
| Judge | 생성 seed 42여도 Judge seed 3407, thinking 비활성 확인 |
| 실행 진입점 | seed 3407·기본 42·공백 포함 모델 경로·인자 오류·잠금 실패 처리 확인 |
| Python·Bash·README 예시 | 전체 문법 검사 통과 |
| Baseline seed 3407 | 10,500문항 재채점 필드 전부 일치, 6,469개 정답 |
| DPO-E seed 3407 | 10,500문항 재채점 필드 전부 일치, 6,918개 정답 |
| DPO-E seed 42 | 10,500문항 재채점 필드 전부 일치, 6,929개 정답 |

## 검사 범위

`test_execution.py`는 실제 생성기·Judge·셸 진입점을 실행하되 모델 추론, 이미지 라이브러리, 잠금 인터페이스를 테스트용 대역으로 대체한다. GPU 연산, 실제 파일 잠금의 OS 동작, 다운로드와 환경 설치 성공을 검증한 것은 아니다. 생성기의 모델 선택·split·프롬프트·샘플링 인자를 검사한다.

기존 31,500개 응답의 재채점은 보관한 Judge 판정을 재사용했다. 모든 저장 채점 필드가 일치했으며 Judge 추론이나 답안의 의미적 정답 여부를 새로 확인한 것은 아니다. 주관식 문자열 포함 판정과 보조 Judge의 알려진 한계는 원본과 동일하다.

## 재실행

패키지 안에서 Python 표준 라이브러리와 Bash로 실행한다. GPU 패키지는 테스트용 대역을 사용하므로 필요하지 않다.

```bash
python -B -m unittest discover -s tests -v
python scripts/protocol.py source-check
bash -n scripts/prepare_test.sh
bash -n scripts/run_test.sh
bash -n scripts/run_eval_dpo2e.sh
```

Bash가 없으면 셸 진입점 검사는 건너뛰며, 전체 30개 검증을 위해 Bash를 설치해야 한다. 실제 평가는 Linux의 `flock`(util-linux)이 필요하다. 실행 도구가 없을 때와 다른 평가가 잠금을 점유할 때의 오류를 구분한다.

실제 서버의 완료 여부는 마지막 `outputs/<tag>.verification.json`을 확인한다. 새 머신의 전체 설치·모델 다운로드·GPU 추론·로그아웃 지속 실행은 이번 검증에서 수행하지 않았다.
