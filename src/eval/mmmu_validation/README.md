# MMMU validation 평가

MMMU validation은 30과목의 900문항을 평가한다. 이 폴더에 [평가 실행기](evaluate.py), [환경 준비 코드](prepare.py), [설정](config.json)이 있다.

저장소 루트에서 실행한다.

```bash
python3.12 src/eval/mmmu_validation/prepare.py
.venv/bin/python src/eval/mmmu_validation/evaluate.py --config src/eval/mmmu_validation/config.json --run-name baseline
```

학습한 모델에는 `--checkpoint /absolute/path/to/merged`를 추가하고 새 `--run-name`을 지정한다. 같은 실행을 재개할 때는 이전 명령과 이름을 사용한다. 이동 전 평가를 시작했다면 기록한 코드 해시와 일치하도록 당시 커밋에서 이어서 실행한다.

- [제출용 베이스라인 보고서](../../../assignment/assignment1.md)
