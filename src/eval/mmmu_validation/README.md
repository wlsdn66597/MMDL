# MMMU validation 평가

MMMU validation은 30과목의 900문항을 평가한다. 팀에서 사용해 온 [평가 실행기](../../evaluate.py), [환경 준비 코드](../../prepare.py), [설정](../../config.json)은 `src/`의 기존 경로에 있다.

저장소 루트에서 실행한다.

```bash
python3.12 src/prepare.py
.venv/bin/python src/evaluate.py --config src/config.json --run-name baseline
```

학습한 모델에는 `--checkpoint /absolute/path/to/merged`를 추가하고 새 `--run-name`을 지정한다. 같은 실행을 재개할 때는 이전 명령과 이름을 사용한다.

- [제출용 베이스라인 보고서](../../../assignment/mmmu_baseline.md)
- [ERA SFT epoch별 결과](../../../reports/mmmu.md)
