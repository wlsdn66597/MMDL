# DPO2-F 자기증류→DPO (seed 42) vs 베이스라인

**DPO2-F 자기증류→DPO (seed 42): 611/900 = 67.89%** (베이스라인 576/900 = 64.00%, 차이 +35문항 / +3.89%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 20 | +1 |
| Agriculture | 15 | 17 | +2 |
| Architecture_and_Engineering | 11 | 18 | +7 |
| Art | 18 | 20 | +2 |
| Art_Theory | 24 | 23 | -1 |
| Basic_Medical_Science | 23 | 20 | -3 |
| Biology | 17 | 16 | -1 |
| Chemistry | 16 | 19 | +3 |
| Clinical_Medicine | 23 | 19 | -4 |
| Computer_Science | 20 | 21 | +1 |
| Design | 24 | 24 | +0 |
| Diagnostics_and_Laboratory_Medicine | 10 | 11 | +1 |
| Economics | 26 | 24 | -2 |
| Electronics | 19 | 18 | -1 |
| Energy_and_Power | 14 | 19 | +5 |
| Finance | 23 | 25 | +2 |
| Geography | 18 | 18 | +0 |
| History | 23 | 22 | -1 |
| Literature | 25 | 24 | -1 |
| Manage | 18 | 19 | +1 |
| Marketing | 26 | 27 | +1 |
| Materials | 15 | 16 | +1 |
| Math | 18 | 18 | +0 |
| Mechanical_Engineering | 11 | 22 | +11 |
| Music | 6 | 11 | +5 |
| Pharmacy | 21 | 27 | +6 |
| Physics | 25 | 24 | -1 |
| Psychology | 24 | 25 | +1 |
| Public_Health | 26 | 26 | +0 |
| Sociology | 18 | 18 | +0 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 611,
 "incorrect": 289,
 "accuracy": 0.6788888888888889,
 "accuracy_percent": 67.88888888888889,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 818,
  "local_qwen3_8b_judge_parser": 73,
  "output_limit_incorrect": 8,
  "official_parser_fallback": 1
 },
 "length_limit_incorrect": 8,
 "judge_count": 73,
 "judge_states": {
  "valid_option": 54,
  "no_match": 19
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
