# DPO2-F 자기증류→DPO (seed 3407) vs 베이스라인

**DPO2-F 자기증류→DPO (seed 3407): 595/900 = 66.11%** (베이스라인 576/900 = 64.00%, 차이 +19문항 / +2.11%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 19 | +0 |
| Agriculture | 15 | 15 | +0 |
| Architecture_and_Engineering | 11 | 17 | +6 |
| Art | 18 | 17 | -1 |
| Art_Theory | 24 | 24 | +0 |
| Basic_Medical_Science | 23 | 22 | -1 |
| Biology | 17 | 15 | -2 |
| Chemistry | 16 | 18 | +2 |
| Clinical_Medicine | 23 | 17 | -6 |
| Computer_Science | 20 | 20 | +0 |
| Design | 24 | 21 | -3 |
| Diagnostics_and_Laboratory_Medicine | 10 | 9 | -1 |
| Economics | 26 | 26 | +0 |
| Electronics | 19 | 18 | -1 |
| Energy_and_Power | 14 | 19 | +5 |
| Finance | 23 | 22 | -1 |
| Geography | 18 | 16 | -2 |
| History | 23 | 21 | -2 |
| Literature | 25 | 24 | -1 |
| Manage | 18 | 23 | +5 |
| Marketing | 26 | 26 | +0 |
| Materials | 15 | 20 | +5 |
| Math | 18 | 22 | +4 |
| Mechanical_Engineering | 11 | 20 | +9 |
| Music | 6 | 10 | +4 |
| Pharmacy | 21 | 25 | +4 |
| Physics | 25 | 21 | -4 |
| Psychology | 24 | 25 | +1 |
| Public_Health | 26 | 25 | -1 |
| Sociology | 18 | 18 | +0 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 595,
 "incorrect": 305,
 "accuracy": 0.6611111111111111,
 "accuracy_percent": 66.11111111111111,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 820,
  "output_limit_incorrect": 9,
  "local_qwen3_8b_judge_parser": 67,
  "official_parser_fallback": 4
 },
 "length_limit_incorrect": 9,
 "judge_count": 67,
 "judge_states": {
  "valid_option": 47,
  "no_match": 19,
  "invalid_option": 1
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
