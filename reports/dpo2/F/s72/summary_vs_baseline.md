# DPO2-F 자기증류→DPO (seed 72) vs 베이스라인

**DPO2-F 자기증류→DPO (seed 72): 599/900 = 66.56%** (베이스라인 576/900 = 64.00%, 차이 +23문항 / +2.56%p)

| 과목 | 베이스라인 | 이번 | 차이 |
|---|---:|---:|---:|
| Accounting | 19 | 22 | +3 |
| Agriculture | 15 | 15 | +0 |
| Architecture_and_Engineering | 11 | 19 | +8 |
| Art | 18 | 18 | +0 |
| Art_Theory | 24 | 25 | +1 |
| Basic_Medical_Science | 23 | 22 | -1 |
| Biology | 17 | 16 | -1 |
| Chemistry | 16 | 18 | +2 |
| Clinical_Medicine | 23 | 19 | -4 |
| Computer_Science | 20 | 22 | +2 |
| Design | 24 | 22 | -2 |
| Diagnostics_and_Laboratory_Medicine | 10 | 8 | -2 |
| Economics | 26 | 26 | +0 |
| Electronics | 19 | 19 | +0 |
| Energy_and_Power | 14 | 17 | +3 |
| Finance | 23 | 24 | +1 |
| Geography | 18 | 18 | +0 |
| History | 23 | 24 | +1 |
| Literature | 25 | 24 | -1 |
| Manage | 18 | 21 | +3 |
| Marketing | 26 | 26 | +0 |
| Materials | 15 | 17 | +2 |
| Math | 18 | 21 | +3 |
| Mechanical_Engineering | 11 | 16 | +5 |
| Music | 6 | 9 | +3 |
| Pharmacy | 21 | 24 | +3 |
| Physics | 25 | 23 | -2 |
| Psychology | 24 | 24 | +0 |
| Public_Health | 26 | 22 | -4 |
| Sociology | 18 | 18 | +0 |

## 기타 집계 (summary.json)

```json
{
 "total": 900,
 "graded": 900,
 "correct": 599,
 "incorrect": 301,
 "accuracy": 0.6655555555555555,
 "accuracy_percent": 66.55555555555554,
 "aggregation": "unweighted mean of 30 subject accuracies",
 "method_counts": {
  "explicit_final_parser": 811,
  "output_limit_incorrect": 7,
  "local_qwen3_8b_judge_parser": 79,
  "official_parser_fallback": 3
 },
 "length_limit_incorrect": 7,
 "judge_count": 79,
 "judge_states": {
  "valid_option": 52,
  "no_match": 25,
  "unparsed": 1,
  "invalid_option": 1
 },
 "random_fallback": 0,
 "exact_official_report_reproduction": false
}
```
