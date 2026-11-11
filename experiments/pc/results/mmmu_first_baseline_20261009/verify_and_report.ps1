$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$run = Join-Path $root 'baseline_seed3407'
$result = Get-Content -Raw -LiteralPath "$run/results.json" | ConvertFrom-Json
$summary = Get-Content -Raw -LiteralPath "$run/summary.json" | ConvertFrom-Json
$status = Get-Content -Raw -LiteralPath "$root/status.json" | ConvertFrom-Json
$predictions = @(Get-Content -LiteralPath "$run/predictions.jsonl" | ForEach-Object { ConvertFrom-Json $_ })
$attempts = @(Get-Content -LiteralPath "$run/judge_attempts.jsonl" | ForEach-Object { ConvertFrom-Json $_ })
$old = Get-Content -Raw -LiteralPath 'C:/project/mmdl_test/baseline_results.json' | ConvertFrom-Json
$oldById = @{}; foreach ($r in $old.records) { $oldById[$r.id] = $r }
$records = @($result.records)
$correct = @($records | Where-Object { $_.grading.final.correct }).Count
$length = @($records | Where-Object { $_.model_response.finish_reason -eq 'length' })
$methods = @($records | Group-Object { $_.grading.final.method } | ForEach-Object {
    [ordered]@{ method=$_.Name; total=$_.Count; correct=@($_.Group | Where-Object { $_.grading.final.correct }).Count }
})
$checks = [ordered]@{
    completed = ($status.state -eq 'complete' -and $status.exit_code -eq 0)
    predictions_900_unique = ($predictions.Count -eq 900 -and @($predictions.id | Sort-Object -Unique).Count -eq 900)
    results_900_unique = ($records.Count -eq 900 -and @($records.id | Sort-Object -Unique).Count -eq 900)
    same_prediction_result_ids = (@(Compare-Object $predictions.id $records.id).Count -eq 0)
    all_graded = (@($records | Where-Object { $_.grading.final.status -ne 'graded' }).Count -eq 0)
    summary_matches_records = ($summary.total -eq 900 -and $summary.graded -eq 900 -and $summary.correct -eq $correct -and $summary.incorrect -eq (900-$correct) -and [math]::Abs($summary.accuracy-$correct/900) -lt 1e-12)
    length_all_incorrect = ($length.Count -eq 81 -and @($length | Where-Object { $_.grading.final.correct -or -not $_.grading.final.token_limit_forced_incorrect }).Count -eq 0)
    random_fallback_absent = (@($records | Where-Object { $_.grading.final.random_fallback }).Count -eq 0)
    all_frozen_inputs_match = (Get-Content -Raw -LiteralPath "$root/provenance/input-validation.json" | ConvertFrom-Json).all_frozen_inputs_match
}
foreach ($p in $summary.by_subject.PSObject.Properties) {
    $group = @($records | Where-Object subject -eq $p.Name)
    $checks['subject_'+$p.Name] = ($group.Count -eq 30 -and @($group | Where-Object {$_.grading.final.correct}).Count -eq $p.Value.correct)
}
foreach ($m in $methods) { $checks['method_'+$m.method] = ($summary.method_counts.($m.method) -eq $m.total) }
$judgeRecords = @($records | Where-Object {$_.grading.final.method -eq 'local_qwen3_8b_judge_parser'})
$checks.judge_54_unique = (@($attempts.id | Sort-Object -Unique).Count -eq 54 -and $judgeRecords.Count -eq 54)
$checks.z_stops_immediately = (@($attempts | Where-Object { $_.decision.status -eq 'no_match' -and $_.attempt -ne 1 }).Count -eq 0 -and @($judgeRecords | Where-Object { $_.grading.final.reason -eq 'no_match' -and $_.grading.final.correct }).Count -eq 0)
$changes = @($records | Where-Object {$_.grading.final.correct -ne $oldById[$_.id].grading.final.correct} | ForEach-Object {
    [ordered]@{id=$_.id;old_correct=$oldById[$_.id].grading.final.correct;new_correct=$_.grading.final.correct;old_answer=$oldById[$_.id].grading.final.extracted_answer;new_answer=$_.grading.final.extracted_answer}
})
$rawDiff = @($records | Where-Object {$_.model_response.raw -cne $oldById[$_.id].model_response.raw}).Count
$audit = [ordered]@{checked_at=[DateTimeOffset]::Now.ToString('o');checks=$checks;all_passed=(@($checks.Values | Where-Object {-not $_}).Count -eq 0);method_stats=$methods;raw_response_difference_count=$rawDiff;correctness_changes=$changes;judge_attempt_count=$attempts.Count}
$audit | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 -LiteralPath "$root/verification.json"
if (-not $audit.all_passed) { throw 'Verification failed; see verification.json' }
$subjectLines = @($summary.by_subject.PSObject.Properties | ForEach-Object { '| '+$_.Name+' | '+$_.Value.correct+'/30 | '+('{0:F2}' -f (100*$_.Value.accuracy))+'% |' }) -join "`n"
$methodLines = @($methods | ForEach-Object { '| '+$_.method+' | '+$_.total+' | '+$_.correct+' |' }) -join "`n"
$report = @"
# MMMU 최초 baseline 900문항 평가 결과

2026년 10월 10일 11:21:15 KST에 seed 3407의 최초 900문항 추론과 Parser + LLM Judge 채점이 종료됐다. **575/900 정답, 63.89%**이며 오답은 325문항이다. 이번 결과를 현재 평가 프로토콜의 새 기준점으로 보존한다. 추가 seed나 전체 재실행은 진행하지 않았다.

## 완료 검증

status.json의 complete·exit_code 0, 추론/채점 각각 고유 ID 900개, 전 문항 graded, 문항별 정답 합계와 summary의 전체·과목별·채점 경로별 집계 일치를 확인했다. length 81건은 모두 강제 오답이고 무작위 fallback은 0건이다. 900개 고정 입력 일치 기록도 확인했다. 검증 결과는 ../results/mmmu_first_baseline_20261009/verification.json에 보존했다. 이는 결과 파일과 집계의 정합성 검증이며, Judge 판단의 의미적 타당성을 전 문항 수동 검수했다는 뜻은 아니다.

## 소요 시간 (측정값)

| 구간 | 시간 |
|---|---|
| 마지막 준비 시작 | 2026-10-09 17:51:47 KST |
| evaluate.py 시작 | 2026-10-09 21:37:32 KST |
| 최종 채점 완료 | 2026-10-10 11:21:15 KST |
| 준비·설치·입력 검증 | 3시간 45분 45초 |
| 평가 종단간: 엔진 초기화·900문항 추론·채점 | 13시간 43분 44초 |
| 마지막 준비 시작부터 완료까지 | 17시간 29분 29초 |
| 4B 문항별 생성 시간 합계 | 49,184.224초 (13시간 39분 44초) |
| Judge 78회 생성 시간 합계 | 61.139초 |

종단간 시간은 status.json, 생성 시간 합계는 문항/시도별 elapsed_seconds 기준이다. Judge 생성 시간 합계에는 모델 교체·초기화·재시도 대기가 포함되지 않는다. 총 준비 시간은 마지막 준비 재시작부터이며, 그 이전 설치 시도 시간은 제외된다. 이전 49,513.468초는 과거 4B 생성 시간 합계이므로 이번 종단간 시간과 직접 비교하지 않는다.

## 채점 경로

| 경로 | 문항 수 | 정답 수 |
|---|---:|---:|
$methodLines

Judge 대상은 54문항이며 실제 시도는 78회다. 최종 상태는 유효 선택지 39문항, Z/no_match 14문항, invalid_option 1문항이다. 무효 출력 1문항은 25번 시도했고, Z 14문항은 첫 시도에서 종료·오답 처리됐다. 4B length 종료 81문항(9.00%)은 Parser/Judge보다 먼저 오답 처리했다.

## 과거 64.00%와의 비교

기존 baseline_results.json은 576/900(64.00%), 이번 실행은 575/900(63.89%)으로 **1문항, 0.1111%p 감소**했다. ID별 대조 결과 4B raw 응답은 900개 모두 정확히 같고, 정오가 달라진 문항은 validation_Electronics_6 하나다.

이 문항은 과거 Judge 마지막 저장 출력이 A여서 정답 처리됐으나, 이번 첫 시도는 Z여서 오답 처리됐다. 과거 기록에는 attempt_count=25, 이번에는 attempt=1이 남아 있다. 과거 출력도 계산값 약 10.7과 정답 선택지 71.6의 불일치를 언급하면서 A를 택했고, 기존 데이터에는 그 타당성 검토 메모가 있었다. 이번 Judge는 불일치를 근거로 Z를 출력했다.

이 차이를 4B 모델의 성능 하락으로 해석할 근거는 없다. 과거에는 마지막 Judge 출력만 저장되어 있으므로 모든 재시도 경과 및 차이의 원인을 확정할 수 없다. 이후 개선 모델은 이번 코드·입력·프롬프트·채점 규칙과 실행 환경을 유지한 새 기준점과 비교해야 한다. 단일 seed 1회 결과로 여러 seed에 걸친 개선이나 재현성 보장을 주장하지 않는다.

## 실행 설정과 근거

| 항목 | 실제 설정 |
|---|---|
| 서버 / run | rlatjswo@aix408-40 / baseline_seed3407 |
| 코드 commit | 7740786a1b6b173db86285a00f06e93a44d8a225 |
| 4B 모델 | Qwen/Qwen3-VL-4B-Instruct |
| 4B revision | ebb281ec70b05090aa6165b016eac8ec08e71b17 |
| Judge | Qwen/Qwen3-8B |
| Judge revision | b968826d9c46dd6066d109eabc6255188de91218 |
| 데이터 | MMMU/MMMU validation, 30과목 × 30문항 |
| 데이터 revision | 98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 |
| seed | 4B·Judge 엔진 및 요청 모두 3407 |
| 4B sampling | temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1, presence_penalty 1.5 |
| 4B 출력/문맥/이미지 상한 | 32,768 tokens / 64,000 tokens / 5장 |
| 이미지 pixels | min 1,003,520 / max 4,014,080 |
| 정밀도 / GPU | BF16, 양자화 없음 / RTX 4090 24GB |
| Judge sampling | temperature 0, top_p 1, top_k -1, thinking false |
| Judge 출력/문맥/재시도 상한 | 4,096 tokens / 32,768 tokens / 25회 |
| 스케줄링 환경 | VLLM_ENABLE_V1_MULTIPROCESSING=0, VLLM_BATCH_INVARIANT 미설정 |
| 실제 환경 | Python 3.12.14, vLLM 0.11.0, torch 2.8.0 (CUDA 12.8), transformers 4.57.0 |
| 기타 고정 라이브러리 | datasets 4.1.1, qwen-vl-utils 0.0.14, Pillow 11.3.0, huggingface-hub 0.36.2 |
| 드라이버 | 595.91.07 |

4B에는 Final answer 지시를 추가했다. length 오답 → 최종 답 우선 Parser → 공식 fallback → 필요한 문항만 Judge 순서로 채점했다. Judge는 보존된 공식 프롬프트와 프로젝트 출력 파서를 사용하고, Z는 즉시 오답, 재시도 후 무효 출력도 오답, 무작위 fallback은 사용하지 않았다. 공식 소스 commit은 96588727e44c78b25ba03ea03b8e12f7e64fd0da이다. 이 실행은 HF 데이터 입력, 프롬프트, seed, context, 이미지 상한 및 채점 세부 규칙 차이가 있는 프로젝트 baseline이며 공식 점수의 완전 재현으로 표기하지 않는다.

manifest.json에는 전체 설정·소스 SHA256·가중치 SHA256, provenance에는 패키지 목록·GPU·모델/데이터 준비·고정 입력 및 이미지 RGB hash·코드 snapshot을 보존했다. Windows CRLF와 Linux LF 차이는 준비 단계에서 분리 검증했다.

## 과목별 결과

| 과목 | 정답/문항 | 정확도 |
|---|---:|---:|
$subjectLines

모든 과목이 30문항이므로 과목별 정확도의 단순 평균과 전체 575/900은 같다.

## 저장 위치

- 로컬 결과·로그·provenance: C:/project/MMDL/results/mmmu_first_baseline_20261009/
- 최종 결과: 위 경로의 baseline_seed3407/results.json 및 summary.json
- 검증 결과와 검증 스크립트: verification.json 및 verify_and_report.ps1
- 원격 원본: /home/rlatjswo/mmmu_baseline_20261009/
- 평가 절차: C:/project/MMDL/reports/mmmu_evaluation_protocol.md

실행 완료 후 GPU 사용률 0%, 사용 메모리 24 MiB, driver 프로세스 종료를 확인했다(2026-10-10 13:31 KST).
"@
$report | Set-Content -Encoding utf8 -LiteralPath 'C:/project/MMDL/reports/mmmu_first_baseline_result_20261009.md'
$audit | ConvertTo-Json -Depth 5
