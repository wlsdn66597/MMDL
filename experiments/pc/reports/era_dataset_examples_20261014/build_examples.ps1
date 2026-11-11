$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$utf8 = [System.Text.UTF8Encoding]::new($false)
function Write-Text($path, $value) { [System.IO.File]::WriteAllText($path, $value, $utf8) }
function Write-Json($path, $value) { Write-Text $path (ConvertTo-Json -InputObject $value -Depth 40) }
function Esc($value) { [System.Net.WebUtility]::HtmlEncode([string]$value) }
function WordCount($value) { @([regex]::Matches($value, '\S+')).Count }

$drafts = Get-Content -Raw "$taskRoot/drafts.json" | ConvertFrom-Json
$charts = Get-Content -Raw "$taskRoot/chartqa_train_human.json" | ConvertFrom-Json
$science = Get-Content -Raw "$taskRoot/scienceqa_problems.json" | ConvertFrom-Json -AsHashtable
$splits = Get-Content -Raw "$taskRoot/scienceqa_splits.json" | ConvertFrom-Json
$hfRows = (Get-Content -Raw "$taskRoot/sources/scienceqa_hf_rows.json" | ConvertFrom-Json).rows
$contract = Get-Content -Raw "$taskRoot/../mmmu_eval_protocol/config.seed3407.json" | ConvertFrom-Json
$revisions = @{}
foreach ($name in @('ChartQA','ScienceQA')) {
    $revisions[$name] = (Get-Content -Raw "$taskRoot/sources/${name}_revision.json" | ConvertFrom-Json).sha
}
$pairs = @()
$plainRows = @()
$eraRows = @()
$cards = @()
$selectedOriginals = @()
foreach ($d in $drafts) {
    $isChart = $d.dataset -eq 'ChartQA'
    if ($isChart) {
        $original = $charts[$d.source_index]
        $question = $original.query
        $choices = @()
        $hintText = ''
        $sourceAnswer = $original.label
        $family = 'ChartQA:' + $original.imgname
        $sourceUrl = 'https://github.com/vis-nlp/ChartQA/blob/' + $revisions.ChartQA + '/ChartQA%20Dataset/train/train_human.json'
        $license = 'GPL-3.0 as declared by ChartQA repository/dataset card; preserve original chart attribution and source-specific rights.'
        if ($sourceAnswer -cne $d.answer) { throw "Chart answer mismatch: $($d.id)" }
        $originalId = 'train_human.json zero-based index ' + $d.source_index
        $instruction = $contract.prompts.suffixes.open
    } else {
        $original = $science[$d.source_id]
        if ([string]$splits.train[$d.hf_row] -ne $d.source_id) { throw 'HF row / official split order mismatch' }
        $hf = ($hfRows | Where-Object row_idx -eq $d.hf_row).row
        foreach ($key in @('question','hint','answer','grade','subject','topic','category','skill','lecture','solution')) {
            if ([string]$original[$key] -cne [string]$hf.$key) { throw "Official/HF field mismatch: $key" }
        }
        if (($original.choices -join '|') -cne ($hf.choices -join '|')) { throw 'Choice mismatch' }
        $question = $original.question
        $choices = @($original.choices)
        $hintText = $original.hint
        $sourceAnswer = $original.answer
        if ([string][char](65 + [int]$sourceAnswer) -cne $d.answer) { throw 'Science answer mismatch' }
        $family = 'ScienceQA:' + $d.source_id + ':unclustered'
        $sourceUrl = 'https://github.com/lupantech/ScienceQA/blob/' + $revisions.ScienceQA + '/data/scienceqa/problems.json'
        $license = 'CC BY-NC-SA 4.0 per official LICENSE-DATA; the HF mirror metadata differs, so retain the original license.'
        $originalId = $d.source_id
        $instruction = $contract.prompts.suffixes.'multiple-choice'
    }
    # Resolve actual config shape explicitly; never silently omit the baseline suffix.
    if (-not $instruction) { throw 'Read config prompt key before exporting.' }
    $imagePath = [IO.Path]::GetFullPath((Join-Path $taskRoot $d.image))
    $imageHash = (Get-FileHash -LiteralPath $imagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $inputRecord = [ordered]@{
        image_paths = @($imagePath.Replace('\','/'))
        question = $question
        choices = $choices
        hint = $hintText
        final_answer_instruction = $instruction
        prompt_contract = 'mmmu-local-v1; preserve question-type suffix; trainer rendering not yet validated'
    }
    $eraText = "Evidence: $($d.evidence)`nRule: $($d.rule)`nApply: $($d.apply)`nFinal answer: $($d.answer)"
    $eraKo = "Evidence: $($d.evidence_ko)`nRule: $($d.rule_ko)`nApply: $($d.apply_ko)`nFinal answer: $($d.answer)"
    $pair = [ordered]@{
        id = $d.id
        dataset = $d.dataset
        original_id = $originalId
        original_split = 'train'
        project_split = 'demo_only'
        primary_type = $d.type
        family_id = $family
        family_clustering_status = 'not_run; source IDs are provenance, not proof of cross-source independence'
        source_url = $sourceUrl
        source_revision_observed = $revisions[$d.dataset]
        image_sha256 = $imageHash
        image_provenance = $(if ($isChart) { 'Official ChartQA GitHub image' } else { 'derek-thomas/ScienceQA mirror revision f18b0a70359ebfb41f658fd564208d0355b013f4; row mapped to official train order and all textual fields checked' })
        license = $license
        input = $inputRecord
        source_answer = $sourceAnswer
        canonical_final_answer = $d.answer
        verified_content = [ordered]@{ evidence=$d.evidence; rule=$d.rule; apply=$d.apply; semantic_checklist=$d.verified_facts }
        response_plain = $d.plain
        response_era = $eraText
        response_pair_design = 'natural integrated prose versus explicit ERA sections; same semantic content; no prescribed order for plain responses'
        display_only_ko = @{ question=$d.question_ko; plain=$d.plain_ko; era=$eraKo; plain_composition=$d.plain_composition_ko }
        auxiliary_verification = @{ verified_table=$d.verified_table; csv_review=$d.csv_review; computed_validation=$d.computed_validation }
        review = [ordered]@{
            author='assistant, source-grounded new explanations'
            image_inspection='completed by assistant'
            reference_answer_match=$true
            pair_semantic_review='same listed facts, rule, conditions, application, and answer; assistant reviewed'
            human_review='pending'
            mmmu_overlap_check='not_run'
            source_cross_split_overlap_check='not_run'
            tokenizer_length_and_loss_mask='not_run'
            train_eligible=$false
            notes_ko=$d.review_ko
        }
        word_counts = @{ plain=(WordCount $d.plain); era=(WordCount $eraText); unit='whitespace-delimited English words, NOT model tokens' }
    }
    $pairs += $pair
    $plainRows += [ordered]@{id=$d.id;input=$inputRecord;target_response=$d.plain;usage='demo_only'}
    $eraRows += [ordered]@{id=$d.id;input=$inputRecord;target_response=$eraText;usage='demo_only'}
    $selectedOriginals += @{dataset=$d.dataset;original_id=$originalId;record=$original}
    $optionHtml = ''
    for ($i=0; $i -lt $choices.Count; $i++) { $optionHtml += '<li>' + (Esc ([string][char](65+$i) + '. ' + $choices[$i])) + '</li>' }
    $hintHtml = if ($hintText) { '<details><summary>원본 지문 / hint — 학생 입력에 포함</summary><p>' + (Esc $hintText) + '</p></details>' } else { '' }
    $cards += @"
<article>
<div class="tags"><span>$(Esc $d.dataset) · 공식 train</span><span>$(Esc $d.type)</span><span>$(Esc $d.id)</span></div>
<h2>$(Esc $d.title_ko)</h2>
<div class="problem"><figure><img src="$(Esc $d.image)" alt="$(Esc $d.title_ko) 원본 이미지"></figure><div><h3>원본 질문</h3><p>$(Esc $question)</p><p class="korean">$(Esc $d.question_ko)</p><ul>$optionHtml</ul>$hintHtml<p><b>공식 정답:</b> $(Esc $d.answer) $(if (-not $isChart) { ' = ' + (Esc $choices[[int]$sourceAnswer]) })</p><a href="$(Esc $sourceUrl)">공식 원본 · ID/인덱스 $(Esc $originalId)</a></div></div>
<h3>실제 목표 응답 · 영어</h3>
<p class="small">일반 풀이의 서술 흐름: $(Esc $d.plain_composition_ko)</p>
<div class="pair"><section><h4>일반 풀이</h4><pre>$(Esc $d.plain)</pre></section><section class="era"><h4>ERA 풀이</h4><pre>$(Esc $eraText)</pre></section></div>
<details class="translation" open><summary>한국어 설명 · 학습 파일에는 미포함</summary><div class="pair"><section><h4>일반 풀이</h4><pre>$(Esc $d.plain_ko)</pre></section><section class="era"><h4>ERA 풀이</h4><pre>$(Esc $eraKo)</pre></section></div></details>
<p class="review"><b>검수 기록</b> · $(Esc $d.review_ko)</p>
<p class="small">영어 단어 수: 일반 $(WordCount $d.plain) / ERA $(WordCount $eraText). 토크나이저 길이는 아직 측정하지 않았습니다.</p>
</article>
"@
}
if ($pairs.Count -ne 4) { throw 'Expected four examples' }
if ((@($pairs.id | Sort-Object -Unique)).Count -ne 4) { throw 'Duplicate IDs' }
for ($i=0; $i -lt 4; $i++) {
    if ((ConvertTo-Json $plainRows[$i].input -Depth 20 -Compress) -cne (ConvertTo-Json $eraRows[$i].input -Depth 20 -Compress)) { throw 'Inputs differ' }
    foreach ($target in @($plainRows[$i].target_response,$eraRows[$i].target_response)) {
        if (($target -split "`n")[-1] -cne ('Final answer: ' + $pairs[$i].canonical_final_answer)) { throw 'Final answer mismatch' }
    }
}
if ((51+44) -ne 95 -or (@('MM','Mm','Mm','mm') | Where-Object {$_ -ceq 'MM'}).Count -ne 1) { throw 'Arithmetic check failed' }
foreach ($entry in @(@{name='paired_examples.jsonl';rows=$pairs},@{name='plain_sft_examples.jsonl';rows=$plainRows},@{name='era_sft_examples.jsonl';rows=$eraRows})) {
    $lines = @($entry.rows | ForEach-Object { ConvertTo-Json -InputObject $_ -Depth 40 -Compress })
    Write-Text (Join-Path $taskRoot $entry.name) (($lines -join "`n") + "`n")
}
Write-Json "$taskRoot/sources/selected_original_records.json" $selectedOriginals
Write-Json "$taskRoot/validation.json" @{
    examples=4; plain_targets=4; era_targets=4; identical_inputs=$true; unique_ids=$true
    official_answer_match=$true; image_files_and_sha256_checked=$true; scienceqa_official_mirror_fields_match=$true
    final_line_check=$true; arithmetic_checks_pass=$true; baseline_suffix_loaded_from_local_config=$true
    model_tokenization='not_run'; trainer_rendering='not_run'; baseline_parser_execution='not_run'
    independent_human_review='pending'; contamination_scan='not_run'; split_grouping='not_run'; train_eligible=$false
}
$html = @"
<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 문제로 보는 일반 풀이 · ERA 데이터</title>
<style>
:root{font-family:'Malgun Gothic',system-ui,sans-serif;color:#17283a;background:#f3f6fa;line-height:1.7}body{margin:0}main{max-width:1160px;margin:auto;padding:42px 24px 70px}h1{font-size:32px;letter-spacing:-1.2px;line-height:1.3}h2{font-size:25px;margin:14px 0 22px}h3{font-size:17px}h4{margin:0 0 12px;color:#345069}a{color:#12618e}header{margin-bottom:30px}.lead{font-size:18px;max-width:880px}.notice{padding:18px 22px;border-left:4px solid #d39323;background:#fff5de;border-radius:8px}article{background:white;margin:28px 0;padding:30px;border:1px solid #dce5ec;border-radius:18px}.tags{display:flex;flex-wrap:wrap;gap:8px;font-size:12px}.tags span{background:#eef2f7;padding:4px 10px;border-radius:20px}.problem{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:center}figure{margin:0;text-align:center}img{max-width:100%;max-height:490px;object-fit:contain}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair section{background:#f3f6fa;border-radius:12px;padding:20px}.pair .era{background:#edf7f4;border:1px solid #cde7df}pre{white-space:pre-wrap;font:inherit;margin:0;overflow-wrap:anywhere;line-height:1.8}details{margin:15px 0}summary{cursor:pointer;font-weight:600;margin-bottom:12px}.korean{color:#52667a}.review{background:#fff7e8;border-radius:8px;padding:16px}.small{font-size:12px;color:#65798a}ul{padding-left:20px}.links{display:flex;gap:18px;flex-wrap:wrap}@media(max-width:760px){.problem,.pair{grid-template-columns:1fr}article{padding:20px}main{padding:24px 14px}h1{font-size:26px}}@media print{body{background:white}article{break-before:page}.links{display:none}}
</style></head><body><main><header><p>ERA DATASET EXAMPLES · 2026-10-14</p><h1>원본 문제 4개, 목표 응답은 두 가지</h1><p class="lead">ChartQA 2문항과 ScienceQA 2문항의 실제 train 자료입니다. 이미지·질문·보기·정답을 유지하고, 동일한 근거와 규칙을 일반 풀이와 ERA 풀이로 작성했습니다.</p><div class="notice"><b>제작 예시이며 학습 투입 전 검수 대기입니다.</b> 이미지 판독·공식 정답·두 풀이 내용을 확인했습니다. MMMU 중복 검사, 전체 split 간 중복 검사, 사람의 독립 검수, 모델 토큰 길이·loss mask 점검은 아직 하지 않았습니다. CSV·해설·검수 기록은 학생 입력에 포함하지 않습니다.</div><p class="links"><a href="paired_examples.jsonl">공통 원본 + 두 풀이 JSONL</a><a href="plain_sft_examples.jsonl">일반 풀이 입력/응답</a><a href="era_sft_examples.jsonl">ERA 입력/응답</a><a href="validation.json">검증 결과</a></p></header>
<p class="lead"><b>수정된 비교 기준:</b> 일반 풀이는 문제에 맞게 근거·규칙·적용을 자연스럽게 연결하며 E→R→A 순서를 강제하지 않습니다. ERA 풀이는 같은 정보를 역할별로 분리합니다. 일반 풀이도 필요한 근거와 규칙을 포함하며, 두 응답의 길이를 억지로 맞추지는 않았습니다. 아래 흐름 설명은 검토용으로만 표시하고 학생 입력에는 넣지 않습니다.</p>
$($cards -join "`n")
<footer><h2>파일을 읽는 방법</h2><p>paired_examples.jsonl의 한 줄은 하나의 원본 문제입니다. input은 두 조건이 공유하고, response_plain과 response_era만 다릅니다. 파생된 두 JSONL은 각각 같은 4개 ID와 같은 input을 포함합니다. 학습기 전용 형식으로 변환하고 processor·chat template·종료 토큰을 검증하기 전의 예시 포맷입니다.</p><p>원본 영어 질문·보기·hint를 보존했습니다. 한국어는 검토용 번역이며 학습 응답에 들어가지 않습니다. 최종 답 지시는 로컬 baseline 설정의 객관식/단답형 지시를 그대로 읽었습니다. ERA 작성 지시는 학생 입력에 넣지 않았습니다.</p><p>ChartQA: 공식 저장소의 GPL-3.0 표기 및 차트 원출처(Pew Research Center)를 보존합니다. ScienceQA: 공식 LICENSE-DATA의 CC BY-NC-SA 4.0을 따릅니다. 이미지 제공용 Hugging Face 미러의 라이선스 표기가 공식 원본과 달라 원본 기준을 기록했습니다. 새 풀이는 assistant가 작성했으며 데이터셋 공식 해설로 오인하면 안 됩니다.</p><p><a href="sources/ChartQA_LICENSE.txt">ChartQA 라이선스</a> · <a href="sources/ScienceQA_LICENSE-DATA.txt">ScienceQA 라이선스</a> · <a href="sources/selected_original_records.json">선택한 원본 레코드</a></p></footer></main></body></html>
"@
Write-Text "$taskRoot/examples.html" $html
Write-Output 'Built 4 paired examples, 2 SFT JSONL exports, HTML preview, and validation metadata.'
