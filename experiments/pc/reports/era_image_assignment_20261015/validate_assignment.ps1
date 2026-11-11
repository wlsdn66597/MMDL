param([string]$Root=$PSScriptRoot, [string]$SourceRoot=(Join-Path $PSScriptRoot '../era_dataset_examples_20261014'))
$ErrorActionPreference='Stop'
function Read-Lines($name) { @(Get-Content -LiteralPath (Join-Path $Root $name) | ForEach-Object { ConvertFrom-Json -InputObject $_ -AsHashtable }) }
$rows=Read-Lines 'source_manifest.jsonl'
$groups=Read-Lines 'image_groups.jsonl'
$originalChart=Get-Content -LiteralPath (Join-Path $SourceRoot 'chartqa_train_human.json') -Raw | ConvertFrom-Json -AsHashtable
$originalScience=Get-Content -LiteralPath (Join-Path $SourceRoot 'scienceqa_problems.json') -Raw | ConvertFrom-Json -AsHashtable
$originalSplits=Get-Content -LiteralPath (Join-Path $SourceRoot 'scienceqa_splits.json') -Raw | ConvertFrom-Json -AsHashtable
$trainIds=[Collections.Generic.HashSet[string]]::new([string[]]$originalSplits.train)
$byId=@{}; $byGroup=@{}; $seenExports=[Collections.Generic.HashSet[string]]::new()
foreach($g in $groups) { if($byGroup.ContainsKey($g.image_group_id)){throw 'Duplicate group ID'}; $byGroup[$g.image_group_id]=$g }
foreach($r in $rows) {
    if($byId.ContainsKey($r.sample_id)){throw 'Duplicate sample ID'}; $byId[$r.sample_id]=$r
    $g=$byGroup[$r.image_group_id]
    if(-not $g -or $g.owner -ne $r.owner -or $g.image_relative_path -cne $r.image_relative_path -or $g.image_group_no -ne $r.image_group_no -or $r.sample_id -notin $g.sample_ids){throw 'Group mapping mismatch'}
    $wanted=if($r.source_dataset -eq 'ChartQA'){if($r.image_group_no%2){'A'}else{'B'}}else{if($r.image_group_no%2){'C'}else{'D'}}
    if($r.owner -ne $wanted -or $r.source_split -ne 'train' -or $r.train_eligible -ne $false){throw 'Assignment/state mismatch'}
    if($r.source_dataset -eq 'ChartQA'){
        $s=$originalChart[$r.source_index]
        if($r.question -cne $s.query -or $r.original_answer -cne $s.label -or $r.image_relative_path -cne ('train/png/'+$s.imgname)){throw 'Chart source mismatch'}
    }else{
        if(-not $trainIds.Contains($r.source_id)){throw 'Science non-train ID'}
        $s=$originalScience[$r.source_id]
        if(-not $s.image -or $r.image_relative_path -cne "train/$($r.source_id)/$($s.image)" -or $r.question -cne $s.question -or $r.original_answer -ne $s.answer){throw 'Science source mismatch'}
        if((ConvertTo-Json -InputObject @($r.original_options) -Compress) -cne (ConvertTo-Json -InputObject @($s.choices) -Compress)){throw 'Science choice mismatch'}
    }
}
foreach($owner in 'A','B','C','D'){
    foreach($r in (Read-Lines "${owner}_candidates.jsonl")){
        if($r.owner -ne $owner -or -not $byId.ContainsKey($r.sample_id) -or -not $seenExports.Add($r.sample_id)){throw 'Owner export overlap/mismatch'}
        if((ConvertTo-Json -InputObject $r -Depth 30 -Compress) -cne (ConvertTo-Json -InputObject $byId[$r.sample_id] -Depth 30 -Compress)){throw 'Owner export differs from master'}
    }
}
if($seenExports.Count -ne $rows.Count){throw 'Incomplete owner export union'}
foreach($g in $groups){if($g.sample_ids.Count -ne $g.question_count){throw 'Group member count mismatch'}}
$scienceExpected=@($originalSplits.train | Where-Object {$originalScience[[string]$_].image}).Count
if(@($rows | Where-Object source_dataset -eq 'ScienceQA').Count -ne $scienceExpected -or @($rows | Where-Object source_dataset -eq 'ChartQA').Count -ne $originalChart.Count){throw 'Incomplete original candidate coverage'}
$checkedQuestions=$rows.Count
$lightMap=@{}
foreach($row in $rows){$lightMap[$row.sample_id]=@($row.owner,$row.image_group_no)}
$rows=$null; $row=$null; $r=$null; $s=$null
$byId.Clear(); $byId=$null
$originalScience=$null; $originalChart=$null; $originalSplits=$null
[GC]::Collect(); [GC]::WaitForPendingFinalizers(); [GC]::Collect()
$html=Get-Content -LiteralPath (Join-Path $Root 'index.html') -Raw
$embedded=[regex]::Match($html,'<script id="data" type="application/json">(.*?)</script>',[Text.RegularExpressions.RegexOptions]::Singleline).Groups[1].Value | ConvertFrom-Json -AsHashtable
if($embedded.Count -ne $checkedQuestions){throw 'HTML data count mismatch'}
foreach($v in $embedded){if($v.owner -ne $lightMap[$v.id][0] -or $v.group -ne $lightMap[$v.id][1]){throw 'HTML owner mapping mismatch'}}
$result=[ordered]@{passed=$true;checked_questions=$checkedQuestions;checked_groups=$groups.Count;owner_exports_disjoint_and_complete=$true;original_source_fields_match=$true;group_parity_and_membership_match=$true;html_embedded_rows_match=$true;image_content_and_mmmu_checks='not_run';browser_rendering='not_verified_browser_attachment_timed_out'}
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root 'independent_validation.json') -Encoding utf8
$result | ConvertTo-Json
