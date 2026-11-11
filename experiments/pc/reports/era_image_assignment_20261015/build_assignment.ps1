param(
    [string]$SourceRoot = (Join-Path $PSScriptRoot '../era_dataset_examples_20261014'),
    [string]$OutputRoot = $PSScriptRoot
)
$ErrorActionPreference = 'Stop'
$version = 'era-image-assignment-v1-20261015'
$utf8 = [Text.UTF8Encoding]::new($false)
function Write-Text($path, $value) { [IO.File]::WriteAllText($path, $value, $utf8) }
function Write-Json($path, $value) { Write-Text $path (ConvertTo-Json -InputObject $value -Depth 30) }
function Write-Jsonl($path, $rows) {
    $writer = [IO.StreamWriter]::new($path, $false, $utf8)
    try { foreach ($row in $rows) { $writer.WriteLine((ConvertTo-Json -InputObject $row -Depth 30 -Compress)) } }
    finally { $writer.Dispose() }
}
function Ordinal-Sort($values) {
    [string[]]$items = @($values)
    [Array]::Sort($items, [StringComparer]::Ordinal)
    return ,$items
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$chartFile = Join-Path $SourceRoot 'chartqa_train_human.json'
$scienceFile = Join-Path $SourceRoot 'scienceqa_problems.json'
$splitFile = Join-Path $SourceRoot 'scienceqa_splits.json'
$charts = Get-Content -LiteralPath $chartFile -Raw | ConvertFrom-Json -AsHashtable
$science = Get-Content -LiteralPath $scienceFile -Raw | ConvertFrom-Json -AsHashtable
$splits = Get-Content -LiteralPath $splitFile -Raw | ConvertFrom-Json -AsHashtable
$chartRev = (Get-Content -LiteralPath (Join-Path $SourceRoot 'sources/ChartQA_revision.json') -Raw | ConvertFrom-Json).sha
$scienceRev = (Get-Content -LiteralPath (Join-Path $SourceRoot 'sources/ScienceQA_revision.json') -Raw | ConvertFrom-Json).sha
$chartUrl = "https://raw.githubusercontent.com/vis-nlp/ChartQA/$chartRev/ChartQA%20Dataset/train/train_human.json"
$scienceUrl = "https://raw.githubusercontent.com/lupantech/ScienceQA/$scienceRev/data/scienceqa/problems.json"
$all = [Collections.Generic.List[object]]::new()
$excluded = [Collections.Generic.List[object]]::new()
for ($i=0; $i -lt $charts.Count; $i++) {
    $c = $charts[$i]
    if (-not $c.imgname) { throw "Chart image reference missing: $i" }
    $path = 'train/png/' + $c.imgname
    $binary = ([string]$c.label).Trim().ToLowerInvariant() -in @('yes','no','true','false')
    $all.Add([ordered]@{
        sample_id=('chartqa_train_human_{0:D6}' -f $i); source_dataset='ChartQA'; source_split='train'; source_subset='human'
        source_id="train_human.json:$i"; source_index=$i; source_revision=$chartRev; source_url=$chartUrl
        source_license='GPL-3.0 declared by repository; original chart rights require review'
        image_relative_path=$path; image_reference_key="ChartQA:$path"
        image_url="https://raw.githubusercontent.com/vis-nlp/ChartQA/$chartRev/ChartQA%20Dataset/$path"
        table_url="https://raw.githubusercontent.com/vis-nlp/ChartQA/$chartRev/ChartQA%20Dataset/train/tables/$([IO.Path]::GetFileNameWithoutExtension($c.imgname)).csv"
        image_sha256=$null; image_hash_status='not_checked'; image_grouping_basis='original_relative_path'
        question=$c.query; original_answer=$c.label; original_options=@(); hint=''
        subject=$null; topic=$null; skill=$null; lecture=''; solution=''
        candidate_flag=$(if ($binary) {'exclude_binary_answer_for_4way_pilot'} else {'needs_visual_and_4way_review'})
        selection_status=$(if ($binary) {'excluded_by_current_input_policy'} else {'unselected'})
        quality_status='unreviewed'; overlap_status='not_run'; visual_dependency_status='not_reviewed'; train_eligible=$false
    })
}
$trainIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($idValue in $splits.train) {
    $id = [string]$idValue
    if (-not $trainIds.Add($id)) { throw "Duplicate official train id: $id" }
    $s = $science[$id]
    if (-not $s -or $s.split -ne 'train') { throw "Science split mismatch: $id" }
    if (-not $s.image) { $excluded.Add([ordered]@{source_dataset='ScienceQA';source_id=$id;reason='no_image'}); continue }
    if ($s.answer -lt 0 -or $s.answer -ge $s.choices.Count) { throw "Invalid answer index: $id" }
    $path = "train/$id/$($s.image)"
    $all.Add([ordered]@{
        sample_id="scienceqa_train_$id"; source_dataset='ScienceQA'; source_split='train'; source_subset='image_present'
        source_id=$id; source_index=$null; source_revision=$scienceRev; source_url=$scienceUrl
        source_license='CC BY-NC-SA-4.0 per official LICENSE-DATA'
        image_relative_path=$path; image_reference_key="ScienceQA:$path"; image_url=$null; table_url=$null
        image_sha256=$null; image_hash_status='not_checked'; image_grouping_basis='original_relative_path'
        question=$s.question; original_answer=$s.answer; original_options=@($s.choices); hint=$s.hint
        answer_letter=[string][char](65+[int]$s.answer); answer_value=$s.choices[$s.answer]
        subject=$s.subject; topic=$s.topic; skill=$s.skill; lecture=$s.lecture; solution=$s.solution
        candidate_flag='needs_visual_dependency_and_quality_review'; selection_status='unselected'
        quality_status='unreviewed'; overlap_status='not_run'; visual_dependency_status='not_reviewed'; train_eligible=$false
    })
}
$groups = [Collections.Generic.List[object]]::new()
$groupMembers = @{}
foreach ($r in $all) {
    if (-not $groupMembers.ContainsKey($r.image_reference_key)) { $groupMembers[$r.image_reference_key] = [Collections.Generic.List[object]]::new() }
    $groupMembers[$r.image_reference_key].Add($r)
}
foreach ($dataset in @('ChartQA','ScienceQA')) {
    $keys = Ordinal-Sort @($groupMembers.Keys | Where-Object { $_.StartsWith($dataset + ':', [StringComparison]::Ordinal) })
    $number = 0
    foreach ($key in $keys) {
        $number++
        $members = $groupMembers[$key]
        $owner = if ($dataset -eq 'ChartQA') { if ($number % 2) {'A'} else {'B'} } else { if ($number % 2) {'C'} else {'D'} }
        $groupId = '{0}_g{1:D6}' -f $dataset.ToLowerInvariant(), $number
        foreach ($r in $members) {
            $r['image_group_id']=$groupId; $r['image_group_no']=$number; $r['owner']=$owner
            $r['reviewer']=@{A='B';B='A';C='D';D='C'}[$owner]; $r['manifest_version']=$version
        }
        $first = $members[0]
        $groups.Add([ordered]@{
            source_dataset=$dataset; image_group_id=$groupId; image_group_no=$number; owner=$owner
            image_relative_path=$first.image_relative_path; image_reference_key=$key; image_url=$first.image_url
            image_sha256=$null; grouping_basis='original_relative_path'; content_duplicate_scan='not_run'
            question_count=$members.Count; selectable_candidate_count=@($members | Where-Object selection_status -eq 'unselected').Count
            sample_ids=@($members | ForEach-Object sample_id); source_ids=@($members | ForEach-Object source_id); manifest_version=$version
        })
    }
}
$orderedRows = @($all | Sort-Object source_dataset,image_group_no,sample_id)
# Existing assignments are immutable: never silently renumber a released manifest.
$groupPath = Join-Path $OutputRoot 'image_groups.jsonl'
if (Test-Path -LiteralPath $groupPath) {
    $oldMap = @{}
    Get-Content -LiteralPath $groupPath | ForEach-Object { $g=$_ | ConvertFrom-Json; $oldMap[$g.image_reference_key]=$g }
    if ($oldMap.Count -ne $groups.Count) { throw 'Frozen manifest changed: create a reviewed new version; do not regenerate numbering.' }
    foreach ($g in $groups) {
        $old = $oldMap[$g.image_reference_key]
        if (-not $old -or $old.image_group_no -ne $g.image_group_no -or $old.owner -ne $g.owner) { throw 'Frozen assignment conflict.' }
    }
}
$uniqueIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($r in $orderedRows) {
    if (-not $uniqueIds.Add($r.sample_id)) { throw "Duplicate sample: $($r.sample_id)" }
    $want = if ($r.source_dataset -eq 'ChartQA') { if ($r.image_group_no % 2) {'A'} else {'B'} } else { if ($r.image_group_no % 2) {'C'} else {'D'} }
    if ($r.owner -ne $want -or $r.source_split -ne 'train') { throw 'Owner/split validation failure.' }
}
$stats = [ordered]@{}
foreach ($owner in @('A','B','C','D')) {
    $rows = @($orderedRows | Where-Object owner -eq $owner)
    $stats[$owner] = [ordered]@{
        dataset=$rows[0].source_dataset; groups=@($groups | Where-Object owner -eq $owner).Count
        all_questions=$rows.Count; selectable_candidates=@($rows | Where-Object selection_status -eq 'unselected').Count
        excluded_binary_questions=@($rows | Where-Object selection_status -ne 'unselected').Count; target_accepted_questions=50
    }
    if ($stats[$owner].selectable_candidates -lt 50) { throw "Insufficient candidates: $owner" }
    Write-Jsonl (Join-Path $OutputRoot "${owner}_candidates.jsonl") $rows
}
Write-Jsonl $groupPath $groups
Write-Jsonl (Join-Path $OutputRoot 'source_manifest.jsonl') $orderedRows
Write-Jsonl (Join-Path $OutputRoot 'scienceqa_no_image_exclusions.jsonl') $excluded
$provenance = foreach ($entry in @(@{path=$chartFile;url=$chartUrl;rev=$chartRev},@{path=$scienceFile;url=$scienceUrl;rev=$scienceRev},@{path=$splitFile;url="https://raw.githubusercontent.com/lupantech/ScienceQA/$scienceRev/data/scienceqa/pid_splits.json";rev=$scienceRev})) {
    [ordered]@{local_file=[IO.Path]::GetFileName($entry.path);sha256=(Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant();source_url=$entry.url;declared_revision=$entry.rev;provenance='local source snapshot collected for 2026-10-14 examples; not re-downloaded in this build'}
}
$report = [ordered]@{
    manifest_version=$version; created_date='2026-10-15'; numbering='per dataset; ordinal lexicographic sort of full original relative image path; 1-based'
    scope='ChartQA human train only; ScienceQA official train with image metadata'; source_files=@($provenance)
    chartqa_human_questions=$charts.Count; scienceqa_train_total=$trainIds.Count; scienceqa_train_without_image=$excluded.Count
    total_assigned_questions=$orderedRows.Count; total_image_reference_groups=$groups.Count; owners=$stats
    checks=[ordered]@{unique_sample_ids=$true;official_train_membership=$true;one_owner_per_original_image_reference=$true;parity_assignment=$true;all_candidates_exported_once=$true;minimum_50_candidates_each=$true;byte_or_pixel_image_dedup='not_run';near_image_dedup='not_run';mmmu_dev_test_overlap='not_run';visual_dependency='not_reviewed';question_quality='not_reviewed';train_eligible=$false}
}
Write-Json (Join-Path $OutputRoot 'validation_report.json') $report
# Compact offline viewer: text is inserted only with textContent.
$viewRows = @($orderedRows | ForEach-Object { [ordered]@{owner=$_.owner;dataset=$_.source_dataset;group=$_.image_group_no;group_id=$_.image_group_id;id=$_.sample_id;source_id=$_.source_id;path=$_.image_relative_path;url=$_.image_url;source_url=$_.source_url;question=$_.question;options=@($_.original_options);hint=$_.hint;answer=$_.original_answer;skill=$_.skill;flag=$_.candidate_flag;status=$_.selection_status} })
$viewJson = (ConvertTo-Json -InputObject $viewRows -Depth 10 -Compress).Replace('<','\u003c').Replace('>','\u003e').Replace('&','\u0026')
$html = @'
<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ERA 파일럿 · ABCD 공용 후보 목록</title><style>
body{font:15px/1.6 system-ui,sans-serif;margin:32px auto;max-width:1400px;padding:0 20px;color:#183247;background:#f6f8fa}h1{font-size:27px}section,.panel{background:white;border:1px solid #dce3e8;border-radius:10px;padding:18px;margin:16px 0}.muted{color:#596c7a}.notice{background:#fff6df;padding:14px;border-radius:8px}select,input,button{font:inherit;padding:7px;margin:4px;border:1px solid #bccbd6;border-radius:5px}input{width:310px}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3e8;vertical-align:top}th{background:#eaf0f4}td:first-child{white-space:nowrap}code{overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer}a{color:#075d99}pre{white-space:pre-wrap}#count{font-weight:600}.excluded{color:#91542a}nav{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
</style><h1>ERA 파일럿 · ABCD 공용 후보 목록</h1>
<p>버전 <code>era-image-assignment-v1-20261015</code> · A/B: ChartQA 홀수/짝수 · C/D: ScienceQA 홀수/짝수</p>
<p class="notice">이 목록은 200문항 확정본이 아닙니다. 담당별 후보에서 최종 50문항씩 선택하세요. 원본 이미지 경로로만 묶었으며 이미지 내용·유사 중복·MMMU 중복·풀이 품질은 아직 검증하지 않았습니다.</p>
<section><strong>시작 방법</strong><ol><li>본인 담당을 선택하고 후보를 살펴봅니다.</li><li>동일 이미지 그룹당 1문항 우선, 최대 2문항을 선택합니다. 처음 50개를 기계적으로 고르지 않습니다.</li><li>선택한 sample_id를 기록하고 원본 이미지를 확인한 뒤 공통 내용·일반·ERA 풀이를 작성합니다.</li></ol>
<p>ChartQA의 예/아니오·참/거짓 답 문항은 4지선다 제작 대상에서 제외 표시했습니다. ScienceQA는 이미지 메타데이터가 있는 문항만 포함했으며 실제 이미지 필요성은 직접 확인해야 합니다.</p>
<p>ChartQA의 이미지 링크는 공식 저장소의 고정 revision을 가리킵니다. ScienceQA는 공식 자료에서 이미지를 내려받아 <code>train/문제ID/image.png</code> 경로로 찾으세요. 이미지 원본은 이 묶음에 포함하지 않았습니다.</p></section>
<nav class="panel"><label>담당 <select id="owner"><option value="">전체</option><option>A</option><option>B</option><option>C</option><option>D</option></select></label><label><input type="checkbox" id="eligible" checked style="width:auto">명시적 제외 후보 숨기기</label><input id="search" placeholder="질문·문항 ID·이미지 경로·skill 검색"><span id="count"></span></nav>
<nav><button id="prev">이전</button><span id="page"></span><button id="next">다음</button><span class="muted">페이지당 40문항</span></nav>
<table><thead><tr><th>담당 / 그룹</th><th>문항 / 원본 이미지</th><th>질문 / 보기 / 참고</th></tr></thead><tbody id="body"></tbody></table>
<p class="muted">그룹 번호는 출처별로 1부터 붙인 고정 번호입니다. 파일명의 숫자·질문 번호와 다릅니다. 삭제 후 번호를 당기거나 각자 다시 번호를 붙이지 마세요.</p>
<script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let page=0,filtered=[];
function render(reset=false){if(reset)page=0;const q=$('search').value.trim().toLowerCase();filtered=rows.filter(r=>(!$('owner').value||r.owner===$('owner').value)&&(!$('eligible').checked||r.status==='unselected')&&(!q||[r.question,r.id,r.path,r.skill,r.source_id,r.group_id].join(' ').toLowerCase().includes(q)));const pages=Math.max(1,Math.ceil(filtered.length/40));page=Math.max(0,Math.min(page,pages-1));$('count').textContent=filtered.length.toLocaleString()+'문항';$('page').textContent=(page+1)+' / '+pages;$('prev').disabled=page===0;$('next').disabled=page===pages-1;$('body').replaceChildren();for(const r of filtered.slice(page*40,page*40+40)){const tr=document.createElement('tr');const a=document.createElement('td');a.textContent=r.owner+' / '+r.group;const b=document.createElement('td');const id=document.createElement('code');id.textContent=r.id;b.append(id,document.createElement('br'));const path=document.createElement('code');path.textContent=r.path;b.append(path);if(r.url){const link=document.createElement('a');link.href=r.url;link.target='_blank';link.rel='noopener';link.textContent='원본 이미지 열기';b.append(document.createElement('br'),link)}const c=document.createElement('td');const p=document.createElement('div');p.textContent=r.question;c.append(p);if(r.options.length){const ol=document.createElement('ol');ol.type='A';for(const opt of r.options){const li=document.createElement('li');li.textContent=opt;ol.append(li)}c.append(ol)}const details=document.createElement('details'),sum=document.createElement('summary'),pre=document.createElement('pre');sum.textContent='원본 답·hint·선별 메모';pre.textContent='원본 답: '+(r.dataset==='ScienceQA'?String.fromCharCode(65+r.answer)+' — '+r.options[r.answer]:r.answer)+'\nSkill: '+(r.skill||'미지정')+'\n선별 표시: '+r.flag+'\nHint: '+r.hint;details.append(sum,pre);c.append(details);if(r.status!=='unselected')c.className='excluded';tr.append(a,b,c);$('body').append(tr)}}
for(const id of ['owner','eligible'])$(id).addEventListener('change',()=>render(true));$('search').addEventListener('input',()=>render(true));$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();
</script></html>
'@
Write-Text (Join-Path $OutputRoot 'index.html') $html.Replace('__DATA__',$viewJson)
$table = foreach($owner in @('A','B','C','D')) { $s=$stats[$owner]; "| $owner | $($s.dataset) | $($s.groups) | $($s.all_questions) | $($s.selectable_candidates) | 50 |" }
$readme = @"
# ABCD 공용 이미지 그룹·후보 목록

2026-10-15 · 버전: $version

## 작업 시작

1. index.html을 브라우저에서 열고 본인의 담당 A/B/C/D를 선택한다.
2. 배정된 범위에서 다양한 문제를 골라 sample_id를 기록한다. 최종 목표는 각자 검증된 50문항이다.
3. 같은 그룹은 1문항 우선, 최대 2문항까지 선택한다. 각자 번호를 다시 붙이지 않는다.
4. 원본 이미지 확인, 공통 내용표, 일반·ERA 풀이 작성 후 취합·교차 검수한다.

| 담당 | 출처 | 이미지 경로 그룹 | 전체 배정 문항 | 명시적 제외 외 후보 | 최종 목표 |
|---|---|---:|---:|---:|---:|
$($table -join "`n")

명시적 제외 외 후보도 아직 품질 검수 전이다. 최종 학습 200문항 또는 사람 검수 합격 목록이 아니다.

## 번호와 중복 방지

- ChartQA: 홀수 A / 짝수 B. ScienceQA: 홀수 C / 짝수 D.
- 출처별 원본 전체 이미지 상대 경로를 Ordinal 문자열 오름차순으로 정렬하고 1부터 번호를 부여했다. 숫자 자연 정렬이 아니므로 파일명 100은 20보다 앞설 수 있다.
- 그룹 ID는 출처를 포함한다. 같은 상대 경로의 모든 질문은 같은 그룹·담당자다.
- ScienceQA의 image.png는 여러 문제에 반복되는 파일명이다. train/문제ID/image.png 전체 경로를 사용했으며 파일명만 같다고 같은 이미지로 묶지 않았다.
- 기존 번호는 고정한다. 탈락으로 인한 재번호 부여는 금지하며 신규 그룹은 마지막 번호 이후로 추가한다. 변경 시 새 목록 버전과 이력을 공유한다.
- 이미지 파일을 전체 확보하지 않았으므로 SHA256·픽셀·유사 이미지 검사는 미실행이다. 다른 경로에 있는 같은 이미지/변형 이미지는 서로 다른 담당에 배정될 수 있으며, 취합 후 통합 검사로 해결해야 한다.

## 후보 범위

- ChartQA: 로컬에 확보된 공식 train_human.json 전체. machine-generated train_augmented.json은 이번 목록에 포함하지 않았다.
- ChartQA 답이 Yes/No/True/False인 문항은 원본 질문을 유지한 4지선다 제작에 부적합하여 제외 표시했다. 이 간단한 검사만으로 모든 부적합 문항이 걸러지는 것은 아니다.
- ScienceQA: 공식 pid_splits.json의 train ID이면서 image 필드가 있는 문항. 실제 이미지 의존성·가독성은 미검토다.
- MMMU·별도 개발·최종 평가셋과의 중복 검사는 미실행이다. 전체 후보의 train_eligible은 false다.
- 원본 이미지·CSV·lecture·solution·정답은 검수 후 사용한다. 참고 해설·CSV·정답을 학생 입력에 넣지 않는다.

## 파일

- image_groups.jsonl: 그룹 번호, 담당, 이미지 경로, 해당 문항 ID 목록.
- source_manifest.jsonl: 전체 후보와 원본 질문·정답·보기·hint, 출처·검수 상태.
- A_candidates.jsonl / B_candidates.jsonl / C_candidates.jsonl / D_candidates.jsonl: 담당자별 후보. ScienceQA lecture·solution도 제작 참고용으로 포함.
- scienceqa_no_image_exclusions.jsonl: 이미지가 없어 제외한 ScienceQA train ID.
- validation_report.json: 출처 파일 해시·기록된 revision, 수량과 배정 검사 결과, 미검사 범위.
- index.html: 네트워크 없이 질문 검색·담당 필터가 가능한 공용 목록. 원본 이미지 링크를 열 때는 네트워크가 필요하다.
- build_assignment.ps1: PowerShell 7용 재현 스크립트. 원본 JSON은 이전 예시 폴더 또는 -SourceRoot로 제공한다. 공개된 목록의 그룹·담당이 달라지는 재생성은 중단한다.

## 이미지 확보

원본 이미지는 이 배정 묶음에 포함하지 않았다. ChartQA는 각 행의 image_url 또는 공식 ChartQA Dataset 폴더를 사용한다. image_relative_path는 ChartQA Dataset 폴더 기준이다. ScienceQA는 공식 저장소의 이미지 다운로드 방법을 따르고, train 폴더가 있는 이미지 루트 기준으로 image_relative_path를 연결한다.

- ChartQA: https://github.com/vis-nlp/ChartQA/tree/$chartRev
- ScienceQA: https://github.com/lupantech/ScienceQA/tree/$scienceRev

이 배정 파일은 팀 내부 작업용이다. 원본 질문·해설을 포함하므로 공개 재배포 전 출처별 이용 조건을 확인한다. ScienceQA는 공식 CC BY-NC-SA 4.0 조건, ChartQA는 저장소 GPL-3.0 표기와 차트 원출처 조건을 유지한다.
"@
Write-Text (Join-Path $OutputRoot 'README.md') $readme
$checksums = Get-ChildItem -LiteralPath $OutputRoot -File | Where-Object { $_.Name -notin @('checksums.json','era_image_assignment_20261015.zip','independent_validation.json') } | ForEach-Object { [ordered]@{file=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$_.Length} }
Write-Json (Join-Path $OutputRoot 'checksums.json') @($checksums)
$report | ConvertTo-Json -Depth 12
