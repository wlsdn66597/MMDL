param(
    [string]$Python = 'C:\Users\sunj0\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $workspace 'results/mmmu_multiseed_20261011'
$remoteHost = 'rlatjswo@aix408-40'
$remoteRoot = '/home/rlatjswo/mmmu_multiseed_20261011'
New-Item -ItemType Directory -Force $destination | Out-Null
$snapshot = & ssh -o BatchMode=yes -o ConnectTimeout=15 $remoteHost "cat $remoteRoot/status.json"
if ($LASTEXITCODE -ne 0) { throw 'Cannot read remote campaign status.' }
$state = ($snapshot -join "`n") | ConvertFrom-Json
foreach ($name in @('status.json','aggregate.json','item_stability.json','report.html')) {
    & scp "${remoteHost}:${remoteRoot}/$name" (Join-Path $destination $name)
    if ($LASTEXITCODE -ne 0) { throw "Failed to transfer $name" }
}
if ($state.phase -eq 'repeatability_mismatch' -or $state.completed_new_runs -contains 'baseline_seed3407_repeat') {
    & scp "${remoteHost}:${remoteRoot}/repeatability.json" (Join-Path $destination 'repeatability.json')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to transfer repeatability evidence.' }
}
if ($state.state -ne 'complete') {
    Write-Output "Current state: $($state.state) / $($state.phase). Summary snapshot downloaded; full results are not complete."
    return
}
$archiveName = 'mmmu_multiseed_20261011.tar.gz'
$remoteArchive = "/home/rlatjswo/$archiveName"
$checksum = & ssh -o BatchMode=yes $remoteHost "cat $remoteArchive.sha256"
if ($LASTEXITCODE -ne 0) { throw 'Final archive is not ready. Run this sync again after packaging finishes.' }
$expectedHash = (($checksum -join '') -split '\s+')[0]
if ($expectedHash -notmatch '^[0-9a-f]{64}$') { throw 'Invalid archive checksum.' }
$archive = Join-Path (Split-Path -Parent $destination) $archiveName
& scp "${remoteHost}:$remoteArchive" "$archive.partial"
if ($LASTEXITCODE -ne 0) { throw 'Archive transfer failed.' }
$actualHash = (Get-FileHash -LiteralPath "$archive.partial" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) { throw 'Archive checksum mismatch; not extracting.' }
Move-Item -LiteralPath "$archive.partial" -Destination $archive -Force
& $Python (Join-Path $PSScriptRoot 'extract_mmmu_multiseed.py') $archive
if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed.' }
& $Python (Join-Path $PSScriptRoot 'mmmu_multiseed_report.py') --root $destination --baseline3407 (Join-Path $workspace 'results/mmmu_first_baseline_20261009/baseline_seed3407') --baseline42 (Join-Path $workspace 'results/mmmu_visual_tokens_seed42_20261010/baseline_seed42')
if ($LASTEXITCODE -ne 0) { throw 'Downloaded result verification failed.' }
@{passed=$true; sha256=$actualHash; source="${remoteHost}:$remoteArchive"; verified_at_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destination 'transfer_verification.json') -Encoding utf8
Write-Output "All five seeds downloaded and validated: $destination/report.html"
