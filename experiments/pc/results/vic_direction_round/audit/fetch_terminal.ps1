$ErrorActionPreference = 'Stop'
$round = Split-Path -Parent $PSScriptRoot
$server = 'rlatjswo@aix408-40'
$connection = @('-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=45')
$ready = "cd ~/vic_direction_round && ~/mmdl_test/.venv/bin/python -c 'import sys; from campaign import ROOT,FULL,wait_event; wait_event(ROOT / sys.argv[1],FULL+900)' terminal_receipt.json"

function Assert-ExternalSuccess {
    if ($LASTEXITCODE -ne 0) { throw "External command failed with exit code $LASTEXITCODE" }
}

# Completion is signaled by an atomic server receipt after scoring and archival.
& ssh @connection $server $ready
Assert-ExternalSuccess
foreach ($file in @('terminal_receipt.json','terminal.tgz')) {
    & scp @connection "${server}:vic_direction_round/$file" (Join-Path $round $file)
    Assert-ExternalSuccess
}
$receipt = Get-Content -LiteralPath (Join-Path $round 'terminal_receipt.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$archive = Join-Path $round 'terminal.tgz'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt.archive_sha256) {
    throw 'Terminal archive checksum mismatch'
}
& tar -xzf $archive -C $round
Assert-ExternalSuccess
if ((Get-FileHash -LiteralPath (Join-Path $round 'final_report.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt.report_sha256) {
    throw 'Final report checksum mismatch'
}
if ((Get-FileHash -LiteralPath (Join-Path $round 'decision.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $receipt.decision_sha256) {
    throw 'Development decision checksum mismatch'
}
$receipt | ConvertTo-Json -Depth 5
