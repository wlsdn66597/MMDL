param([ValidateSet('audit','final')][string]$Stage='audit')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$opts=@('-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=40')
$hostspec='rlatjswo@aix408-40'
if ($Stage -eq 'audit') { $receipt='audit_ready.json'; $archive='audit_ready.tgz' }
else { $receipt='archive_receipt.json'; $archive='results.tgz' }
$wait="cd ~/era_r_only_round && ~/mmdl_test/.venv/bin/python -c 'import sys;from campaign import ROOT,waitfile,read;waitfile(ROOT/sys.argv[1],read(ROOT/sys.argv[2])[sys.argv[3]]+180)' $receipt started.json deadline"
& ssh @opts $hostspec $wait
if ($LASTEXITCODE -ne 0) { throw "Remote completion check failed" }
foreach ($file in @($receipt,$archive)) {
    & scp @opts "${hostspec}:era_r_only_round/$file" (Join-Path $root $file)
    if ($LASTEXITCODE -ne 0) { throw "Failed to fetch $file" }
}
$record=Get-Content -LiteralPath (Join-Path $root $receipt) -Raw -Encoding utf8 | ConvertFrom-Json
if ((Get-FileHash -LiteralPath (Join-Path $root $archive) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) { throw 'Archive checksum mismatch' }
& tar -xzf (Join-Path $root $archive) -C $root
if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed' }
$record | ConvertTo-Json
