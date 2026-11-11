param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('status', 'pause', 'resume')]
    [string]$Action
)
$ErrorActionPreference = 'Stop'
$remoteHost = 'rlatjswo@aix408-40'
$remoteRoot = '/home/rlatjswo/mmmu_multiseed_20261011'
& ssh -o BatchMode=yes -o ConnectTimeout=15 $remoteHost "/home/rlatjswo/mmdl_test/.venv/bin/python $remoteRoot/control_mmmu_multiseed.py $Action --root $remoteRoot"
if ($LASTEXITCODE -ne 0) { throw "Campaign $Action failed; review the error above." }
