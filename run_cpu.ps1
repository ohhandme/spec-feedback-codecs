param(
    [string]$Python = 'C:/Users/asus/.raft/agents/6946fc4f-34e5-42f8-abf5-cd3fd9376a5e/review-v71-next-heads-20261108/cpu-env/Scripts/python.exe',
    [string]$Deps = 'C:/Users/asus/.raft/agents/6946fc4f-34e5-42f8-abf5-cd3fd9376a5e/task58-draft-opd/cpu-deps',
    [string]$Out = (Join-Path $PSScriptRoot ('cpu-run-' + [guid]::NewGuid().ToString('N')))
)
$ErrorActionPreference = 'Stop'
$demo = Join-Path $PSScriptRoot 'demo'
$tests = Join-Path $PSScriptRoot 'tests'
$env:PYTHONPATH = $Deps
$env:KV_REVIEW_DEMO = $demo
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$outDir = (New-Item -ItemType Directory -Path $Out).FullName
$cfg = Get-Content -LiteralPath (Join-Path $demo 'variant_config.json') -Raw | ConvertFrom-Json
foreach ($entry in $cfg.contract.source_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $demo $entry.Name)).Hash -ne $entry.Value) {
        throw "Source hash mismatch: $($entry.Name)"
    }
}
if ((Get-FileHash -LiteralPath (Join-Path $demo 'roster.jsonl')).Hash -ne $cfg.contract.roster_sha256) {
    throw 'Roster hash mismatch'
}
function Invoke-Check ([string]$Name, [string[]]$PyArgs) {
    & $Python @PyArgs *> (Join-Path $outDir "$Name.log")
    if ($LASTEXITCODE -ne 0) { throw "$Name failed. See $outDir/$Name.log" }
    Write-Output "$Name passed"
}
Invoke-Check 'regression' @("$tests/review_regression.py")
Invoke-Check 'parity' @("$tests/cpu_parity.py", '--out', "$outDir/cpu_parity.json")
Invoke-Check 'terminal' @("$tests/independent_terminal_parity.py", '--out', "$outDir/terminal.json")
foreach ($mode in 'reference', 'boundary', 'kv') {
    Invoke-Check "tcp-$mode" @("$demo/cpu_test.py", '--implementation', $mode, '--out', "$outDir/tcp-$mode")
}
Write-Output "Evidence: $outDir"
