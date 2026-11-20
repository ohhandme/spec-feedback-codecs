param(
    [string]$Out = (Join-Path $PSScriptRoot ('test-' + [guid]::NewGuid().ToString('N'))),
    [string]$Python = 'C:/Users/asus/.raft/agents/6946fc4f-34e5-42f8-abf5-cd3fd9376a5e/review-v71-next-heads-20261108/cpu-env/Scripts/python.exe',
    [string]$Deps = 'C:/Users/asus/.raft/agents/6946fc4f-34e5-42f8-abf5-cd3fd9376a5e/task58-draft-opd/cpu-deps'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH=$Deps
$env:HF_HUB_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES=''
$env:KV_REVIEW_DEMO=Join-Path $PSScriptRoot 'demo'
$outDir=(New-Item -ItemType Directory -Path $Out).FullName
function Run([string]$name, [string[]]$arguments) {
    & $Python @arguments *> (Join-Path $outDir "$name.log")
    if ($LASTEXITCODE -ne 0) { throw "Failed $name. See $outDir/$name.log" }
    Write-Output "$name passed"
}
Push-Location $PSScriptRoot
try {
    $cfg=Get-Content -Raw demo/variant_config.json | ConvertFrom-Json
    foreach ($prop in $cfg.contract.source_sha256.PSObject.Properties) {
        if ((Get-FileHash (Join-Path demo $prop.Name)).Hash -ne $prop.Value) { throw "Source mismatch $($prop.Name)" }
    }
    Run train @('tests/train_codecs.py',$outDir)
    Run regression @('tests/review_regression.py')
    Run parity @('tests/cpu_parity.py','--out',"$outDir/parity.json")
    Run terminal @('tests/independent_terminal_parity.py','--out',"$outDir/terminal.json")
    foreach ($method in 'pca_int8','mlp_int8','grouped_vq8') {
        foreach ($mode in 'reference','kv') {
            Run "$method-$mode" @('demo/cpu_test.py','--codec',$method,'--implementation',$mode,'--checkpoint',"$outDir/$method.pt",'--out',"$outDir/$method-$mode")
        }
    }
    Run legacy @('demo/cpu_test.py','--implementation','kv','--out',"$outDir/legacy")
    Run precision @('tests/precision_check.py',$outDir)
    Run audit @('tests/audit_methods.py',$outDir)
} finally { Pop-Location }
Write-Output "Evidence $outDir"
