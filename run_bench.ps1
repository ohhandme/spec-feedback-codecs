param(
    [string]$TestRoot=(Join-Path $PSScriptRoot 'release-tests'),
    [string]$Out=(Join-Path $PSScriptRoot ('bench-' + [guid]::NewGuid().ToString('N'))),
    [string]$Python='C:/Users/asus/.raft/agents/6946fc4f-34e5-42f8-abf5-cd3fd9376a5e/review-v71-next-heads-20261108/cpu-env/Scripts/python.exe',
    [string]$Deps='C:/Users/asus/.raft/agents/6946fc4f-34e5-42f8-abf5-cd3fd9376a5e/task58-draft-opd/cpu-deps'
)
$env:PYTHONPATH=$Deps
$env:HF_HUB_OFFLINE='1'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& $Python (Join-Path $PSScriptRoot 'tests/bench_codecs.py') $TestRoot $Out
exit $LASTEXITCODE
