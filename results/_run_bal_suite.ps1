$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONPATH = Join-Path $PSScriptRoot '..\..\.venv\Lib\site-packages'
$env:PYTHONIOENCODING = 'utf-8'
$py = Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'
$common = @('scripts\train_cn241.py','--mode','finetune','--data-root','data/splits/cn241_21',
            '--classes','results/tables/cn241_21_classes.json','--init','results/runs/cn241_all_pre/best.pt',
            '--epochs','8','--batch-size','64','--workers','4','--eval-workers','4')

$runs = @(
    @{ tag = 'bal';   extra = @('--class-balance','weights','--balance-cap','4') },
    @{ tag = 'focal'; extra = @('--class-balance','weights','--balance-cap','4','--focal-gamma','1.0') }
)
foreach ($r in $runs) {
    $out = "results/tables/cn241_21_$($r.tag).json"
    $dir = "results/runs/cn241_21_$($r.tag)"
    "=== run $($r.tag) -> $out ===" | Out-File -FilePath "results\_cn_$($r.tag).log" -Encoding utf8
    & $py @common @($r.extra) --out $out --save-dir $dir *>&1 | Out-File -FilePath "results\_cn_$($r.tag).log" -Encoding utf8 -Append
    "exit=$LASTEXITCODE" | Out-File -FilePath "results\_cn_$($r.tag).log" -Encoding utf8 -Append
}
'ALL DONE' | Out-File -FilePath results\_cn_suite.log -Encoding utf8 -Append
