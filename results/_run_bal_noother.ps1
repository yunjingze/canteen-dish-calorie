$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONPATH = Join-Path $PSScriptRoot '..\..\.venv\Lib\site-packages'
$env:PYTHONIOENCODING = 'utf-8'
$py = Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'
$log = 'results\_cn_bal_noother.log'
'=== run bal_noother (waiting for focal) ===' | Out-File -FilePath $log -Encoding utf8
while (-not (Test-Path 'results/tables/cn241_21_focal.json')) { Start-Sleep -Seconds 30 }
'focal done, starting bal_noother' | Out-File -FilePath $log -Encoding utf8 -Append
$args = @('scripts\train_cn241.py','--mode','finetune','--data-root','data/splits/cn241_21',
          '--classes','results/tables/cn241_21_classes.json','--init','results/runs/cn241_all_pre/best.pt',
          '--epochs','8','--batch-size','64','--workers','4','--eval-workers','4',
          '--class-balance','weights','--balance-cap','4','--balance-skip','other_cn',
          '--out','results/tables/cn241_21_bal_noother.json','--save-dir','results/runs/cn241_21_bal_noother')
& $py @args *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
'exit=' + $LASTEXITCODE | Out-File -FilePath $log -Encoding utf8 -Append
'ALL DONE' | Out-File -FilePath $log -Encoding utf8 -Append