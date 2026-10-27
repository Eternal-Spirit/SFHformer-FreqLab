# scripts/auto_pipeline.ps1
param(
    [int]$WaitPid = 33832
)

$ErrorActionPreference = "Continue"
$Root = "D:\SFHformer-FreqLab"
$Python = "$Root\.venv\python.exe"

Write-Host "=========================================================="
Write-Host "AUTO-PIPELINE: Waiting for R1 Baseline (PID $WaitPid) to finish..."
Write-Host "=========================================================="

if ($WaitPid -gt 0) {
    Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
}

Write-Host ">>> R1 Baseline training completed! Starting Phase 9: R2 Gated Training..."
Set-Location -Path $Root

# Launch Phase 9: R2 Gated training
& $Python scripts/train_freq.py --model SFHformer_m_gated --gate --iters 20000 --batch-size 1 --seed 0 --run r2_gated --log-every 100 --val-every 2000 --save-every 2000 --num-workers 2

Write-Host ">>> R2 Gated training completed! Starting Phase 11-12: Evaluation Sweeps..."

# Run Phase 11 & 12 sweeps
& $Python scripts/sweep.py --runs r1_baseline r2_gated --out results/sweep.csv

Write-Host ">>> Sweeps completed! Generating plots and RESULTS.md..."
& $Python scripts/plot_results.py --csv results/sweep.csv --out results

Write-Host ">>> Generating spectral analysis figures..."
& $Python scripts/spectrum_figs.py --runs r1_baseline r2_gated

$Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
Set-Content -Path "$Root\results\pipeline_complete.txt" -Value "Full pipeline (R1, R2, Sweeps, Figures, RESULTS.md) completed at $Timestamp"

Write-Host "=========================================================="
Write-Host "ALL PHASES COMPLETE! Check results/RESULTS.md"
Write-Host "=========================================================="
