$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$InputDir = "E:\result\eventvgg\rgb_f"
$OutputFile = Join-Path $InputDir "rgb_f_four_scene_pose_summary.json"
$PythonScript = Join-Path $PSScriptRoot "summarize_rgb_f_pose_by_ev.py"

if (-not (Test-Path -LiteralPath $InputDir -PathType Container)) {
    throw "Input directory does not exist: $InputDir"
}
if (-not (Test-Path -LiteralPath $PythonScript -PathType Leaf)) {
    throw "Python summary script does not exist: $PythonScript"
}

Push-Location $ProjectRoot
try {
    python $PythonScript `
        --input-dir $InputDir `
        --output $OutputFile
    if ($LASTEXITCODE -ne 0) {
        throw "Pose summary failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Finished. JSON saved to:"
Write-Host $OutputFile
