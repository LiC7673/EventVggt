param(
    [string]$Scene = "F:\TreeOBJ\reflective_raw\Bearded Man_Ceramic_Glazed_White",
    [int]$MaskThreshold = 250,
    [switch]$Overwrite
)

$scriptPath = Join-Path $PSScriptRoot "visualize_ldr_with_mask.py"
$arguments = @(
    $scriptPath,
    "--scene", $Scene,
    "--mask-threshold", $MaskThreshold
)
if ($Overwrite) { $arguments += "--overwrite" }
python @arguments
