$ErrorActionPreference = "Stop"

$Root = if ($env:REFLECTIVE_ROOT) {
    $env:REFLECTIVE_ROOT
} else {
    "F:\TreeOBJ\reflective_raw"
}

python tools/visualize_all_scenes_additive_event_frames.py `
    --root "$Root" `
    --frames 120 `
    --fps 120 `
    --mask-threshold 250

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
