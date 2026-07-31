$ErrorActionPreference = "Stop"

$Root = if ($env:EVENT_ROOT) {
    $env:EVENT_ROOT
} else {
    "F:\TreeOBJ\reflective_raw\Actaeon_Anodized_Red\events_additive"
}

python tools/visualize_additive_event_components.py `
    --root "$Root" `
    --output "$Root\vis_event_components"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
