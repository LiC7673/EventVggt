# Diffuse Geometry-event Oracle Ablation

This ablation uses the additive `geometry_motion` / diffuse-proxy event branch
instead of the noisy `full` event stream.

It tests the question:

```text
If the model receives only geometry-related diffuse events, does the event
branch help more than full events?
```

Expected scene layout:

```text
scene_xxx/
  events_additive/
    geometry_motion/events.h5
    material_reflection/events.h5
    noise/events.h5
    full/events.h5
```

Run:

```bash
bash diffuse_event_ablation/run_geometry_event_reliability_2gpu.sh
```

Default behavior:

- trains the previous best `temporal_detail + image-guided reliability` setup;
- replaces the dataloader event source with
  `events_additive/geometry_motion/events.h5`;
- uses random LDR exposure sampling during training;
- dilates the event voxel itself before use with `GEOMETRY_EVENT_DILATE=5`.

The dilation is intentional: geometry events are sparse and can otherwise be
removed by hard object masks or voxel resizing. Increase it to 7 if the event
support is visibly too thin:

```bash
GEOMETRY_EVENT_DILATE=7 bash diffuse_event_ablation/run_geometry_event_reliability_2gpu.sh
```

After training, the Scene-12 EAG3R-style manifest already includes
`geometry_event_oracle_scene12`, so it can be evaluated with:

```bash
bash ablation/run_scene12_heldout_eval.sh
```
