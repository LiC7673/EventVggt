# Signed multi-scale pixel model

This is an independent experiment path. Use only:

```bash
bash paired_token_reliability/run_signed_multiscale_12train_4test.sh data.root=/real/data/root
```

The loader creates the historical 10-channel voxel only at the data boundary.
The model immediately converts it to five signed temporal channels with values
in `{-1, 0, +1}`. Every learned event module receives only these five channels.

Both absolute event normals and bounded relative depth updates come from a
stride-one, full-resolution multi-dilation encoder. No event update is injected
at a DPT/ViT patch grid, and no resize/interpolation occurs in the event-normal
or pixel-depth path.

Before encoding, bin centers are exponentially decayed toward the current
window end: `w_i=exp(-(t_current-t_i)/tau)`. The default is
`EVENT_DECAY_TAU=0.003` seconds. Each saved training visualization also writes
`*_temporal_decay.png` containing the signed projection, decayed mass, and the
five numerical bin weights.
