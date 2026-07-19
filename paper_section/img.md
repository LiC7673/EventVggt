# Method figures: wireframe drafts and drawing instructions

This file specifies what each method figure should communicate. The diagrams
below are layout drafts rather than final artwork. Use the same variable names
as `method_streamvggt_revised.tex`.

## Figure 1: Overall architecture

### Wireframe

```text
 TRAINING-ONLY SUPERVISION (top, dashed border)
 ┌──────────────────────┐             ┌────────────────────────┐
 │ controlled E_geo     │────────────▶│ Geo teacher feature F_G│
 └──────────────────────┘             └──────────┬─────────────┘
                                                 │ stop-gradient
 ┌──────────────────────┐                        │
 │ reference RGB ev_0   │── Frozen StreamVGGT ─▶│ HDR teacher tokens Z_0
 └──────────────────────┘                        │
                                                ▼
══════════════════════════════════════════════════════════════════════
 INFERENCE PATH

 Multi-LDR RGB I_L                              full event E_full
        │                                               │
        ▼                                               ▼
┌────────────────────┐                         ┌────────────────────┐
│ Frozen StreamVGGT  │                         │ shared EventEncoder│
│ RGB Aggregator     │                         └─────────┬──────────┘
└─────────┬──────────┘                                   │ F_full
          │ Z_L                                           ▼
          │                                    ┌────────────────────┐
          │                                    │ Full→Geo Aligner   │
          │                                    └─────────┬──────────┘
          │                                              │ F_hat_geo
          │                     ┌────────────────────────┴─────────┐
          │                     ▼                                  ▼
          │            ┌────────────────┐                 ┌────────────────┐
          │            │ C_fusion       │                 │ dNormal head   │
          │            │ reliability    │                 │ predict ∇N     │
          │            └───────┬────────┘                 └───────┬────────┘
          │                    │ selected event token             │
          ▼                    ▼                                  │
┌────────────────────────────────────┐                            │
│ Event-conditioned HDR Adapter      │                            │
│ Z_H = Z_L + A_H(Z_L,T_event)       │                            │
└─────────────────┬──────────────────┘                            │
                  ▼                                               │
         ┌──────────────────┐                                     │
         │ Frozen DepthHead │                                     │
         └────────┬─────────┘                                     │
                  │ D_base, N_base                                │
                  └────────────────────┬──────────────────────────┘
                                       ▼
                           ┌────────────────────────┐
                           │ Pixel Depth Refiner    │
                           │ event + ∇N + coarse G  │
                           │ + C_refine (NO RGB)    │
                           └────────────┬───────────┘
                                        ▼
                           D_final = D_base exp(ΔD)
                                        │
                                        ▼
                           N_final = NormalFromDepth(D_final)
```

### How to draw it

- Use a two-row design. Put all training-only teachers in a pale gray band at
  the top with dashed outlines. Put the deployable path in the main lower row.
- Use blue for RGB modules, orange for event modules, purple for fusion, and
  green for geometry outputs/losses.
- Draw the frozen StreamVGGT blocks with a small snowflake icon. Never draw
  raw RGB entering the pixel refiner.
- Draw `E_geo` only in the training band. The inference event must be labelled
  `E_full` or `E_cur`.
- Make the two uses of reliability explicit: `C_fusion` on event patch tokens
  and `C_refine` beside the pixel refiner. Do not collapse them into one scalar.
- Put a bold `+` inside the HDR Adapter block to show residual token addition.
- Put `exp(ΔD)` beside the final depth arrow to show log-depth refinement.
- Recommended caption message: RGB restores global/absolute geometry; selected
  events restore missing token evidence and constrain local differential
  geometry.

## Figure 2: Controlled event attribution and Full-to-Geo transfer

### Wireframe

```text
                         shared weights
 E_geo ───────────────▶ [ EventEncoder ] ─────▶ F_geo (teacher, stop-grad)
                                                       ▲
                                                       │ L_F→G
 E_full ──────────────▶ [ EventEncoder ] ─▶ [Aligner] ─┘
       │                                      │ F_hat_geo
       │                                      ▼
       │        RGB_L + coarse depth/normal ─▶[ContributionNet]
       │                                      │
       ├─ event mass |E_full|                 ▼
 E_geo └─ event mass |E_geo| ─────────────▶ C_gt = |E_geo|/(|E_full|+eps)
                                              │
                                      L_C ◀── C_pred

 INFERENCE: E_full → EventEncoder → Aligner → C_pred; E_geo removed
```

### How to draw it

- Put the two event streams side by side and use one encoder box with a shared
  weight symbol. Do not draw two unrelated encoders.
- Place the aligner only on the `E_full` branch. This is the most important
  structural detail in this figure.
- Use a stop-gradient octagon on `F_geo`.
- Show two different supervision arrows: feature alignment (`L_F→G`) and
  source attribution (`L_C`). This prevents readers from interpreting the
  method as only ordinary token distillation.
- Add three small event thumbnails: full, controlled geometry, and predicted
  reliability. Use the same spatial crop and the same color scale.
- Add a bottom inference strip in which the `E_geo` branch is crossed out.

## Figure 3: Multi-LDR event-conditioned HDR token completion

### Wireframe

```text
 I_LDR ── Frozen RGB Aggregator ──▶ Z_L ───────────────┐
                                                        │ +
 selected event feature ─ Projection/Pooling ─▶ T_E ─▶ HDR Adapter ─▶ Z_H
                                                                         │
 I_ev0/HDR ─ Frozen RGB Aggregator ─▶ Z_0 (stop-grad) ───── L_HDR ───────┘
                                                                         │
                                                                         ▼
                                                               Frozen DepthHead
                                                                         │
                                                                    D_base
```

### How to draw it

- Draw at least three LDR thumbnails (`ev_2`, `ev_5`, `ev_10`) feeding the same
  RGB aggregator to communicate exposure variation.
- The reference `ev_0` must appear only as a training teacher, not as an input
  to the final model.
- Inside the adapter, draw two narrow lanes:
  `RGB context` and `event modulation`, followed by a multiplication symbol.
  Then draw residual addition to `Z_L`.
- Add a note: `T_E=0 ⇒ Z_H=Z_L`. This visually explains the exact identity
  path when no event evidence is available.
- Display token grids rather than images after the aggregator. The target is
  HDR-like geometry features, not HDR image reconstruction.

## Figure 4: Differential-normal guided pixel refinement

### Wireframe

```text
 aligned dense event feature ──────────────────────────────┐
                                                          │
 event dNormal head ──▶ predicted (∂xN, ∂yN) ─────────────┤
                                                          │
 D_base ──▶ log(D_base) ──────────────────────────────────┤
      └──▶ NormalFromDepth ──▶ N_base ────────────────────┤
                                                          ▼
 C_refine ───────────────────────────────────────▶ Pixel Refiner
                                                          │
                                                  bounded ΔlogD
                                                          │
                           D_final = D_base exp(ΔlogD) ◀───┘
                                      │
                                      ▼
                       N_final = NormalFromDepth(D_final)
                                      │
                       ┌──────────────┴───────────────┐
                       ▼                              ▼
                 L_depth / L_normal             L_DN / L_HF
```

### How to draw it

- Show all tensors at full image resolution to distinguish this branch from
  patch-token fusion.
- Mark the refiner with `RGB pixels/tokens: none`.
- Visualize predicted and GT normal-derivative magnitude next to each other,
  using one fixed color range. Include an error map as the third panel.
- Show the event-support/recency mask as a transparent overlay rather than a
  hard crop. The network predicts dense updates, while derivative supervision
  is trusted only where evidence is valid.
- Use a diverging blue-white-red color map for `ΔlogD`; keep the same fixed
  physical/log range for all compared methods.
- To demonstrate improvement, add
  `|D_final-D_gt|-|D_base-D_gt|`: green/negative means improvement and
  red/positive means degradation. This is stronger evidence than an
  independently auto-scaled update image.

## Recommended paper layout

```text
 Main paper Figure 2: Figure 1 overall architecture (full page width)
 Main paper Figure 3: Figure 2 attribution + Figure 3 Multi-LDR (two columns)
 Main paper Figure 4: Figure 4 refinement plus qualitative result
 Supplement: detailed training schedule and all intermediate heatmaps
```

For the main overview, avoid more than seven large boxes. Put losses and
training-only supervision above the main inference arrow so the reader can
trace the deployed path from left to right without crossing lines.

