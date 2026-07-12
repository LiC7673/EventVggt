下面这份可以直接作为 `toDo.md` 给 Codex。它不是让 Codex局部打补丁，而是要求它按你现在最终收敛的故事，完成一次**最小侵入式但结构明确的重构**。

````markdown
# EvGeo-VGGT Final Refactoring TODO

## 0. Task objective

Refactor the current Event-VGGT pipeline into the following final design:

1. VGGT keeps stable global/coarse RGB geometry.
2. ContributionNet learns geometry-attributable event responses.
3. The event branch is strictly event-conditioned and cannot use RGB features to generate arbitrary geometry updates.
4. Contribution is applied only once through a continuous soft gate.
5. Events primarily refine local surface orientation / normal variation.
6. Depth is constrained by normal-depth geometric consistency instead of being freely rewritten by an unconstrained event depth residual.
7. Event updates are restricted to high-resolution/local geometry pathways.
8. Empty-event and low-contribution regions must remain close to the RGB-only geometry.

The final method should implement:

\[
\text{source-supervised event attribution}
\rightarrow
\text{single soft gating}
\rightarrow
\text{event-only local normal refinement}
\rightarrow
\text{depth-normal geometric consistency}.
\]

Do not redesign the dataset or VGGT backbone from scratch. Reuse the current code wherever possible.

---

# 1. Current failure modes

The current implementation produces several artifacts:

- depth holes after adding the event adapter;
- event contours becoming artificial depth depressions;
- patch-grid / stripe artifacts in predicted normals;
- large geometry updates outside actual event support;
- contribution maps acting as generic attention rather than geometry attribution.

The main structural causes are:

## 1.1 RGB shortcut inside GeometryAdapter

Current form:

```python
raw_update = adapter(
    torch.cat([rgb_feature, event_feature], dim=1)
)
````

This permits:

[
F_{\mathrm{rgb}}\rightarrow \Delta F,
]

as soon as the event gate opens.

Therefore the module is effectively:

```text
RGB-conditioned refinement with event gating
```

instead of:

```text
event-conditioned local geometry refinement
```

## 1.2 Repeated or hard gating

Current logic may contain both:

```python
selected_event = contribution * event_voxel
```

and:

```python
applied_update = contribution * raw_update
```

which approximately creates:

[
\Delta F_{\mathrm{effective}}\propto C^2E.
]

Other paths use:

```python
event_support = contribution > 1e-6
```

which converts a continuous score into a discontinuous patch-grid mask.

## 1.3 Sparse events become dense updates

The event encoder contains spatial/temporal convolution, normalization, and interpolation. These operations can create nonzero event features outside the original event support.

The current pipeline only guarantees that an entirely empty event tensor falls back to RGB. It does not guarantee local zero update where the event support is absent.

## 1.4 Four-scale, phase-aligned event injection

All four DPT adapters currently operate from the same patch grid, approximately:

```text
392 / 14 = 28
518 / 14 = 37
```

Hard or strongly varying updates on this common grid are injected into multiple DPT scales. Their aligned artifacts reinforce one another and become especially visible after depth-to-normal differentiation.

## 1.5 Event branch has excessive freedom over depth

The event branch can produce large arbitrary feature updates, which the VGGT head may translate into artificial depth valleys or holes.

An event edge provides evidence of a brightness transition, but does not directly determine:

* the sign of a depth change;
* whether the surface is concave or convex;
* whether the response is a material boundary;
* whether the event is caused by reflection or occlusion.

Therefore the event branch must not freely generate unconstrained depth geometry.

---

# 2. Final target architecture

The final architecture should follow:

```text
Overexposed RGB
      |
Frozen / low-LR VGGT RGB backbone
      |
      +------------------------------+
      |                              |
RGB geometry features F_rgb    Coarse geometry
                               D_c, P_c, N_c
      |
      |                         Full event voxel V
      |                                |
      |                        ContributionNet
      |                                |
      |                         contribution C
      |                                |
      |                         Event Encoder
      |                                |
      |                     event-only local features
      |                                |
      |                    Normal Refinement Adapter
      |                                |
      |                   bounded local normal update
      |                                |
      +---------- depth-normal geometric consistency
                                       |
                             Final D / P / N
```

Camera pose must remain RGB-only.

Do not allow the event branch to alter the camera head.

---

# 3. Module A: Geometry-attributable ContributionNet

## 3.1 Contribution semantics

ContributionNet predicts:

[
C\in[0,1]^{B\times S\times 1\times H\times W}.
]

It should represent:

```text
how strongly the local event response can be attributed to geometry-related image changes
```

It is not:

* a binary event mask;
* a saturation mask;
* a bridge mask;
* a depth-edge label;
* a generic attention weight.

## 3.2 Synthetic attribution supervision

Use Blender full-event and geometry-event streams.

Given aligned event voxels:

[
V_{\mathrm{full}},\quad V_{\mathrm{geo}},
]

compute event mass:

[
A_{\mathrm{full}}(x,y)=
\sum_t|V_{\mathrm{full}}(t,x,y)|,
]

[
A_{\mathrm{geo}}(x,y)=
\sum_t|V_{\mathrm{geo}}(t,x,y)|.
]

Construct the soft target:

[
C^*(x,y)=
\operatorname{clip}
\left(
\frac{A_{\mathrm{geo}}(x,y)}
{A_{\mathrm{full}}(x,y)+\epsilon},
0,1
\right).
]

Suggested name in code:

```python
event_geometry_attribution_target
```

or:

```python
geometry_event_ratio
```

Do not call it a binary geometry mask.

## 3.3 Attribution loss

Add:

```python
loss_attr = smooth_l1_loss(
    contribution_pred,
    contribution_target,
    reduction="none",
)
```

Apply it only on valid event-supported regions, but keep the target continuous.

Recommended reduction:

```python
support = full_event_voxel.abs().sum(dim=event_channel_dim) > 0
loss_attr = masked_mean(loss_attr, support)
```

## 3.4 Multi-LDR contribution consistency

For the same viewpoint and event observation under two RGB exposures:

[
C^a=\mathcal C(I^a,V,\operatorname{sg}(G_c^a)),
]

[
C^b=\mathcal C(I^b,V,\operatorname{sg}(G_c^b)).
]

Add:

[
\mathcal L_{C\text{-cons}}
==========================

\operatorname{mean}_{S_e}
|C^a-C^b|.
]

Only compare on shared event support.

Detach coarse RGB geometry context if it is supplied to ContributionNet.

This loss should prevent ContributionNet from simply reproducing the saturation pattern of the current RGB image.

## 3.5 Bridge usage

Bridge must not be:

```python
target = bridge
```

and must not enter inference.

Bridge may only be used as:

* optional loss weighting;
* sample selection;
* diagnostic visualization.

If retained, use:

[
W=M_{\mathrm{valid}}(1+\beta B)
]

instead of supervising geometry only inside (B).

---

# 4. Module B: Strictly event-only encoder and adapter

## 4.1 Remove RGB shortcut

Find all code equivalent to:

```python
adapter_input = torch.cat(
    [rgb_feature, event_feature],
    dim=1,
)
raw_update = adapter(adapter_input)
```

Replace with:

```python
raw_update = adapter(event_feature)
```

The adapter must not read `rgb_feature`.

Update constructor channel sizes:

```python
in_channels = event_channels
```

instead of:

```python
in_channels = rgb_channels + event_channels
```

Update all affected:

* `Conv2d`;
* projection layers;
* normalization layers;
* test shapes;
* checkpoint loading logic.

RGB remains the residual base:

```python
refined_feature = rgb_feature + event_update
```

but it must not be used to generate `event_update`.

## 4.2 Zero-preserving event encoder

Audit the event encoder for operations that violate local zero preservation.

Prefer:

* convolution layers with `bias=False`;
* local normalization or masked normalization;
* no global GroupNorm effect on zero-event regions unless masked afterward;
* explicit support masking after event encoding.

Do not remove the existing temporal encoder without need, but ensure a zero event tensor produces exactly or numerically near-zero event features.

Add a unit test:

```python
event_voxel = torch.zeros_like(event_voxel)
event_features = event_encoder(event_voxel)

assert max_abs(event_features) < tolerance
```

Use a realistic tolerance such as `1e-6` or a justified value.

---

# 5. Module C: Single continuous soft gating

## 5.1 Choose one gating location

Use the recommended final design:

```python
event_features = event_encoder(full_event_voxel)
raw_normal_update = normal_adapter(event_features)
normal_update = gate * raw_normal_update
```

Contribution must not also multiply the event voxel.

Remove:

```python
selected_event = contribution * event_voxel
```

from the final main path if contribution is used on the final normal/event update.

The final effective relation should be approximately:

[
\Delta N_{\mathrm{effective}}
\sim C,A(E),
]

not:

[
C^2A(E).
]

## 5.2 No hard contribution threshold

Remove code such as:

```python
event_support = contribution > 1e-6
```

and:

```python
applied_update = event_support.float() * raw_update
```

Use continuous resized contribution:

```python
gate = F.interpolate(
    contribution,
    size=raw_update.shape[-2:],
    mode="bilinear",
    align_corners=False,
)

applied_update = gate * raw_update
```

Do not threshold `gate`.

## 5.3 Soft event-support constraint

Contribution alone is not enough to guarantee local zero update because encoder features may spread spatially.

Compute original support:

```python
event_support_pixel = (
    event_voxel.abs().sum(dim=event_channel_dim, keepdim=True) > 0
).float()
```

Create a soft, locally dilated support map.

Recommended implementation:

1. Apply a small morphological dilation or max pooling in pixel space:

```python
support_dilated = F.max_pool2d(
    event_support_pixel,
    kernel_size=5,
    stride=1,
    padding=2,
)
```

2. Resize continuously to the update scale:

```python
support_gate = F.interpolate(
    support_dilated,
    size=raw_update.shape[-2:],
    mode="bilinear",
    align_corners=False,
)
```

3. Final gate:

```python
effective_gate = contribution_gate * support_gate
applied_update = effective_gate * raw_update
```

The support dilation provides local convolutional context without opening an entire `14×14` patch.

Do not use a `>1e-6` threshold after interpolation.

---

# 6. Module D: Normal-oriented local geometry refinement

## 6.1 Do not use unconstrained raw depth residual as the primary event output

Remove or disable any direct path equivalent to:

```python
delta_depth = depth_refiner(event_features)
depth_final = depth_coarse + delta_depth
```

when `delta_depth` is unrestricted.

The event branch should primarily predict local surface-orientation correction.

## 6.2 Coarse normals

Obtain coarse normal (N_c) consistently from either:

* the coarse VGGT point map; or
* coarse depth and correctly resized camera intrinsics.

Use one convention throughout training and evaluation.

Verify:

* normal coordinate frame;
* camera intrinsics after image resize;
* valid mask;
* depth scale.

## 6.3 Bounded normal residual

The event normal adapter outputs:

```python
raw_delta_normal
```

Use a bounded residual:

```python
delta_normal = normal_update_scale * torch.tanh(raw_delta_normal)
```

The configurable scale should initially be conservative.

Example config:

```yaml
normal_update_scale: 0.15
```

Do not assume this exact value is optimal; expose it in configuration.

Apply the single effective soft gate:

```python
delta_normal = effective_gate * delta_normal
```

Final normal:

```python
normal_final = F.normalize(
    normal_coarse + delta_normal,
    dim=normal_channel_dim,
    eps=1e-6,
)
```

## 6.4 High-resolution injection only

Do not inject identical event updates into all four DPT levels.

Add configuration:

```yaml
event_adapter_levels: [0, 1]
```

where these indices correspond to the highest-resolution/local geometry levels in the current implementation.

Disable event updates on the lower-resolution/global-semantic levels by default.

The implementation must clearly document which actual DPT feature maps correspond to these indices.

Retain an ablation option:

```yaml
event_adapter_levels: [0, 1, 2, 3]
```

but do not use it as the default.

---

# 7. Depth-normal geometric consistency

A separately predicted normal can become visually plausible while being inconsistent with the reconstructed depth. Add an explicit coupling.

## 7.1 Normal derived from final depth

Compute:

[
N_D=N(D_f,K)
]

using the existing differentiable `depth_to_normals` function.

Ensure intrinsics match the depth resolution.

## 7.2 Consistency loss

Add:

[
\mathcal L_{DN}
===============

1-\langle N_D,\operatorname{sg}(N_f)\rangle.
]

Recommended code:

```python
normal_from_depth = depth_to_normals(
    pred_depth,
    intrinsics,
)

loss_depth_normal_consistency = masked_mean(
    1.0 - (
        F.normalize(normal_from_depth, dim=normal_dim, eps=1e-6)
        *
        F.normalize(normal_pred.detach(), dim=normal_dim, eps=1e-6)
    ).sum(dim=normal_dim),
    valid_normal_mask,
)
```

Initially detach `normal_pred` in this consistency term so that the normal branch supervises depth rather than both branches moving toward an arbitrary compromise.

Expose an option to reverse or remove the detach for ablation, but default to detached normal target.

## 7.3 Conservative depth update

Prefer one of the following two implementations.

### Preferred first implementation

Keep the VGGT depth output as the final depth prediction and allow the normal consistency loss to update high-resolution geometry features through the shared decoder.

Do not add a separate event depth residual initially.

### Optional implementation

If a separate depth correction is required, make it bounded and local:

```python
raw_delta_depth = depth_local_head(event_features)
delta_depth = depth_update_scale * torch.tanh(raw_delta_depth)
delta_depth = effective_gate * delta_depth
depth_final = depth_coarse + delta_depth
```

Use a small configurable scale and strong outside-support regularization.

Do not enable this optional residual by default until the normal-only path has been tested.

---

# 8. Losses

## 8.1 Normal cosine loss

[
\mathcal L_N
============

1-\langle N_f,N_{\mathrm{gt}}\rangle.
]

Use valid foreground/geometry masks.

## 8.2 Normal gradient loss

Add local orientation variation supervision:

[
\mathcal L_{\nabla N}
=====================

|\nabla N_f-\nabla N_{\mathrm{gt}}|_1.
]

Implement finite differences in both x and y.

Only compare neighboring pixels where both pixels are valid.

Pseudo-code:

```python
pred_dx = pred_normal[..., :, 1:] - pred_normal[..., :, :-1]
gt_dx = gt_normal[..., :, 1:] - gt_normal[..., :, :-1]

pred_dy = pred_normal[..., 1:, :] - pred_normal[..., :-1, :]
gt_dy = gt_normal[..., 1:, :] - gt_normal[..., :-1, :]

valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
valid_dy = valid[..., 1:, :] & valid[..., :-1, :]

loss_normal_grad = (
    masked_l1(pred_dx, gt_dx, valid_dx)
    +
    masked_l1(pred_dy, gt_dy, valid_dy)
)
```

## 8.3 Depth loss

Retain current stable depth supervision.

If supported by the existing code, combine:

```text
scale-aware or scale-invariant depth loss
+
weak depth-gradient loss
```

Do not allow depth-gradient loss to dominate.

## 8.4 Update magnitude loss

Prevent the event branch from overwriting coarse geometry:

[
\mathcal L_{\mathrm{mag}}
=========================

|\Delta N|_1
]

and, if optional depth correction is enabled:

[
+\eta|\Delta D|_1.
]

## 8.5 Outside-support invariance

Enforce:

[
V=0 \Rightarrow \text{no local event update}.
]

Use:

[
\mathcal L_{\mathrm{outside}}
=============================

|(1-S)\odot\Delta N|_1.
]

Also constrain final depth outside the soft support:

[
\mathcal L_{\mathrm{depth-outside}}
===================================

|(1-S)\odot(D_f-D_c)|_1.
]

Use the soft/dilated support (S), not a patch-grid binary threshold.

## 8.6 Attribution and Multi-LDR losses

The complete contribution loss should include:

[
\mathcal L_C
============

\lambda_{\mathrm{attr}}\mathcal L_{\mathrm{attr}}
+
\lambda_{C\text{-cons}}\mathcal L_{C\text{-cons}}
+
\lambda_{\mathrm{budget}}\mathcal L_{\mathrm{budget}}.
]

Budget is optional and must not force a universal `0.5` average if decomposition supervision already provides a meaningful event ratio.

If kept, use a weak or scheduled budget term.

## 8.7 Total refinement loss

Suggested structure:

[
\mathcal L_{\mathrm{refine}}
============================

\lambda_D\mathcal L_D
+
\lambda_N\mathcal L_N
+
\lambda_{\nabla N}\mathcal L_{\nabla N}
+
\lambda_P\mathcal L_P
+
\lambda_{DN}\mathcal L_{DN}
+
\lambda_{\mathrm{mag}}\mathcal L_{\mathrm{mag}}
+
\lambda_{\mathrm{outside}}\mathcal L_{\mathrm{outside}}
+
\lambda_{\mathrm{depth-outside}}
\mathcal L_{\mathrm{depth-outside}}.
]

All weights must be configurable and logged.

Do not silently add losses without reporting both raw and weighted terms.

---

# 9. Training schedule

Implement three phases.

## Phase A: Attribution initialization

Train:

* ContributionNet only.

Freeze:

* RGB VGGT backbone;
* camera head;
* Event Encoder;
* Normal Refinement Adapter;
* optional depth local head.

Loss:

[
\mathcal L_A
============

\mathcal L_{\mathrm{attr}}
+
\lambda_{C\text{-cons}}\mathcal L_{C\text{-cons}}
+
\lambda_{\mathrm{budget}}\mathcal L_{\mathrm{budget}}.
]

This phase gives (C) an independent physical/source-attribution meaning.

## Phase B: Normal-oriented event refinement

Freeze ContributionNet initially.

Train:

* Event Encoder;
* high-resolution Normal Refinement Adapter;
* only the minimum required high-resolution geometry decoder parameters;
* keep RGB aggregator and camera head frozen.

Use predicted contribution (C), not only oracle (C^*).

Optionally mix predicted and target contribution during early Phase B:

```python
gate_train = teacher_ratio * contribution_target + (
    1.0 - teacher_ratio
) * contribution_pred.detach()
```

Anneal:

```text
teacher_ratio: 1.0 -> 0.0
```

over the early portion of Phase B.

Loss:

[
\mathcal L_B=\mathcal L_{\mathrm{refine}}.
]

## Phase C: Alternating calibration

Do not immediately unfreeze everything for long joint training.

Alternate:

### Step C1: Contribution update

Freeze Event Encoder and Refiner.

Update ContributionNet with:

[
\mathcal L_{\mathrm{attr}}
+
\lambda_{C\text{-cons}}\mathcal L_{C\text{-cons}}
+
\lambda_{\mathrm{task}}\mathcal L_{\mathrm{refine}}.
]

Use a small `lambda_task` so contribution does not lose attribution semantics.

### Step C2: Refiner update

Freeze ContributionNet.

Update Event Encoder and Refiner using:

[
\mathcal L_{\mathrm{refine}}.
]

Implement either:

* one epoch ContributionNet / one epoch Refiner; or
* configurable numbers of alternating batches.

After alternating calibration, optionally run at most one very-low-LR joint epoch.

---

# 10. Required configuration options

Add explicit config entries similar to:

```yaml
model:
  event_adapter_uses_rgb: false
  event_adapter_levels: [0, 1]
  normal_update_scale: 0.15
  enable_event_depth_residual: false
  depth_update_scale: 0.03
  support_dilation_kernel: 5
  contribution_gate_location: "normal_update"

loss:
  lambda_attr: 1.0
  lambda_contribution_consistency: 0.2
  lambda_budget: 0.0
  lambda_depth: 1.0
  lambda_normal: 2.0
  lambda_normal_gradient: 0.5
  lambda_point: 0.5
  lambda_depth_normal_consistency: 0.5
  lambda_update_magnitude: 0.01
  lambda_outside_support: 0.2
  lambda_depth_outside_support: 0.2
  lambda_task_for_contribution: 0.1

training:
  attribution_epochs: 5
  refinement_epochs: 15
  alternating_epochs: 4
  alternating_contribution_steps: 1
  alternating_refiner_steps: 1
  contribution_teacher_start: 1.0
  contribution_teacher_end: 0.0
```

These values are initial defaults only. Preserve command-line override support.

---

# 11. Debug metrics and visualization

Log the following for every training phase.

## Contribution statistics

```text
Cmean
Cstd
Cmin
Cmax
attribution MAE
attribution correlation
```

## Event-update statistics

For each enabled event adapter level:

```text
raw update norm
gated update norm
RGB feature norm
gated_update_norm / RGB_feature_norm
outside-support update norm
```

## Counterfactual geometry tests

At validation intervals compute:

1. full event;
2. zero event;
3. zero contribution;
4. reversed event time;
5. swapped event polarity;
6. oracle geometry event, when available.

Report:

```text
mean |D_full - D_zero|
mean angular difference N_full vs N_zero
mean |D_full - D_coarse|
outside-support depth change
```

## Visualization panels

Save:

* target RGB;
* reference RGB;
* full event projection;
* geometry event projection, when available;
* event temporal bins;
* predicted contribution;
* attribution target;
* soft event support;
* effective gate;
* coarse depth;
* final depth;
* signed depth difference `D_final - D_coarse`;
* absolute full-vs-zero depth difference;
* predicted normal;
* GT normal;
* normal angular error map;
* update magnitude per enabled scale.

Do not normalize coarse and final depth independently in difference analysis. Use the same range or direct signed difference.

---

# 12. Required unit and sanity tests

The implementation is not complete until these tests pass.

## Test 1: Entire zero event

Set:

```python
event_voxel.zero_()
```

Expect:

```text
event features approximately zero
normal update approximately zero
final depth approximately coarse depth
final normal approximately coarse normal
```

## Test 2: Local zero event

Construct an event tensor active only in a small region.

Expect:

* update concentrated within the soft dilated support;
* outside-support update below tolerance;
* no patch-sized rectangular update far from events.

## Test 3: Zero contribution

Force:

```python
contribution.zero_()
```

Expect:

```text
normal update = 0
final depth approximately coarse depth
```

## Test 4: Full contribution

Force:

```python
contribution.fill_(1)
```

Expect the strongest event-conditioned update, but still bounded by `normal_update_scale`.

## Test 5: RGB shortcut removal

Hold event features fixed and replace `rgb_feature` with a shuffled tensor before residual addition.

The raw event update produced by the event adapter must remain unchanged.

Only the final residual base should differ.

## Test 6: Patch-grid artifact check

Use a sparse synthetic event pattern.

Visualize per-scale update maps and verify there are no hard `28×37` binary boundaries caused by contribution thresholding.

## Test 7: Oracle geometry event

Evaluate:

```text
RGB only
RGB + full event
RGB + predicted contribution
RGB + oracle geometry event
```

If oracle geometry event does not improve normal metrics over RGB-only, print a warning because the event-refinement path is not yet validated.

---

# 13. Ablation switches

Preserve switches for the following variants:

```yaml
ablation:
  use_rgb_in_event_adapter: false
  use_hard_contribution_gate: false
  use_double_contribution_gate: false
  use_all_dpt_levels: false
  use_normal_refinement: true
  use_depth_residual_refinement: false
  use_attribution_supervision: true
  use_multildr_contribution_consistency: true
  use_depth_normal_consistency: true
  use_outside_support_regularization: true
```

The default configuration must correspond to the final proposed method.

---

# 14. Engineering constraints

Do not rewrite the full project.

Prefer modifying:

* `stage2_geometry_adapter/model.py`;
* contribution model files;
* current unified training script;
* loss utilities;
* configuration files;
* tests;
* visualization code.

Preserve:

* current dataset loading;
* exposure pair construction;
* VGGT checkpoint loading;
* distributed training support;
* mixed-precision support;
* existing output/checkpoint conventions.

Checkpoint loading must:

* clearly report missing/new keys;
* allow old checkpoints to initialize unchanged compatible modules;
* reject silently incompatible adapter shapes.

---

# 15. Execution process

Before modifying code:

1. Read the current model and training implementation.
2. List affected files and functions.
3. Describe current tensor shapes at every affected point.
4. Map every required change in this TODO to the actual function name and line region.
5. Identify any conflict with the current checkpoint format or DDP logic.

Then implement the changes.

After implementation:

1. Run unit tests.
2. Run one-batch forward/backward smoke tests for all three phases.
3. Run zero-event and zero-contribution tests.
4. Produce one debug visualization.
5. Print a concise implementation summary.
6. Provide exact training commands for Phase A, Phase B, and Phase C.

Do not stop after only changing model definitions. Update training, loss, config, checkpoint loading, tests, and visualization together so the final pipeline is executable end-to-end.

```

这份提示词的核心是把 Codex 的任务锁定在四个不可妥协的原则上：

\[
\boxed{
C\text{ 有独立归因监督};
\quad
Adapter\text{ 只能读取事件};
\quad
C\text{ 只软门控一次};
\quad
Event\text{ 主要修正法向并通过几何一致性约束深度}
}
\]

这样改完后，原先出现的“大洞、凹槽、法向网格纹”才有可能从结构上消失，而不是靠调小学习率暂时掩盖。
```
