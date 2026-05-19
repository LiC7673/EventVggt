可以。你的不同 LDR 等级图片不要简单当成普通数据增强扔进去，而应该设计成 **同一场景/同一视角/同一几何的多曝光组**。核心是：

> **几何不随曝光变化，RGB feature 会随曝光退化，event cue 近似不随 LDR gain 变化。**

所以训练里要显式利用这个“不变量”。

---

## 1. 一个训练 sample 应该长什么样

假设一个 scene 有 (V) 个视角，每个视角有多个曝光等级：

```text
scene_id
  view_1:
    I_ev0, I_ev2, I_ev4, I_ev6
    event_1
    depth_gt / normal_gt / teacher
  view_2:
    I_ev0, I_ev2, I_ev4, I_ev6
    event_2
    depth_gt / normal_gt / teacher
  ...
```

训练时一个 batch 不是随机抽单张图，而是抽：

```text
同一个 scene
同一组 views
随机选 2~3 个曝光等级
```

例如：

```text
views = [1,2,3,4]
exposures = [EV0, EV4]
```

输入给模型：

```text
EV4 多视角 RGB + event
EV0 多视角 RGB + event
```

然后让它们输出的几何保持一致。

---

## 2. 最推荐的 batch 组织方式

每次训练构造两个分支：

```text
branch A: normal / mild exposure
branch B: overexposed / strong exposure
```

例如：

```python
I_a = clip(gain_a * I)   # gain_a = 1 or 2
I_b = clip(gain_b * I)   # gain_b = 4 or 8
E   = same event
```

模型共享权重：

```text
pred_a = model(I_a, event)
pred_b = model(I_b, event)
```

然后监督：

```text
pred_a 对 GT / teacher 学
pred_b 对 GT / teacher 学
pred_a 和 pred_b 之间做曝光一致性
```

这样模型会学到：

```text
曝光变了，几何不该变
```

---

## 3. 训练输入建议

对于每个 view (i)，输入通道可以是：

```text
RGB_LDR_i
event_abs_i
event_signed_i
event_time_surface_i
saturation_mask_i
```

如果你不想一开始改 VGGT 主干，可以这样：

```text
VGGT 只吃 RGB_LDR
event/saturation 只进 refinement head
```

也就是：

```text
RGB_LDR → VGGT → coarse depth / feature
event + saturation + coarse geometry → detail refinement
```

这比把 event 直接拼到 VGGT 输入更稳。

---

## 4. 关键 loss 设计

### 4.1 几何监督 loss

每个曝光等级都应该对同一个 GT / teacher：

[
L_{\text{geo}} =
|\rho^{pred}_{ev}-\rho^{gt}|*1
+
\lambda_N(1-N^{pred}*{ev}\cdot N^{gt})
]

如果没有真实 GT，就用正常曝光 teacher：

```text
EV0 / normal RGB → teacher geometry
EV4 / EV6 + event → student geometry
```

[
L_{\text{teacher}} =
|\rho^{pred}*{over}-\text{sg}(\rho^{teacher}*{normal})|_1
]

---

### 4.2 曝光一致性 loss

同一个 scene、同一 view，不同曝光输出应该一致：

[
L_{\text{exp}} =
|\rho^{pred}*{a}-\rho^{pred}*{b}|*1
+
\lambda_N(1-N^{pred}*{a}\cdot N^{pred}_{b})
]

但建议重点加权在饱和区域：

[
W = M_{sat}^{a} \cup M_{sat}^{b}
]

[
L_{\text{exp}} =
W\left(
|\rho_a-\rho_b|_1
+
\lambda_N(1-N_a\cdot N_b)
\right)
]

这条 loss 是你的高曝光核心。

---

### 4.3 event-supported detail loss

event 不当标签，只当权重：

[
R = M_{sat} \cdot S_{event}
]

[
L_{\text{detail}} =
R \left(
|\nabla N^{pred}-\nabla N^{gt}|*1
+
\lambda*\rho|\nabla \rho^{pred}-\nabla \rho^{gt}|_1
\right)
]

意思是：

```text
event 告诉模型哪里应该恢复细节
GT / teacher 告诉模型细节长什么样
```

---

### 4.4 多视角一致性 loss

同一个 3D 点在不同 view 下几何要一致：

[
L_{\text{mv}} =
W_{ij}
|X_i(u)-X_j(\pi_j(X_i(u)))|_1
]

其中：

[
W_{ij}=1+\alpha M_{sat}(u)S_{event}(u)
]

也就是普通区域也做多视角一致，event-supported saturated 区域权重大。

---

## 5. 不同 LDR 等级怎么采样

推荐三阶段采样。

### Stage 1：先训正常和轻度过曝

```text
gain = [1, 2]
```

目标：模型先学稳定，不要一开始就面对全白图。

---

### Stage 2：加入强过曝

```text
gain = [1, 2, 4]
```

训练曝光一致性：

```text
EV1 vs EV4
EV2 vs EV4
```

---

### Stage 3：加入极端过曝

```text
gain = [1, 2, 4, 8]
```

这时再加 event detail / saturated-region loss。

否则一开始 gain=8 太强，模型可能直接学坏。

---

## 6. 一个 batch 的具体形式

例如 batch size 按 scene 来算：

```python
scene = sample_scene()
views = sample_views(scene, num_views=4)

gain_a = random.choice([1.0, 2.0])
gain_b = random.choice([4.0, 8.0])

I_a = clip(gain_a * I_clean)
I_b = clip(gain_b * I_clean)

E = event
M_sat_a = I_a > 0.95
M_sat_b = I_b > 0.95
```

forward：

```python
pred_a = model(I_a, E, M_sat_a)
pred_b = model(I_b, E, M_sat_b)
```

loss：

```python
loss_geo = geo_loss(pred_a, gt) + geo_loss(pred_b, gt)

loss_exp = exp_consistency(pred_a, pred_b, M_sat_a | M_sat_b)

loss_detail = event_weighted_detail_loss(
    pred_b, gt, event, M_sat_b
)

loss_mv = multiview_consistency(pred_b, cameras, event, M_sat_b)

loss_reg = offset_regularization(pred_b.delta, event, M_sat_b)

loss = (
    loss_geo
    + 0.5 * loss_exp
    + 0.5 * loss_detail
    + 0.2 * loss_mv
    + 0.3 * loss_reg
)
```

---

## 7. 你要避免的训练方式

不要这样：

```text
把 EV0、EV2、EV4、EV8 当作普通独立图片随机打散
```

因为这样模型不知道：

```text
这些图片其实是同一个几何
```

也就学不到曝光不变性。

也不要这样：

```text
只用 EV8 训练
```

因为 EV8 太退化，模型会学到过度平滑或依赖 event 噪声。

---

## 8. 最适合你论文的方法名

你可以把这个训练策略叫：

```text
Exposure-Grouped Multi-view Training
```

或者更贴切：

```text
Exposure-Consistent Event-Guided Geometry Training
```

核心贡献可以写成：

> For each scene, we jointly train on multiple LDR exposure levels and impose geometry consistency across exposure variants. Events are used only as saturation-conditioned detail support, guiding where high-frequency geometry losses and bounded refinement are applied.

---

## 9. 最终建议

你的不同 LDR 等级图片应该按这个逻辑进入训练：

```text
同一 scene / 同一 view / 不同曝光 → 成对输入
event 共享或对齐
GT/teacher 共享
输出几何做曝光一致性
强过曝区域用 event 加权细节恢复
```

一句话：

> **不要把 LDR 等级当普通增强；要把它当“同一几何的不同退化观测”，用 exposure consistency 把它们绑在一起。**
