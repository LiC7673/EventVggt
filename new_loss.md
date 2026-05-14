要证明 **“微分几何 cue 有效”**，不能只说模型加了 Sobel / Hessian / normal gradient 后指标变好。你要证明三件事：

1. **event pattern 和局部几何微分量确实相关。**
2. **把 event 转成 differential geometry cue，比 raw event feature 更有效。**
3. **提升集中发生在高曝光/饱和区域的几何细节上，而不是模型容量增加带来的。**

下面给你一个比较完整、但可执行的实验设计。

---

# 1. 先定义你要证明的 cue

你的 cue 不要太抽象，建议明确成：

```text
Event Differential Geometry Cue
```

输入包括：

```text
event_abs = E+ + E-
event_signed = E+ - E-
event_time_surface
event_orientation / local event direction
```

几何侧包括：

```text
∇ρ
Hρ = [ρ_xx, ρ_xy, ρ_yy]
∇N
Δρ
directional curvature κ_v = v^T Hρ v
```

其中：

```text
ρ = 1 / depth
```

你的核心假设是：

[
\frac{d}{dt}\log I
\approx
v^\top H_\rho q
]

也就是说，event 的正负和局部时序模式，应该与 depth Hessian / normal variation / curvature 有统计相关性。

---

# 2. 证明一：相关性实验

这是最重要的基础实验。

## 实验设置

用 synthetic 数据，因为你需要 GT depth / normal / curvature。

渲染：

```text
单色物体
固定材质
不同凹凸表面
不同运动方向
不同曝光强度
事件流
GT depth / normal
```

表面可以包括：

```text
sphere
bump
dent
ridge
groove
gear-like surface
sinusoidal surface
concave / convex paired shapes
```

然后计算 GT 微分量：

```text
GT ∇ρ
GT Δρ
GT Hρ
GT normal gradient
GT directional curvature κ_v
```

再计算 event cue：

```text
event_abs
event_signed
event_time_surface_gradient
event local orientation
```

## 指标

算相关性：

```text
Corr(event_abs, |∇N|)
Corr(event_abs, |Δρ|)
Corr(event_signed, sign(κ_v))
Corr(event orientation, geometry boundary orientation)
```

更具体一点：

```text
AUC: event_abs 是否能预测 high-curvature regions
Accuracy: event_signed 是否能区分 convex / concave sign
F1: event edge 与 geometry differential edge 的重合
```

你想得到的结论：

```text
event_abs 与几何细节区域高度相关；
event_signed 对凹凸/曲率符号有区分能力；
这种相关性在 RGB 过曝后仍然存在。
```

这一步不是训练模型，只是证明物理 cue 存在。

---

# 3. 证明二：可视化 event cue 与几何微分量对齐

这个实验很重要，因为 reviewer 很容易被图说服。

对每个样例展示：

```text
Overexposed RGB
Event abs
Event signed
GT normal map
GT depth Laplacian / curvature
Your differential cue map
```

最好放一组：

```text
convex bump
concave dent
```

你要让图上看出来：

```text
凸起和凹陷在 RGB 过曝后都看不清；
event signed pattern 不同；
GT curvature sign 也不同；
你的 cue 能捕捉这个差异。
```

这张图是你方法的核心动机图。

---

# 4. 证明三：raw event vs differential cue

这是主消融。

你要比较：

| 方法                      | 输入                                                        |
| ----------------------- | --------------------------------------------------------- |
| VGGT coarse             | RGB only                                                  |
| Raw event offset        | RGB + raw event voxel                                     |
| Event density offset    | RGB + E_abs                                               |
| Event signed offset     | RGB + E_abs + E_signed                                    |
| Event + Sobel depth     | RGB + event + ∇ρ                                          |
| Event + Hessian depth   | RGB + event + Hρ                                          |
| Event + normal gradient | RGB + event + Hρ + ∇N                                     |
| Full differential cue   | RGB + event differential + geometry differential residual |

核心要证明：

```text
raw event voxel 不稳定；
event density 只能补边界；
signed event + Hessian / normal gradient 才能恢复凹凸和细节；
full cue 最好。
```

这里你要注意控制参数量。为了避免 reviewer 说是参数更多导致的提升，可以加一个：

```text
Raw event + same-size MLP/Conv
```

保证模型容量相同。

---

# 5. 证明四：高曝光越强，cue 越有价值

这是你的任务核心。

设置曝光等级：

```text
EV 0
EV +2
EV +4
EV +6
```

对比：

```text
VGGT
VGGT + raw event
VGGT + differential cue
Ours full
```

指标分全图和饱和区域：

```text
Depth RMSE
Normal MAE
Curvature error
Saturated-region Normal MAE
Saturated-region Curvature error
Boundary F1
```

你希望结果是：

```text
EV 0：大家差距小
EV +4 / +6：RGB 和 raw event 明显退化，differential cue 保持优势
```

这能证明 cue 是为高曝光设计的，不是普通 event trick。

---

# 6. 证明五：几何细节指标，而不是只看 depth

只看 depth AbsRel 不够，因为你的 cue 主要恢复细节。

你应该报这些指标：

## Normal MAE

[
\text{NormalErr}= \arccos(N_{pred}\cdot N_{gt})
]

证明表面方向恢复更好。

## Curvature / Laplacian error

[
|\Delta \rho_{pred}-\Delta \rho_{gt}|_1
]

或者：

```text
Laplacian RMSE
Hessian L1
Curvature sign accuracy
```

## Boundary F1

从 depth / normal 提边界：

```text
B_pred = |∇ρ_pred| + |∇N_pred|
B_gt = |∇ρ_gt| + |∇N_gt|
```

计算 F1。

## Saturated-region metrics

只在：

```text
M_sat = RGB > 0.95 或 0.98
```

区域算：

```text
Sat-Normal MAE
Sat-Curvature Error
Sat-Boundary F1
```

这几项比普通 depth PSNR / AbsRel 更能证明你的 cue 有效。

---

# 7. 证明六：offset 是否真的由 cue 驱动

你要证明二阶段 decoder 不是在乱修。

做这些诊断：

## 1. offset 和 event cue 的相关性

计算：

```text
Corr(|Δρ|, event_abs)
Corr(|Δρ|, |event_geometry_residual|)
Corr(|Δρ|, reflect/highlight mask)
```

你希望：

```text
|Δρ| 与 event_geometry_residual 高相关；
与无事件区域低相关；
与反光区域不过度相关。
```

## 2. offset 是否集中在饱和细节区域

可视化：

```text
RGB overexposed
event cue
|Δρ|
GT curvature
error reduction map
```

你要展示：

```text
offset 出现在 event 支持且 GT 有几何细节的位置；
不是全图齿轮/波纹。
```

## 3. 关闭 cue 的 counterfactual

把 event cue 打乱：

```text
shuffle event across images
randomize event_signed
rotate event map
zero event_signed
```

结果应该明显下降。

如果 shuffle 后不变，说明模型没用 cue。

---

# 8. 证明七：凹凸可分性实验

这是非常适合你“内凹/外凸事件模式不同”的论点。

## 数据

构造成对样本：

```text
convex bump
concave dent
```

保证：

```text
RGB overexposed 后外观几乎一样
几何 GT 不同
事件 signed pattern 不同
```

## 任务

让模型预测：

```text
depth / normal / curvature sign
```

或者专门做一个辅助分类：

```text
convex vs concave
```

对比：

```text
RGB only
RGB + event_abs
RGB + event_signed
RGB + full differential cue
```

如果：

```text
event_abs 只能知道有结构
event_signed / time cue 能区分凹凸
```

这个结果非常有说服力。

---

# 9. 证明八：不要被光流/IWE baseline 打掉

因为 reviewer 可能问：

> 你为什么不用 event warping / IWE sharpness？

你可以加一个 baseline：

```text
IWE / event warping supervision
```

或：

```text
event flow consistency
```

然后展示在你的高曝光单色几何细节任务上：

```text
IWE 更擅长对齐运动边缘；
但对局部凹凸 / normal detail / curvature 恢复不如 differential cue。
```

指标上看：

```text
Normal MAE
Curvature error
Concave/convex accuracy
```

这正好证明你的 cue 不是随便设计的。

---

# 10. 主表应该长这样

| Method                     |  Depth ↓ | Normal ↓ |   Curv ↓ | Sat-Normal ↓ | Sat-Curv ↓ | Boundary F1 ↑ |
| -------------------------- | -------: | -------: | -------: | -----------: | ---------: | ------------: |
| VGGT                       |          |          |          |              |            |               |
| Raw Event Offset           |          |          |          |              |            |               |
| Event Density Cue          |          |          |          |              |            |               |
| Event Signed Cue           |          |          |          |              |            |               |
| Event + Sobel Depth        |          |          |          |              |            |               |
| Event + Hessian/Normal Cue |          |          |          |              |            |               |
| Full Ours                  | **best** | **best** | **best** |     **best** |   **best** |      **best** |

再加曝光曲线：

```text
EV +0 / +2 / +4 / +6
```

画：

```text
Sat-Normal Error vs Exposure
Sat-Curvature Error vs Exposure
```

你的线应该在高曝光下明显更稳。

---

# 11. 如果时间有限，最小证明集合

你至少做这 4 个实验：

## A. Synthetic correlation

证明 event cue 与 GT curvature / normal gradient 相关。

## B. Ablation

```text
raw event
event_abs
event_signed
event + Hessian/normal
full
```

## C. 高曝光曲线

证明曝光越强，cue 越有用。

## D. 凹凸成对实验

证明 signed event / time pattern 能区分 concave vs convex。

这四个足够支撑“微分几何 cue 有效”。

---

# 12. 你的论文结论可以这样写

> We verify that event patterns under motion are strongly correlated with local differential geometry, including normal variation and inverse-depth curvature. Compared with raw event fusion, explicitly providing event-derived differential cues and geometry differential residuals significantly improves saturated-region normal and curvature recovery.

中文：

> 我们验证了运动事件模式与局部微分几何结构之间存在稳定关联。相比直接融合原始事件流，将事件转化为微分几何提示，并与当前深度/法向的微分残差结合，能显著提升高曝光区域的法向和曲率恢复。

---

一句话：

**你要证明的不是“event 有用”，而是“event 的正负、时间和局部方向模式，能预测 RGB 过曝后丢失的深度/法向微分结构”。**
