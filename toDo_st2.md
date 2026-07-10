对，直接预测 raw (\Delta D) 不适合你这个任务。它会把“事件提供局部结构”强行转化成“事件直接修改深度数值”，而事件本身并不包含稳定的绝对尺度信息。尤其 VGGT 的深度来自多视角 token、相机关系和预训练几何先验，单独的事件残差头很容易破坏这些结构。

所以 Stage 2 应该从：

[
D_f=D_c+\Delta D_{\mathrm{event}}
]

改成：

[
\boxed{
\text{可靠事件细化 VGGT 的几何表征，由原有几何头重新解码}
}
]

也就是**feature/token-level refinement，而不是 depth-level refinement**。

# Stage 2：Reliability-Guided Geometry Representation Refinement

Stage 1 已经学习得到事件贡献图：

[
C=\mathcal C_\theta(I,V,G_c),
]

它回答：

> 哪些事件能够补充高曝光 RGB 中丢失的几何证据。

Stage 2 的任务不是让这些事件直接回归深度，而是将筛选后的事件转化成局部几何特征，用来修复被饱和 RGB 污染的 VGGT dense geometry features。

整体流程改成：

[
I
\rightarrow
\text{VGGT backbone}
\rightarrow
F_{\mathrm{rgb}}^1,\ldots,F_{\mathrm{rgb}}^L
]

同时：

[
V_{\mathrm{sel}}=C\odot V
\rightarrow
\text{Event Encoder}
\rightarrow
F_{\mathrm{event}}^1,\ldots,F_{\mathrm{event}}^L.
]

然后在 VGGT 的 DPT/geometry decoder 中进行多尺度特征细化：

[
\widetilde F_{\mathrm{geo}}^l
=============================

F_{\mathrm{rgb}}^l
+
\Delta F_{\mathrm{event}}^l.
]

最后仍然由 VGGT 原来的几何头输出：

[
D_f,N_f,P_f
===========

\mathcal H_{\mathrm{VGGT}}
\left(
\widetilde F_{\mathrm{geo}}^1,\ldots,
\widetilde F_{\mathrm{geo}}^L
\right).
]

这里不存在显式的：

[
D_c+\Delta D.
]

模型是在几何特征空间中恢复结构，最终深度由原来的预训练几何头统一解码。

---

# 为什么 feature refinement 更合理

VGGT 的中间表征已经编码了：

* 多视角对应；
* 全局场景尺度；
* 相机关系；
* 局部表面结构；
* 预训练获得的几何先验。

事件流更擅长提供：

* 饱和区域中仍然存在的边缘变化；
* 局部高频结构；
* 时序边界；
* RGB 中缺失的对比度证据。

所以两者的合理分工是：

[
F_{\mathrm{rgb}}
================

\text{全局几何基础},
]

[
F_{\mathrm{event}}
==================

\text{局部结构补充}.
]

而不是：

[
D_c=\text{粗深度},
\qquad
\Delta D_{\mathrm{event}}=\text{事件直接修改深度}.
]

后者要求事件分支自己理解绝对尺度和多视角几何，任务太重，效果差是正常的。

---

# 推荐的具体融合形式

## 方案一：多尺度 Geometry Adapter

这是我最建议你先实现的版本。

对于 VGGT/DPT 解码器第 (l) 层：

[
E^l
===

\mathcal E_l(C\odot V),
]

然后将事件特征和 RGB 几何特征拼接：

[
A^l
===

\operatorname{Adapter}*l
\left(
[F*{\mathrm{rgb}}^l,E^l]
\right).
]

得到：

[
\widetilde F_{\mathrm{geo}}^l
=============================

F_{\mathrm{rgb}}^l
+
\alpha_l A^l.
]

其中：

[
\alpha_l=\tanh(a_l)
]

或者设为一个可学习的较小系数，并将 (a_l) 初始化为 0。

初始化时：

[
\alpha_l\approx0,
]

模型等价于原始 VGGT。训练过程中才逐渐引入事件修正，避免一开始就破坏预训练表示。

实际结构可以是：

```text
VGGT dense feature F_rgb_l
             │
             ├───────────────┐
             │               │
Selected events        Event feature E_l
C × V                         │
             │               │
             └──── concat ───┘
                        │
                 Geometry Adapter
                        │
                     ΔF_l
                        │
                 F_rgb_l + α_l ΔF_l
                        │
                 Original VGGT head
                        │
               Depth / Normal / Point map
```

---

## 方案二：Event-to-Geometry Cross-Attention

如果你想让方法看起来更完整，可以使用：

[
Q=F_{\mathrm{rgb}}^l,
\qquad
K=E^l,
\qquad
V=E^l.
]

然后：

[
\Delta F^l
==========

\operatorname{CrossAttn}
\left(
F_{\mathrm{rgb}}^l,E^l,E^l
\right),
]

[
\widetilde F^l
==============

F_{\mathrm{rgb}}^l
+
\alpha_l\Delta F^l.
]

其含义是：

> RGB 几何 token 主动查询筛选事件中缺失的局部结构。

但这比卷积 Adapter 更贵，而且不一定更有效。你目前优先实现多尺度 Adapter 即可，没必要为了论文复杂度强行使用 attention。

---

# Contribution map 在 Stage 2 中怎么使用

贡献图不能只在输入处乘一次，然后后续完全消失。建议同时在事件体和多尺度特征上使用：

[
V_{\mathrm{sel}}=C\odot V,
]

并将 (C) 下采样到第 (l) 个尺度：

[
C^l=\operatorname{Downsample}(C).
]

事件更新变为：

[
\Delta F^l
==========

C^l\odot
\operatorname{Adapter}*l
\left(
[F*{\mathrm{rgb}}^l,E^l]
\right).
]

最终：

[
\widetilde F^l
==============

F_{\mathrm{rgb}}^l+\alpha_l\Delta F^l.
]

这样：

* (C) 高的位置，事件允许显著修复几何特征；
* (C) 低的位置，保持 VGGT 的原始表示；
* 事件不会直接修改整张深度图。

如果 (C) 是时空形式 (B\times H\times W)，进入 Event Encoder 前按 bin 使用；进入图像特征层时，再聚合为二维空间贡献图：

[
\bar C(\mathbf x)
=================

\frac{
\sum_b C_b(\mathbf x)|V_b(\mathbf x)|
}{
\sum_b|V_b(\mathbf x)|+\epsilon
}.
]

---

# 最好在哪一层融合

不建议直接修改 VGGT 最前面的 image patch tokens。因为最浅层更多是颜色和纹理表示，事件注入可能导致预训练特征分布发生较大变化。

也不建议只在最终 depth map 前融合，因为这又接近 raw depth residual。

最合适的是：

[
\boxed{
\text{VGGT aggregator 之后、dense geometry head/DPT decoder 内部}
}
]

也就是：

* VGGT backbone 和 aggregator 保留全局多视角建模；
* event branch 在多尺度 dense geometry features 上补充局部结构；
* 原始 depth/point/normal head 完成几何解码；
* camera head 尽量保持 RGB 主导。

如果 VGGT 的 DPT head 使用四层中间 token，可以在这四个尺度分别加入 event adapter。

---

# 位姿怎么处理

Stage 2 初版不要让事件直接修改 pose。

保持：

[
P_f=P_{\mathrm{VGGT}}.
]

事件主要细化：

* depth；
* point map；
* normal。

理由是可靠事件主要提供局部结构，直接回归相机位姿会引入新的不稳定因素。等深度和法向效果稳定以后，再考虑是否将事件特征注入共享 aggregator。

论文中可以说：

> We preserve the camera prediction of the RGB backbone and restrict event refinement to dense geometry representations.

这样方法范围更清楚，也更容易实验。

---

# Stage 2 的损失

由于最终仍然输出完整几何，使用 VGGT 原有任务损失即可：

[
\mathcal L_{\mathrm{geo}}
=========================

\mathcal L_{\mathrm{depth}}
+
\lambda_N\mathcal L_{\mathrm{normal}}
+
\lambda_P\mathcal L_{\mathrm{point}}.
]

高曝光区域提高权重：

[
W_{\mathrm{sat}}
================

1+\gamma M_{\mathrm{sat}}.
]

另外加入一个很轻的特征更新约束：

[
\mathcal L_{\mathrm{update}}
============================

\sum_l
\left|
(1-C^l)\odot\Delta F^l
\right|_1.
]

它约束：

> 在贡献较低的位置，事件 Adapter 不应大幅改变 RGB 几何特征。

最终：

[
\mathcal L_{\mathrm{stage2}}
============================

\mathcal L_{\mathrm{geo}}
+
\lambda_{\mathrm{update}}
\mathcal L_{\mathrm{update}}.
]

不需要再加入 depth residual magnitude loss，因为已经不预测 raw (\Delta D)。

---

# 训练流程

## Stage 2-A：只训练事件细化模块

冻结：

* VGGT backbone；
* VGGT aggregator；
* 原始几何头；
* Stage 1 ContributionNet。

训练：

* Event Encoder；
* 多尺度 Geometry Adapters。

此时只有事件 Adapter 能够改善几何，能够确认它确实学到了有效补充。

## Stage 2-B：低学习率联合微调

随后解冻：

* DPT geometry head；
* VGGT 最后少数几个 block；
* ContributionNet 可以使用很低的学习率。

学习率建议满足：

[
\eta_{\mathrm{VGGT}}
<
\eta_{\mathrm{Contribution}}
<
\eta_{\mathrm{EventAdapter}}.
]

避免 VGGT 直接适应合成数据，而事件模块没有发挥作用。

---

# Stage 2 伪代码

```text
Stage 2: Contribution-Guided Geometry Feature Refinement

Input:
    Single-exposure LDR multi-view images I
    Event volume V
    Pretrained ContributionNet

1. Extract RGB multi-view geometry features:

       Z, {F_rgb_l}, P
           = VGGT_Backbone_And_Aggregator(I)

2. Estimate event contribution:

       C =
           ContributionNet(
               I,
               V,
               StopGrad({F_rgb_l})
           )

3. Select reliable events:

       V_selected = C * V

4. Encode selected events into a multi-scale pyramid:

       {E_l} = EventEncoder(V_selected)

5. For each geometry-decoder scale l:

       C_l = DownsampleAndAggregate(C)

       Delta_F_l =
           C_l *
           GeometryAdapter_l(
               F_rgb_l,
               E_l
           )

       F_refined_l =
           F_rgb_l
           + alpha_l * Delta_F_l

6. Decode geometry using the original VGGT heads:

       D_final, N_final, PointMap_final
           = VGGT_GeometryHead(
               {F_refined_l}
           )

       Pose_final = VGGT_CameraHead(Z)

7. Optimize:

       L_stage2 =
           L_depth
         + lambda_normal * L_normal
         + lambda_point * L_point
         + lambda_update * L_update
```

---

# 这样两阶段故事更完整

## Stage 1

利用 Multi-LDR 的可见性变化学习：

[
C=
\text{event contribution}.
]

它解决：

> 哪些事件能够补充被曝光抹除的几何证据？

## Stage 2

利用 (C) 筛选事件，在 VGGT 的几何特征空间中进行局部修复：

[
F_{\mathrm{rgb}}
\rightarrow
F_{\mathrm{rgb}}+\Delta F_{\mathrm{event}}.
]

它解决：

> 如何在不破坏预训练全局几何的前提下，将可靠事件转化为最终几何提升？

最终不再说：

> events predict a depth residual。

而应该说：

> Reliable event responses refine the intermediate dense geometry representations of the RGB backbone, while the pretrained geometry heads preserve global multi-view structure and decode the refined features into final geometry.

我认为这比 raw (\Delta D) 合理得多，也更符合 VGGT 这种大型预训练几何模型的使用方式。
