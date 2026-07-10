结合前面的攻击点，我建议把第一阶段收缩成一个更稳、更容易实现的版本：

# 第一阶段：Multi-LDR-Guided Event Contribution Learning

它不再声称：

> 判断每个事件是不是由几何产生。

而是学习：

> 在当前高曝光 RGB 条件下，哪些事件能够实际改善几何预测。

可靠性图也最好改称 **event contribution map**：

[
C\in[0,1]^{B\times H\times W}.
]

这样可以避免“镜面事件是否也含有几何”“纹理事件是不是几何事件”等语义攻击。

---

## 一、Multi-LDR 到底提供什么

不要对两个曝光的深度、token 或 reliability 直接做一致性。

Multi-LDR 只用于找出一种特殊区域：

[
\boxed{
\text{在较好曝光中可见，但在高曝光中因饱和而丢失的区域}
}
]

假设同一个视角有：

[
I^1,I^2,I^5,I^{10},
]

并且共享：

[
V,\quad D^*,\quad N^*.
]

从中选择一个参考曝光 (I^r) 和一个高曝光 (I^b)。

参考曝光不一定固定为 `ev_1`，可以根据饱和比例选择：

[
r=\arg\min_e \operatorname{SatRatio}(I^e),
]

然后选择一个饱和程度明显更高的 (b)。

实际实现可以只使用三类有序 pair：

[
\mathrm{ev}_1\rightarrow\mathrm{ev}_5,
\qquad
\mathrm{ev}*1\rightarrow\mathrm{ev}*{10},
\qquad
\mathrm{ev}*2\rightarrow\mathrm{ev}*{10}.
]

不需要遍历全部六种组合。

---

# 二、构造真正有意义的 Bridge Mask

首先计算高曝光饱和区域：

[
M_{\mathrm{sat}}^b(\mathbf x)
=============================

\mathbb 1
\left[
\max_c I_c^b(\mathbf x)>\tau_{\mathrm{sat}}
\right].
]

参考曝光中，该区域不仅要没有饱和，还应保留可见图像结构：

[
M_{\mathrm{vis}}^r(\mathbf x)
=============================

\mathbb 1
\left[
\max_c I_c^r(\mathbf x)<\tau_{\mathrm{sat}}
\right]
\cdot
\mathbb 1
\left[
|\nabla I^r(\mathbf x)|>\tau_g
\right].
]

事件支持区域：

[
M_E(\mathbf x)
==============

\mathbb 1
\left[
\sum_b|V_b(\mathbf x)|>0
\right].
]

最终：

[
M_{\mathrm{bridge}}
===================

M_{\mathrm{sat}}^b
\odot
M_{\mathrm{vis}}^r
\odot
M_E.
]

它表示：

> 同一结构在参考曝光中仍然可见，在高曝光中已经丢失，并且该位置存在事件响应。

如果：

[
\frac{\sum M_{\mathrm{bridge}}}{HW}<\tau_{\mathrm{area}},
]

这个曝光对就直接跳过。

这也解决了你之前 `ev_2` 与 `ev_5` 几乎相同的问题：不是检查全图平均 RGB 差，而是检查有没有足够大的有效 bridge 区域。

---

# 三、网络结构

RGB-VGGT 在第一阶段冻结：

[
G_c^b
=====

F_{\mathrm{RGB}}(I^b),
]

其中可以包括：

[
G_c^b={D_c^b,N_c^b,F_c^b}.
]

贡献网络输入：

[
C
=

\mathcal C_\theta
\left(
V,\ I^b,\ \operatorname{sg}(G_c^b)
\right).
]

然后得到加权事件：

[
V_{\mathrm{sel}}=C\odot V.
]

事件细化器只允许看到加权后的事件：

[
\Delta G
========

F_{\mathrm{event}}
\left(
V_{\mathrm{sel}},\operatorname{sg}(G_c^b)
\right).
]

最终：

[
\widehat D^b
============

D_c^b\exp(\Delta\log D),
]

[
\widehat N^b
============

\operatorname{Normalize}
\left(
N_c^b+\Delta N
\right).
]

这里有一个非常关键的结构要求：

[
\boxed{
\text{完整事件 }V\text{ 不能绕过贡献图直接进入几何解码器}
}
]

否则后面的 Event Refiner 可以忽略 (C)，第一阶段学不到真正有意义的筛选。

---

# 四、怎么利用你的理论推导

你的推导可以保留：

[
\Delta L
\approx
\nabla L^\top\mathbf u\Delta t.
]

对于局部平滑材质，亮度变化的一部分与表面法向变化有关。因此，法向变化和深度边界区域中的事件更可能携带局部几何结构。

但是不要把：

[
|\nabla N^*|
]

直接当 reliability label。

最稳妥的做法是把它变成**几何损失权重**：

[
W_{\mathrm{geo}}
================

1+
\alpha
\operatorname{Norm}
\left(
|\nabla N^*|
+
\lambda_D
|\nabla\log D^*|
\right).
]

这样：

* 法向变化和深度边界区域受到更强监督；
* 平面纹理区域不会被标成负样本；
* 镜面事件也不会被理论上直接判为无用；
* 推导真正成为归纳偏置，而不是错误的硬标签。

这句话可以概括为：

[
\boxed{
\text{物理推导决定重点监督哪里，几何任务决定具体保留哪些事件。}
}
]

---

# 五、第一阶段核心损失

只在 (M_{\mathrm{bridge}}) 内监督高曝光分支。

## 深度损失

[
\mathcal L_D
============

\frac{
\sum_{\mathbf x}
M_{\mathrm{bridge}}(\mathbf x)
W_{\mathrm{geo}}(\mathbf x)
\left|
\log\widehat D^b(\mathbf x)
---------------------------

\log D^*(\mathbf x)
\right|
}{
\sum_{\mathbf x}M_{\mathrm{bridge}}(\mathbf x)+\epsilon
}.
]

## 法向损失

[
\mathcal L_N
============

\frac{
\sum_{\mathbf x}
M_{\mathrm{bridge}}(\mathbf x)
W_{\mathrm{geo}}(\mathbf x)
\left[
1-
\widehat N^b(\mathbf x)^\top N^*(\mathbf x)
\right]
}{
\sum_{\mathbf x}M_{\mathrm{bridge}}(\mathbf x)+\epsilon
}.
]

这两个损失直接回答：

> 经过贡献图选出的事件，能否恢复高曝光丢失的几何？

---

# 六、如何防止 (C\equiv1)

这是第一阶段必须解决的退化问题。

最简单的方案不是设计复杂的 source label，而是加入一个固定的软保留预算：

[
\bar C
======

\frac{
\sum |V|\odot C
}{
\sum |V|+\epsilon
}.
]

然后：

[
\mathcal L_{\mathrm{budget}}
============================

|\bar C-\rho|.
]

例如：

[
\rho=0.5.
]

这意味着模型平均只能重点使用约一半的有效事件。

它不能通过：

[
C\equiv1
]

直接保留全部事件，也不能通过：

[
C\equiv0
]

完全关闭事件分支。

最终第一阶段损失非常简单：

[
\boxed{
\mathcal L_{\mathrm{stage1}}
============================

\mathcal L_D
+
\lambda_N\mathcal L_N
+
\lambda_B\mathcal L_{\mathrm{budget}}
}
]

不要再加入 token loss、可靠性一致性、事件分解损失或光流 warp。

如果担心固定 (\rho) 太武断，可以训练使用 (\rho=0.5)，然后只在推理阶段测试 (0.3/0.5/0.7) 的敏感性，不需要重新训练三次。

---

# 七、为什么这里仍然必须使用 Multi-LDR

Reviewer 可能问：

> 有 GT 深度，为什么不能直接在高曝光图像上训练？

你的回答是：

单独使用高曝光图像，只能知道哪些地方预测错误；它无法区分：

* RGB 本身就没有结构的区域；
* 高曝光造成结构丢失的区域；
* 没有事件支持的区域。

配对曝光提供了受控变量：

[
\text{场景、视角、几何、事件不变，只有 RGB 可见性改变。}
]

因此：

[
M_{\mathrm{bridge}}
]

明确定位了：

> 结构在另一个曝光中确实存在，但在当前高曝光中被饱和抹除，并且有事件可以补充。

所以 Multi-LDR 不是用来重复 GT，也不是普通一致性，而是用来构造**因曝光变化而失去视觉证据的训练区域**。

这个表述比“不同曝光预测应一致”更难被攻击。

---

# 八、第一阶段完整伪代码

```text
Stage 1: Multi-LDR-Guided Event Contribution Learning

Input:
    Multi-LDR images {Iy,Ix}，(x \in our exists sets 2>=x>=0,y>x)
    Shared event volume V
    GT depth D*
    GT normal N*

For each scene window:

    1. Select an ordered exposure pair:

           I_ref = an exposure with low saturation
           I_bad = an exposure with clearly higher saturation

    2. Construct masks:

           M_sat_bad =
               saturated pixels in I_bad

           M_visible_ref =
               non-saturated pixels in I_ref
               with visible image gradients

           M_event =
               pixels containing events

           M_bridge =
               M_sat_bad
               AND M_visible_ref
               AND M_event

    3. Skip the pair if M_bridge is too small.

    4. Freeze the RGB geometry backbone:

           G_coarse =
               StopGrad(RGB_Backbone(I_bad))

    5. Predict event contribution:

           C =
               ContributionNet(
                   V,
                   I_bad,
                   G_coarse
               )

    6. Select events:

           V_selected = C * V

    7. Predict geometry residual:

           Delta_G =
               EventRefiner(
                   V_selected,
                   G_coarse
               )

           G_pred =
               Refine(G_coarse, Delta_G)

    8. Construct geometry-aware weights:

           W_geo =
               1
               + alpha * normal-gradient magnitude
               + beta  * depth-boundary magnitude

    9. Compute losses inside M_bridge:

           L_depth
           L_normal
           L_budget

   10. Update only:

           ContributionNet
           EventRefiner

Output:
    Pretrained ContributionNet
    Pretrained EventRefiner
```

---

# 九、第二阶段怎么接

第一阶段训练完成后：

* 加载 ContributionNet；
* 加载 Event Refiner；
* 解冻部分 VGGT；
* 使用所有曝光进行正常几何训练。

第二阶段输入单个曝光和事件：

[
(I^e,V)
\rightarrow
C^e
\rightarrow
C^e\odot V
\rightarrow
\widehat G^e.
]

使用完整 GT 几何损失：

[
\mathcal L_{\mathrm{stage2}}
============================

\mathcal L_{\mathrm{depth}}
+
\lambda_N\mathcal L_{\mathrm{normal}}
+
\lambda_P\mathcal L_{\mathrm{pose}}.
]

第一阶段的 bridge loss 可以保留一个很低的权重，也可以只把第一阶段作为初始化。

---

# 十、从你当前脚本怎么改

当前脚本本质上是：

```text
加载 paired-token teacher
→ 提取两个曝光 token
→ 计算 token agreement
→ 导出 reliability npz
```

修改后应变成训练脚本，而不是标签导出脚本。原有的 token teacher、`token_a/token_b`、cosine similarity、`token_agreement` 和离线 target 导出全部删除。当前脚本中的 paired exposure 读取、共享事件检查、场景划分和 preview 逻辑仍然可以保留。

新脚本的逻辑是：

```text
读取 Multi-LDR pair
→ 构造 bridge mask
→ 冻结 RGB backbone
→ ContributionNet 预测 C
→ C * event
→ Event Refiner 恢复几何
→ bridge-region GT loss
→ 反向更新
```

---

# 十一、最低限度必须做的实验

真正需要重新训练的只有四个配置：

| 配置                                   | 目的                      |
| ------------------------------------ | ----------------------- |
| RGB-VGGT                             | 基础结果                    |
| RGB + full events                    | 证明直接融合的效果               |
| ContributionNet，不使用 Multi-LDR bridge | 证明筛选结构本身                |
| 完整方法                                 | 证明 Multi-LDR bridge 的价值 |

另外只用最终 checkpoint 做三个便宜测试：

* (C=1)：全部事件；
* 随机 (C)：相同平均保留率；
* 删除最高分事件与删除最低分事件。

最后一个测试尤其重要：

> 删除低贡献事件，性能应基本不变或改善；删除高贡献事件，性能应明显下降。

这能够最直接证明你学到的是“对几何有贡献的事件”，而不是一张看起来合理的热力图。

最终第一阶段应该保持得非常克制：

[
\boxed{
\text{Multi-LDR 找到曝光丢失区域}
}
]

[
\boxed{
\text{几何误差监督事件贡献}
}
]

[
\boxed{
\text{预算约束防止全部事件通过}
}
]

这三件事已经足够形成一个完整、可实现，并且相对抗审稿攻击的第一阶段。
