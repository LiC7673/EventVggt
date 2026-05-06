
# 1. 正确结构：global + local 双分支

我建议你把模型改成：

```text
Global branch：负责多视角一致性和相机/粗几何
Local branch：负责 event-guided 几何细节
```

结构：

```text
RGB multi-view
   ↓
VGGT backbone
   ↓
coarse geometry tokens
   ↓
global consistency module
   ↓
D_coarse / pose / pointmap_coarse

Event stream
   ↓
high-res event encoder
   ↓
motion/reflection decomposition
   ↓
local detail tokens
   ↓
ΔD / Δpoint / normal detail

Final:
D_final = D_coarse + ΔD
P_final = P_coarse + ΔP
```

核心思想：

```text
global token 不直接预测全部几何；
global token 只给 coarse geometry；
event local branch 负责细节 residual。
```

---

# 3. 最推荐的模型改法

## 改法一：Global token 只做 coarse，不做 final

不要让：

```text
G_global → depth_final
```

而是：

```text
G_global → depth_coarse
event_local → depth_residual
depth_final = depth_coarse + depth_residual
```

也就是：

[
D = D_{coarse} + \Delta D_{event}
]

其中：

* (D_{coarse})：全局一致结构；
* (\Delta D_{event})：事件引导的局部细节。

这会显著改善细节。

---

## 改法二：事件分支保持高分辨率

event encoder 不要全部下采样到 H/16。

建议：

```text
E_high: H/2 或 H/4
E_mid:  H/8
E_low:  H/16
```

然后：

```text
E_low 参与 global token
E_high 参与 depth refinement
```

例如：

```text
E_low → global consistency
E_high → edge/detail refinement
```

---

## 改法三：用 cross-attention，而不是全局平均

不要：

```text
G = mean(all view tokens)
```

建议：

```text
G queries attend to all view tokens
local pixels attend to G and event tokens
```

也就是两步：

```text
1. G ← CrossAttn(G, multi-view tokens)
2. F_local ← CrossAttn(F_local, G)
3. F_local ← CrossAttn(F_local, event_local)
```

这样 global token 不是信息终点，而是一个中间记忆。

---

# 4. 更稳的结构：Global Memory + Local Decoder

你可以这样实现：

```text
F_i = VGGT image tokens for each view
E_i = event tokens for each view

G = GlobalMemory({F_i})

F_i' = F_i + CrossAttn(F_i, G)

E_i^m = motion_event_encoder(E_i)
F_i'' = F_i' + CrossAttn(F_i', E_i^m)

D_i = DepthHead(F_i'')
```

其中：

```text
G 提供跨视角一致性
E_i^m 提供局部几何细节
```

这比“G 直接预测 depth”稳定很多。

---

# 5. 你可以用 Perceiver-style latent，但要分层

如果你想继续用 global tokens，建议不要只用一组。

改成：

```text
Scene tokens:   负责全局场景
View tokens:    每个 view 一个 latent，负责视角局部
Patch tokens:   局部细节
Event tokens:   高分辨率事件结构
```

结构：

```text
Scene tokens ↔ View tokens ↔ Patch tokens
                         ↘ Event detail tokens
```

也就是：

```text
global token = scene memory
不是所有信息都塞进去
```

---

# 6. Loss 也要配合，否则 global 还是会糊

如果 loss 只有：

```text
depth L1
feature consistency
```

它天然鼓励平滑。

要加细节 loss：

## 6.1 Event-guided depth edge loss

[
L_{edge}
========

\left|
Norm(|\nabla D|)
----------------

Norm(E_m)
\right|_1
]

其中 (E_m) 是 motion-consistent event density。

注意：

```text
只用 motion event，不用反光 event。
```

---

## 6.2 Normal loss

[
L_{normal}
==========

1-\langle N_{pred},N_{gt}\rangle
]

如果没有 GT normal，可以用 teacher / pseudo normal。

---

## 6.3 Residual detail loss

如果你有正常曝光 teacher：

```text
D_teacher = VGGT(normal exposure)
D_student = model(overexposed + event)
```

不要只蒸馏 depth，而是蒸馏：

```text
∇D
normal
local curvature
edge map
```

loss：

[
L_{detail}
==========

|\nabla D_s-\nabla D_t|_1
+
\lambda(1-N_s\cdot N_t)
]

---

# 7. 一个很实用的训练策略

不要一开始训练 global token + event refinement 全部。

按阶段来：

## Stage 1：冻结 VGGT，只训 local event detail branch

```text
VGGT 给 D_coarse
event branch 学 ΔD
```

目标是先让 event 能补细节。

---

## Stage 2：加入 global token

```text
global token 学跨视角一致性
local event branch 保持细节
```

---

## Stage 3：微调后几层 VGGT

小学习率：

```text
VGGT last blocks lr = 1e-5 或 5e-6
event branch lr = 1e-4
```

这样不会把 VGGT 原来的几何能力破坏掉。

---

# 8. 如果 global token 现在已经很差，建议你做这个最小改版

你当前可能是：

```text
event + global token → 直接预测
```

改成：

```text
VGGT RGB → coarse depth / pointmap
event high-res encoder → Δdepth / Δpoint
global token → only consistency regularization
```

也就是：

```text
不要让 global token 主导预测；
让它只做 regularizer / memory。
```

具体：

```text
F_rgb = VGGTEncoder(rgb)
D_coarse = VGGTDepthHead(F_rgb)

G = GlobalTokenAggregator(F_rgb)
F_global = CrossAttn(F_rgb, G)

E_motion = EventHighResEncoder(events * motion_mask)

delta_D = DetailRefiner(F_global, E_motion, D_coarse)
D_final = D_coarse + delta_D
```

这是最容易救效果的版本。

---

# 9. 你应该做的诊断实验

为了确认 global token 哪里坏了，做这几个：

## 诊断一：只看 coarse vs residual

输出：

```text
D_coarse
ΔD_event
D_final
```

如果 ΔD_event 接近 0，说明 event 没起作用。

---

## 诊断二：token 数量消融

```text
4 global tokens
16 global tokens
64 global tokens
256 global tokens
```

如果 token 多了细节变好，说明之前是瓶颈。

如果 token 多了还差，说明结构/监督不对。

---

## 诊断三：local branch ablation

```text
global only
local event only
global + local
```

你大概率会看到：

```text
global only：稳定但糊
local only：细但不稳
global + local：最好
```

这就是你论文故事。

---

## 诊断四：event resolution ablation

```text
event feature H/16
event feature H/8
event feature H/4
event feature H/2
```

如果高分辨率 event 提升明显，说明细节瓶颈来自下采样。

---

# 10. 论文故事也要改一点

不要说：

> global token 直接恢复所有几何。

应该说：

> Global tokens enforce exposure-invariant multi-view consistency, while local motion-event tokens recover geometry details lost under overexposure.

中文：

> 全局 token 负责跨曝光多视角一致性，局部运动事件 token 负责恢复过曝区域丢失的几何细节。

这更符合你现在观察到的问题。

---

# 11. 最终建议

你现在最该做的是：

```text
不要废掉 global token；
把它降级为“全局一致性约束 / coarse geometry memory”；
再加一个 high-res motion event local detail branch。
```

一句话：

> **global token 保一致性，local event branch 保细节。**

如果你只用 global token，效果差是正常的；如果加上局部事件残差分支，才有可能把事件流的高频结构真正转成几何细节。
