# 组会汇报：反光/过曝光场景下的事件辅助精细几何恢复

## 一句话总结

本阶段已经从“事件能否提供有效线索”的验证，推进到“利用事件时序可靠性约束局部深度细化”的模型实现：在强反光、LDR 过曝光导致 RGB 细节缺失的区域，事件信号能够作为局部几何细节恢复的补充证据。

## 当前主线

研究问题：

> 在高反射和过曝光条件下，RGB 图像中的纹理与边缘容易丢失；能否利用事件流保留的时序变化信息，恢复更清晰、更稳定的细粒度深度和法向结构？

目前形成的技术路线：

```text
多视角 LDR RGB -> VGGT / StreamVGGT -> coarse depth / pose / point map
事件 voxel bins -> temporal reliability gate
GT detail supervision + multi-LDR consistency -> local depth residual
final depth = coarse depth + event-conditioned detail correction
```

核心观点：

- RGB 主干负责全局结构与多视角几何稳定性。
- GT depth 导出的法向/高频细节用于明确监督精细结构。
- 事件流不直接充当深度标签，而用于判断哪些局部修正可靠、哪些可能是高光拖影。
- 多 LDR 配对训练用于学习不同曝光下仍然稳定的局部几何修正。

## 已完成工作

### 1. 数据链路与事件可视化检查

- 检查并修正事件流的时间区间、坐标变换和 LDR 对齐流程。
- 将事件下采样改为 `voxel grid + antialias resize` 的思路，降低直接坐标缩放带来的块状伪影。
- 已能够按 view 可视化事件 bins，并进行法向误差和事件区域的对应分析。

### 2. 事件线索有效性验证

修正数据加载后，事件与几何细节之间出现了稳定的正相关信号：

| 指标 | 结果 | 说明 |
| --- | ---: | --- |
| `corr(event_abs, geom_detail)` | `0.2503` | 事件强度与几何细节有正相关 |
| `AUC(event_abs -> high_geom_detail)` | `0.6334` | 事件可以区分较高细节区域 |
| Shuffle AUC | `0.4987` | 打乱后接近随机，说明相关性不是偶然位置偏差 |
| `orientation_alignment` | `0.7401` | 事件方向与几何边界方向存在对齐信息 |

汇报用结论：

> 事件不是逐像素深度真值，但在 RGB 容易失真的反光区域，确实保留了可用于细节恢复的局部结构提示。

### 3. 多类细节监督与消融框架

已经搭建了完整的 `mul_loss_fine` 实验框架，包含：

- Baseline 微调。
- GT depth / normal 导出的高频细节监督。
- Event-supported 多视角 normal / high-frequency / presence 约束。
- Temporal-bin 事件输入实验。
- Pixel-domain residual 与 gated residual 模型。
- 多 LDR 配对训练与曝光不变性实验。

其中 `finetune_mul_loss_detail_gt.py` / `detail_gt_uniform` 证明：针对法向和高频结构的监督能够显著改善视觉细节，是当前最稳定的强基线。

### 4. 事件参与的 gated detail 模型

为避免事件拖影直接写入深度，已经将模型改为：

```text
RGB + coarse depth -> residual proposal
event temporal evidence -> confidence gate
proposal * gate -> bounded log-depth residual
```

这比“直接将事件特征写入深度”更稳健：事件只决定修正是否可信，RGB/粗几何决定修正形状。

一组已有指标显示 gated refinement 能改善法向细节：

| 指标 | Coarse | Refined | 改善 |
| --- | ---: | ---: | ---: |
| Depth-derived normal loss | `0.148090` | `0.143565` | `0.004525`，约 `3.1%` |

法向可视化上，细节边界比粗预测更清晰，说明局部修正方向是有效的。

### 5. 代表性细节误差改善

在法向误差分析中，已有模型相对于 baseline 在代表性 view 上获得了明显下降：

| View | Baseline normal error | Detail model normal error | 降低 |
| --- | ---: | ---: | ---: |
| View 2 | `31.12 deg` | `27.71 deg` | `3.41 deg` |
| View 3 | `31.58 deg` | `28.75 deg` | `2.83 deg` |

同时，高事件区域的误差差距提示：困难细节区域正是后续事件可靠性建模应重点优化的位置。

## 最新方法：Temporal Reliability V2

### 动机

已有 gated 方案能利用事件，但简单事件活动量不足以区分：

- 真正对应物体几何边界的瞬时事件。
- 高光随运动扫过表面产生的持续性事件轨迹。

因此，新版本不只看“有没有事件”，而是显式分析事件在时间维度上的行为。

### 新机制

新增模型：

- `eventvggt/models/streamvggt_temporal_reliability_v2.py`
- `mul_loss_fine/finetune_mul_loss_detail_gt_temporal_reliability_v2.py`

事件可靠性分支使用：

| 特征 | 作用 |
| --- | --- |
| Temporal persistence | 判断同一区域是否长时间持续激活 |
| Polarity mixture | 判断正负事件是否混杂、可能来自反光变化 |
| Temporal entropy | 判断事件是否在多个时间 bin 中扩散 |
| Time spread / signed direction | 表达事件运动时序形态 |

最终结构：

```text
delta_log = RGB_geometry_proposal * event_reliability_gate * residual_scale
```

其优势是：

- 事件真实进入最终深度预测路径。
- 当事件置零时，二阶段事件修正严格消失。
- 可靠性 gate 可以用 GT 所需修正区域进行监督。
- 同一场景不同 LDR 下可以约束最终深度/法向及修正误差一致。

### 已准备的验证方式

新脚本支持一键训练并自动运行反事实实验：

```bash
GPUS=6,7 bash mul_loss_fine/run_temporal_reliability_v2_2gpu.sh \
  data.root=/data1/lzh/dataset/reflective_raw
```

反事实测试包含：

```text
real events
zero events
reverse temporal bins
swap polarity
```

预期证明两件事：

1. `real` 与 `zero/reverse` 输出显著不同，证明事件确实参与运算。
2. `real` 的法向误差优于扰动事件，证明事件贡献是有效而非随机扰动。

## 明天建议汇报结构

### 第 1 页：问题与动机

标题：

> Event-guided Fine Geometry Recovery under Reflective and Over-exposed Observations

讲法：

- 反光材质中，LDR RGB 会出现过曝、纹理缺失和高光污染。
- 单纯 RGB 模型能够恢复粗轮廓，但手指、褶皱、局部边界等细节容易被抹平。
- 事件相机记录亮度变化，可能为这些困难区域提供额外证据。

### 第 2 页：任务与数据

展示：

- LDR RGB。
- Event bins 可视化。
- GT depth / GT normal。
- 模型预测 normal 的细节差异图。

一句话：

> 我们关注的不是整体轮廓能否出来，而是反光/过曝下局部高频几何能否恢复。

### 第 3 页：事件线索是否真的存在

展示相关性结果表：

```text
corr(event, geom_detail) = 0.2503
AUC = 0.6334
shuffle AUC = 0.4987
```

讲法：

> 结果说明事件与真实几何细节存在统计关联，但这种关联不足以把事件直接视为 dense depth label。

### 第 4 页：从直接融合到可靠性门控

画模型演进图：

```text
Direct event fusion -> Detail GT supervision -> Event-gated residual -> Temporal reliability V2
```

强调：

- 保留 VGGT 的全局几何能力。
- 事件只参与局部受限残差，不破坏粗结构。
- 这是从“事件写几何”转向“事件筛选可靠细节修正”。

### 第 5 页：目前定量进展

重点展示两张表：

```text
Normal loss: 0.148090 -> 0.143565 (约 3.1% 改善)
Representative normal error:
View 2: 31.12 deg -> 27.71 deg
View 3: 31.58 deg -> 28.75 deg
```

讲法：

> 当前收益主要体现在法向和局部几何细节，而不是粗深度整体误差，这符合方法目标。

### 第 6 页：定性可视化

放：

- Baseline 法向图。
- GT 法向图。
- 当前最佳 detail 模型法向图。
- 有明显手指/边缘/褶皱改善的局部放大图。

避免主动展示仍有强波纹的失败 case；如果被问到，再说明已用 reliability gate 专门处理。

### 第 7 页：最新 V2 与验证设计

展示：

```text
Temporal persistence + polarity + entropy
        -> reliability gate
        -> bounded residual correction
```

讲法：

> 最新版本不仅要求事件影响预测，还将通过置零、反序和极性交换验证这种影响是否具有正确方向。

### 第 8 页：下一步计划

用积极表述：

1. 完成 V2 多 LDR 训练与反事实验证，形成事件贡献闭环。
2. 统一 baseline / GT-detail / gated / V2 的定量和可视化对比。
3. 增加不同曝光等级、不同反光场景下的泛化验证。
4. 与现有事件三维重建方法做公平对比，整理论文实验表。

## 汇报时可以直接说的三段话

### 开场

> 我这阶段主要解决的是反光和过曝条件下深度细节恢复的问题。RGB 主干对整体几何已经比较稳定，但局部细节会变平。我的观察是事件流在这些区域仍然保存了变化信息，因此我把它作为精细几何修正的可靠性证据，而不是直接当深度标签。

### 结果页

> 目前最稳定的提升体现在法向细节指标上。gated refinement 相比 coarse prediction 的 depth-derived normal loss 下降约 3.1%，代表性视角的平均法向误差下降约 2.8 到 3.4 度。这个提升集中在局部高频区域，符合我们想恢复细节而不是只优化整体深度数值的目标。

### 下一步

> 现在我已经完成了时序可靠性 V2：它显式建模事件持续性、极性和时间熵，同时保证置零事件会撤掉二阶段修正。下一步重点不是继续堆 loss，而是用反事实实验和多曝光泛化把事件贡献验证完整。

## 老师可能追问与回答建议

### Q1：事件是否真的被模型使用，而不是 GT detail 在起作用？

回答：

> GT detail 建立了强监督基线，最新 V2 专门解决事件贡献可验证性问题。V2 的最终 residual 乘以 event-derived gate，事件置零后事件修正严格归零；同时已经准备 real / zero / reverse / polarity-swap 的反事实测试来量化输出变化和误差差异。

### Q2：为什么重点报告 normal，而不是 depth loss？

回答：

> 目标是恢复手指、折线、局部表面起伏等精细几何。整体 depth L1 更容易被大面积平滑区域主导，而法向与高频法向对局部形状变化更敏感，因此能更准确反映当前方法解决的问题。

### Q3：事件中的高光拖影会不会反而误导模型？

回答：

> 会，因此我没有让事件直接生成最终几何，而是引入 temporal reliability gate。持续激活、极性混杂和高时间熵可以作为高光轨迹的提示，GT 所需修正区域则用于教 gate 筛选可靠事件。

### Q4：多 LDR 训练有什么意义？

回答：

> 测试时仍然只输入单 LDR，但训练时同一帧在不同曝光下具有相同几何和共享事件线索。通过多曝光一致性约束，可以减少模型把某个曝光下的高光外观误认为真实几何细节。

### Q5：这和已有事件三维重建工作相比，新意在哪里？

回答：

> 当前重点不是用事件独立完成几何重建，而是在强 RGB 多视角几何主干上，为反光和过曝场景补充局部精细结构。核心区别是事件时序可靠性门控和多曝光下的局部残差一致性学习，面向的是反光物体的精细几何退化问题。

## 距离论文还有多远

### 当前所处阶段

比较积极也准确的判断是：

> 已经完成了方法雏形、问题定位、关键正相关证据、强基线和新的可验证事件模块，正在进入“把贡献闭环并整理成论文实验”的阶段。

这已经不是只有想法或只有 demo 的状态。现在已经有：

- 明确的问题定义：反光/过曝下的精细几何退化。
- 清晰的方法演进：GT detail -> event-gated residual -> temporal reliability V2。
- 初步正结果：相关性超过随机、法向细节误差下降、代表性可视化改善。
- 可执行的消融和反事实验证框架。

### 形成论文需要补齐的四组证据

| 需要补齐 | 目标 |
| --- | --- |
| 主结果表 | RGB baseline、GT-detail、gated、V2 在统一测试集上的 depth / normal / detail 指标 |
| 事件贡献闭环 | real / zero / reversed / swapped event 的反事实结果 |
| 泛化与鲁棒性 | 多曝光等级、更多场景、不同反光强度下的结果 |
| 外部对比 | 与合适的 RGB 或 event-assisted reconstruction 方法进行可复现实验对比 |

### 汇报中的措辞

建议不要说“现在已经足够投稿顶会”，而是说：

> 当前已经得到可写成论文方法主线的结果和模型设计。下一阶段将重点完成统一评测、事件反事实验证以及与现有方法的比较；如果 V2 能稳定保持当前法向细节收益并证明事件输入的增益，这条线具备进一步冲击高质量会议的潜力。

## 今晚最值得准备的材料

1. 一张最漂亮的 `baseline / prediction / GT normal` 对比图，标出手指或局部边缘。
2. 一张事件 bins 与 GT detail 的并列图。
3. 一张相关性指标小表。
4. 一张 gated 模型结构图，重点标出 `event gate -> residual`。
5. 一张 V2 方案图，说明它将作为下一轮核心实验。

组会的重点不是展示所有尝试，而是让大家看到：问题明确，事件线索有效，已有模型在细节指标上取得改善，新版本可以严格验证事件贡献，研究正沿着一条可形成论文的路线收敛。
