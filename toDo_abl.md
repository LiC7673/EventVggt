你现在需要基于当前 EventVggt 项目，实现一个快速 ablation validation pipeline。

目标：
不是最终论文实验，而是在小规模数据上验证论文核心假设：
"Geometry-aware event contribution learning 是否比 raw event fusion 有效。"

请不要大规模重构代码，尽量复用现有 train_contribution_stage1.py、stage2 训练流程和配置系统。

实验设置：

Synthetic dataset:
- 使用 scene-level split，禁止随机拆 view。
- 快速验证阶段只使用：
  - Train: 20 scenes
  - Test: 5 scenes
- 每个 scene 保留完整 multi-view 和 multi-exposure 结构。
- 保持原始 exposure 设置不变。
- 测试时只输入目标曝光 RGB + event，不使用 reference exposure。

需要实现以下 ablation：

==============================
Ablation 1: RGB-only baseline
==============================

目的：
验证 event 是否提供额外几何信息。

实现：
- 禁用 event branch。
- 使用原始 RGB VGGT/StreamVGGT 输出结果。
- 保持相同 evaluation protocol。

输出：
- depth error
- normal error
- pose error（如果已有）
- runtime（可选）


==============================
Ablation 2: Raw Event Fusion
==============================

目的：
验证 event 是否不能直接全部使用。

实现：
- 保留 event encoder。
- 去掉 geometry-aware event contribution score。
- contribution map 固定：

C(x,t)=1

即：

V_selected = V_event

直接输入 event refinement 模块。

保持：
- 网络结构尽量一致。
- 参数量接近。


==============================
Ablation 3: Geometry-aware Event Contribution (Ours)
==============================

目的：
验证学习 event contribution 的有效性。

使用当前完整 pipeline：

Event volume
+
Contribution Network
+
Geometry guidance
+
Multi-LDR supervision

输出：

C_geo(x,t)

然后：

V_selected = C_geo * V_event


==============================
Ablation 4: Without Multi-LDR supervision
==============================

目的：
验证 Multi-LDR 是否是真正的 supervision，而不是额外数据增强。

实现：

- 去掉 reference exposure。
- 不生成 bridge mask。
- contribution 只能依靠 geometry loss 学习。

保持其他结构不变。


==============================
Ablation 5: Saturation-mask selection
==============================

目的：
排除 contribution map 只是学习 saturation 区域。

实现：

不使用 ContributionNet。

构造：

C(x)=M_saturation(x)

然后：

V_selected = C * V_event

比较：

Raw event
vs
Saturation mask
vs
Learned contribution


==============================

训练要求：

1. 所有实验统一：
- dataset split
- training epoch
- optimizer
- learning rate
- evaluation script

2. 保存：
experiments/ablation/

目录：

ablation/
├── rgb_only/
├── raw_event/
├── ours/
├── no_multildr/
└── saturation_mask/


3. 每个实验保存：

- checkpoint
- evaluation json
- training log
- contribution visualization（如果存在）


==============================

重点检查：

1. Contribution map 是否 collapse。

每次 evaluation 输出：

C mean
C std
C min
C max


如果：

Cstd≈0

需要报告，不要隐藏。


2. 保存固定 test scene visualization：

每个实验至少保存：

- RGB input
- event projection
- contribution map
- selected event
- predicted depth
- GT depth

要求：可视化，每个test场景下的ev-0，1，2，5，10的结果

==============================

最终输出：

生成一个 ablation_results.csv：

columns:

Method,
Depth,
Normal,
Pose,
C_mean,
C_std


并给出简单分析：

- Raw event 是否优于 RGB-only？
- Learned contribution 是否优于 raw event？
- Multi-LDR supervision 是否提高 contribution selectivity？
- Contribution map 是否真的具有空间变化？

注意：
当前目标是快速验证论文假设，不追求最终 AAAI 实验规模。