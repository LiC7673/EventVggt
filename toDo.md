请直接修改当前 EventVggt 工作区，完成一个能够在短时间内训练和验证的最小版本。

不要先写设计报告，不要等待确认，直接检查当前代码并实施修改。

本次不是重写模型，只修改现有统一 A/B/C pipeline。优先搜索并复用以下现有类：

- ContributionNet
- TemporalContributionNet
- PolarityTemporalEventPyramid
- GeometryFeatureAdapter
- 当前统一模型 forward
- 当前 Phase A / Phase B / Phase C 训练脚本

不要修改：

- VGGT / StreamVGGT Aggregator
- CameraHead
- 原始 DPT scratch decoder 结构
- RGB checkpoint 加载逻辑
- 数据集原有 RGB、depth、pose、intrinsics 读取逻辑
- legacy SelectedEventRefiner；它保持未使用状态即可

==================================================
一、最终目标
==================================================

实现以下训练路径：

Phase A:
    使用 Blender geometry event E_geo
    训练 EventEncoder + GeometryFeatureAdapter
    学习如何利用几何来源事件改善 depth / point / normal

Phase B:
    使用 full event E_full
    冻结 EventEncoder + GeometryFeatureAdapter
    只训练 ContributionNet
    通过 C * E_full 模拟 E_geo 的作用

Phase C:
    暂时保持关闭，默认 epochs=0
    本轮不要重点实现或调试联合训练

推理只使用：

    target RGB + full event

不能依赖：

    E_geo
    reference RGB
    Bridge
    GT depth / normal

==================================================
二、数据集：加载 geometry event
==================================================

在当前 Dataset 中增加：

    geometry_event_voxel

其 shape、时间区间、bin 数、极性顺序、空间 resize 必须与当前：

    event_voxel

完全一致，均为：

    [2B, H, W]

geometry event 文件按以下顺序查找，找到第一个存在的路径：

1. <scene>/events_additive/geometry_motion/events.h5
2. <scene>/events_additive/diffuse_motion/events.h5
3. <scene>/events_additive/geometry/events.h5

full event 继续使用当前 event_voxel 的读取路径，不要重新实现 full event loader。

geometry event 必须使用与 full event 完全相同的：

- start timestamp
- end timestamp
- voxel bins
- polarity convention
- resize
- y flip / spatial transform

若当前 Phase A 或 Phase B 启用了 decomposition supervision，但缺少 geometry event，必须立即抛出包含 scene 名和文件路径的 RuntimeError，不能静默回退。

在 Dataset 中生成 channel-level soft target：

    C_gt = abs(V_geo) / (abs(V_full) + eps)

要求：

    C_gt = clamp(C_gt, 0, 1)

shape：

    [2B, H, W]

只在 full event support 上有效：

    event_support = abs(V_full).sum(channel) > 0

同时生成仅用于可视化的空间 target：

    C_gt_spatial =
        sum_channel(abs(V_full) * C_gt)
        /
        (sum_channel(abs(V_full)) + eps)

不要使用 depth gradient 生成 C_gt。
不要把 C_gt 转成二值 mask。

collate 后需要得到：

    geometry_event_voxel: [N, V, 2B, H, W]
    contribution_target:  [N, V, 2B, H, W]

==================================================
三、Phase A：只用 geometry event 训练 Adapter
==================================================

保留现有 Phase A 训练框架，但修改事件输入。

Phase A 中：

    selected_event = geometry_event_voxel

不要使用 full event。
不要运行 ContributionNet。
不要使用随机 contribution mask。
不要使用 budget loss。
不要使用 pair consistency。
不要使用 decomposition loss。

冻结：

- VGGT Aggregator
- CameraHead
- ContributionNet
- 原始 DPT 参数

训练：

- PolarityTemporalEventPyramid
- DepthHead 中四个 GeometryFeatureAdapter
- PointHead 中四个 GeometryFeatureAdapter

损失：

    L_A =
        L_depth
        + lambda_normal * L_normal
        + lambda_point * L_point

Bridge 不能限制 Phase A 的监督范围。

depth / normal / point loss 应在全部 valid geometry 上计算。

保存：

    checkpoint-adapter-best.pth

Phase A 的验证需要同时报告：

- RGB-only coarse depth metrics
- E_geo adapter final depth metrics
- improvement over coarse depth

==================================================
四、GeometryFeatureAdapter：最小 LoRA-style 改造
==================================================

不要实现动态权重生成，不要修改 Transformer 的 Q/K/V 权重。

这里实现的是 LoRA-style bottleneck adapter，不是真正的 LoRA。

保持 Adapter 位于当前四个 DPT intermediate layer，不移动 decoder。

当前每层已有：

    rgb_feature
    event_feature

增加：

    coarse_geometry_feature

coarse geometry 由：

    coarse depth:  1 channel
    coarse normal: 3 channels

拼接为：

    G_coarse: [N*V, 4, H, W]

执行：

1. coarse depth 先转为 log depth：

       log_depth = log(clamp(depth, 1e-6))

2. 将 log_depth 和 normal 拼成 4 channels。

3. bilinear resize 到当前统一 patch grid。

4. 使用每层独立的 1x1 Conv 投影到 rank=16：

       geo_proj: Conv2d(4, 16, 1)

修改 GeometryFeatureAdapter 为：

    input = concat(
        rgb_feature,
        event_feature,
        geo_feature
    )

    hidden = Conv2d(input_channels, 16, 1)
    hidden = GELU(hidden)
    delta = Conv2d(16, rgb_channels, 1)

    output =
        rgb_feature
        + tanh(alpha) * support * delta

要求：

- 最后一个 up projection 必须 zero initialization；
- alpha 初始值设为 0；
- support 是 selected_event 的二值 support；
- support resize 到当前 feature 尺寸时使用 nearest；
- continuous contribution C 只能在 event voxel 上乘一次；
- Adapter 内不能再次乘 continuous C，避免 C^2；
- 不预测 raw depth residual；
- 不替换 RGB feature；
- 输出 shape 必须与原 rgb_feature 完全一致。

coarse depth、coarse normal、RGB tokens 进入 Adapter 前全部 detach。

==================================================
五、Phase B：冻结 Adapter，训练 ContributionNet
==================================================

Phase B 开始前加载：

    checkpoint-adapter-best.pth

冻结：

- VGGT Aggregator
- CameraHead
- EventEncoder
- 所有 GeometryFeatureAdapter
- 原始 Depth/Point DPT parameters

只允许 ContributionNet 参数：

    requires_grad=True

输入：

    full event V_full

ContributionNet 输出：

    C_pred: [N, V, 2B, H, W]

选择事件：

    selected_event = C_pred * V_full

然后通过冻结的：

    EventEncoder
    GeometryFeatureAdapter
    DPT decoder

得到 final depth / point / normal。

Phase B 损失只保留以下四项，不要加入 ranking loss 或其他新损失：

1. Attribution loss

    support_channels = abs(V_full) > 0

    L_attr =
        SmoothL1(
            C_pred[support_channels],
            C_gt[support_channels]
        )

2. Multi-LDR contribution consistency

对相同 event、相同 view、不同 exposure：

    C_target = ContributionNet(target RGB, V_full, ...)
    C_reference = stopgrad(
        ContributionNet(reference RGB, V_full, ...)
    )

使用 event-mass weighted L1：

    L_pair =
        sum(abs(V_full) * abs(C_target - C_reference))
        /
        (sum(abs(V_full)) + eps)

3. Geometry loss

使用 final depth / normal / point：

    L_geo =
        L_depth
        + lambda_normal * L_normal
        + lambda_point * L_point

监督区域：

    weight = valid_mask * (1 + beta_bridge * bridge)

Bridge 只增加权重，不能作为唯一监督区域。

4. Budget loss

不要再固定 rho=0.5。

从 C_gt 计算每个 batch 的目标事件质量比例：

    rho_gt =
        sum(abs(V_full) * C_gt)
        /
        (sum(abs(V_full)) + eps)

预测比例：

    rho_pred =
        sum(abs(V_full) * C_pred)
        /
        (sum(abs(V_full)) + eps)

    L_budget = abs(rho_pred - stopgrad(rho_gt))

最终固定为：

    L_B =
        1.0 * L_attr
        + 0.2 * L_pair
        + 0.2 * L_geo
        + 0.05 * L_budget

所有权重增加为命令行参数，但默认值按上面设置。

保存：

    checkpoint-contribution-best.pth

==================================================
六、验证模式
==================================================

在统一 evaluate 函数中增加三个 event mode：

    --event-mode full
        selected_event = V_full

    --event-mode oracle
        selected_event = V_geo

    --event-mode predicted
        selected_event = C_pred * V_full

三种模式必须使用完全相同的：

- RGB 输入
- VGGT checkpoint
- EventEncoder
- GeometryAdapter
- DPT decoder
- 测试样本

输出一个 JSON：

{
    "rgb_only": {...},
    "full_event": {...},
    "oracle_geo_event": {...},
    "predicted_contribution": {...}
}

指标至少包括：

- depth AbsRel
- depth RMSE
- normal cosine error
- point error
- C MAE on event support
- C Pearson correlation
- predicted event mass ratio
- target event mass ratio

==================================================
七、可视化
==================================================

每次验证固定保存前 4 个样本：

1. target RGB
2. full event projection
3. geometry event projection
4. predicted spatial contribution
5. target spatial contribution
6. selected event projection
7. coarse depth
8. final depth
9. GT depth
10. final normal

可视化不得重新遍历完整 DataLoader。

直接复用当前 validation forward 已得到的 prepared 和 prediction，只保存指定 batch 的结果。

==================================================
八、必须增加的测试
==================================================

增加或修改单元测试，至少验证：

1. Phase A 的 selected_event 与 geometry_event_voxel 完全一致；
2. Phase A 不运行 ContributionNet；
3. Phase B 中只有 ContributionNet 存在非零梯度；
4. C 只在 event voxel 上连续相乘一次；
5. Adapter 输出 shape 与 RGB feature 一致；
6. Adapter 的最后 up projection 初始输出严格为 0；
7. coarse geometry 已 detach；
8. C_gt 范围位于 [0,1]；
9. 不存在 geometry event 时给出明确错误；
10. full / oracle / predicted 三种模式能够各跑一个 batch。

==================================================
九、兼容要求
==================================================

不要删除旧 checkpoint。
新 checkpoint schema 需要升级版本，避免误加载旧模型。

保持当前 shell 入口可用。

Phase C 默认：

    EPOCHS_JOINT=0

先保证以下快速命令可以运行：

Phase A 快速测试：

    MAX_TRAIN_BATCHES=20 \
    MAX_VAL_BATCHES=5 \
    EPOCHS_ADAPTER=1 \
    NUM_WORKERS=0 \
    bash contribution_stage1.sh

Phase B 快速测试：

    MAX_TRAIN_BATCHES=20 \
    MAX_VAL_BATCHES=5 \
    EPOCHS_CONTRIBUTION=1 \
    NUM_WORKERS=0 \
    bash contribution_stage1.sh

实现完成后输出：

1. 修改文件列表；
2. 每个文件的关键修改；
3. 两条实际运行命令；
4. 单元测试结果；
5. 参数量统计：
   ContributionNet、EventEncoder、全部 Adapter 分别多少参数。

不要继续设计额外模块。
不要实现动态 LoRA。
不要修改 pose branch。
不要实现 post-decoder depth residual。