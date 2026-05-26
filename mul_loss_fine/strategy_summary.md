# 多损失与细节学习策略简要总结

## 核心问题

当前模型的主要矛盾不是 loss 数值是否下降，而是深度细节学不到，深度导出的法向图容易出现过平滑、拖影、网格或条纹伪影。由于纯 RGB 微调也会出现网格感，说明问题不完全来自事件流，也和 VGGT 的 patch/token 到 dense depth 解码方式有关。

## 这几轮策略

1. **先验证事件流是否真的有用**
   - 写了事件流可视化和 cue correlation 测试。
   - 结论是事件流和几何边界/法向梯度有一定相关性，但直接把事件当 dense 几何标签不可靠。
   - 事件更适合作为“哪里可能有细节”的支持区域，而不是直接监督深度值。

2. **修正事件流加载和下采样**
   - 检查了 `t, x, y, p` 的顺序、时间区间和空间翻转/旋转对齐。
   - 将事件流下采样从简单坐标除法，改向 `voxel grid + antialias resize` 的思路。
   - 目标是避免事件图出现块状 aliasing，减少事件拖影对深度的误导。

3. **尝试事件支持的多视角细节约束**
   - 不把事件直接当 GT，而是用事件 co-support 给多视角一致性加权。
   - 在两个 view 之间投影后，约束预测法向、细节存在性、高频法向等。
   - 目标是让事件区域推动模型恢复细节，同时避免无事件区域被错误强化。

4. **引入 GT detail 辅助**
   - 如果多视角事件约束只能改善指标但仍不出细节，就用 GT 法向/GT depth 导出的细节作为更直接的高频监督。
   - 事件只作为权重增强项，用来强调事件支持的细节区域。

5. **处理纯 RGB 也有的网格伪影**
   - 在 `finetune_event.py` 和 `finetune_no_event.py` 中加了默认关闭的二阶平滑和 patch-grid 抑制 loss。
   - 另外新增了 `antigrid` 模型变体，在旧模型输出后接轻量 refiner，预测 log-depth residual，用结构方式减轻网格感。

## `mul_loss_fine` 脚本作用

| 脚本 | 作用 |
| --- | --- |
| `event_supported_mv_loss.py` | 核心 loss 实现。基于事件 support，计算多视角法向一致性、细节存在性、高频一致性、方向一致性，以及 GT detail 辅助监督。 |
| `launcher.py` | 统一注入多损失配置。每个消融脚本自己的权重会覆盖 yaml，避免多个脚本实际跑成同一组权重。 |
| `finetune_mul_loss_baseline.py` | 基线，只使用原始 `finetune_event.py` 监督，不加额外 multi-loss。 |
| `finetune_mul_loss_mv_normal.py` | 事件支持区域内，多视角投影后的预测法向保持一致。 |
| `finetune_mul_loss_mv_presence.py` | 事件支持区域应该有足够的几何细节/法向梯度，避免事件区域被预测成过平滑。 |
| `finetune_mul_loss_mv_hf.py` | 事件支持区域内，对齐多视角高频法向分量，重点压细节高频一致性。 |
| `finetune_mul_loss_mv_presence_hf.py` | 推荐细节组合：只开 presence 和 high-frequency，不开完整 normal consistency，避免把细节拉平。 |
| `finetune_mul_loss_mv_normal_hf.py` | 组合 normal consistency 和 high-frequency consistency。 |
| `finetune_mul_loss_mv_all.py` | 同时使用 normal、presence、high-frequency 三类事件支持多视角约束。 |
| `finetune_mul_loss_mv_all_orient.py` | 在 `mv_all` 基础上增加细节方向一致性，用来约束边缘/纹理走向。 |
| `finetune_mul_loss_detail_gt.py` | 使用 GT 法向/GT depth 导出的细节监督，事件只用于增强细节区域权重。 |
| `finetune_mul_loss_detail_gt_uniform.py` | 严格对照：固定使用 GT depth 推导法向的高频细节监督，不使用事件重新加权。 |
| `finetune_mul_loss_detail_gt_selective_event.py` | 在相同 depth-derived GT 监督上，仅对时序/极性事件支持最强的前 20% 像素增强权重，用来验证事件是否真的帮助困难细节。 |
| `finetune_mul_loss_detail_gt_temporal_bins.py` | 与 `uniform` 使用同一套 GT-detail loss，但用正确注入 token 的时序 bin 事件模型，专门验证事件输入贡献。 |
| `finetune_mul_loss_detail_gt_temporal_detail.py` | 推荐的细节模型：保留正负时间 bin，在像素域预测受限深度残差，避开直接事件 patch-token 注入带来的规则网格。 |
| `finetune_mul_loss_detail_gt_temporal_gated.py` | 防波纹细节模型：事件只产生低通门控，RGB/粗深度决定残差几何形状，并加残差二阶约束。 |
| `finetune_mul_loss_detail_gt_temporal_gated_multildr.py` | 同帧多 LDR 配对训练曝光不变的 RGB/事件 agreement，测试仍只用单 LDR；用于压制随曝光变化的反射伪细节。 |
| `finetune_mul_loss_detail_gt_temporal_adapter.py` | 从已收敛 `uniform` 初始化并冻结 RGB/heads，仅训练时序事件 token 分支；用于检验事件能否提供增量细节修正。 |
| `finetune_mul_loss_detail_gt_salient.py` | 在 `detail_gt` 基础上进一步聚焦 GT 中最显著的高频几何细节，强化高频形状、幅值和细节存在性。 |
| `finetune_mul_loss_mv_all_detail_gt.py` | 最强组合：事件支持的多视角约束 + GT detail 辅助监督。 |
| `finetune_mul_ldr_event.py` | 多 LDR 曝光训练，用不同曝光下的 RGB/事件增强鲁棒性，并可加入曝光一致性约束。 |
| `run_mul_loss_2gpu_8gpu.sh` | 八卡批量跑消融，每个脚本默认吃两张卡。 |
| `run_detail_gt_event_pair_2gpu_5678.sh` | 默认使用物理第 5-8 张卡（CUDA ID `4,5` 与 `6,7`）并行跑 GT-detail 对照和选择性事件加权实验。 |
| `run_temporal_bins_compare_2gpu_5678.sh` | 用同一 GT-detail loss 对比旧 `uniform` 与真正保留/融合事件时间 bin 的模型。 |
| `run_temporal_detail_compare_2gpu_5678.sh` | 两卡一任务对比 `uniform` 与像素域 temporal-detail 残差模型。 |
| `run_mul_ldr_2gpu.sh` | 两卡跑多 LDR 训练实验。 |

## 当前判断

`mul_loss_fine` 的核心思想是：**事件流不直接监督深度，而是作为细节区域的可靠性权重**。进一步检查发现旧 `streamvggt.py` 虽生成事件特征，却因 token 张量维度判断和通道宽度不匹配而未实际注入 RGB token，因此 `detail_gt_uniform` 应视为强 GT-detail 对照，而非事件输入结果。`detail_gt_temporal_bins` 可验证 token 级事件输入，但强高频监督下容易显露 patch 网格；用于最终细节实验时，应优先采用像素域残差的 `detail_gt_temporal_detail`。
