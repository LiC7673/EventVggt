# EventVggt 当前统一流程 Overview

本文档描述当前工作区代码的真实执行路径，而不是尚未实现的论文设想。

## 一、输入与数据配对

1. 每个样本包含连续 `V` 个视角。
2. 每个视角包含：
   - 单曝光 RGB：`img`；
   - 深度 GT：`depthmap`；
   - 相机内参：`camera_intrinsics`；
   - 相机位姿 GT：`camera_pose`；
   - 事件体：`event_voxel`，形状为 `[2B,H,W]`；
   - Phase A 专用几何事件体：`geometry_event_voxel`，形状同样为 `[2B,H,W]`；
   - 有效区域 mask。
3. 训练时为相同场景、相同帧、相同相机和相同事件区间读取两个曝光 RGB。
4. 曝光集合为 `ev_0, ev_1, ev_2, ev_5, ev_10`。
5. 根据饱和度将两个曝光重新排序：
   - `target RGB`：质量更差、过曝更严重的图像；
   - `reference RGB`：质量相对更好的图像。
6. 两个曝光共用同一份事件流。事件流不会因为 RGB 曝光不同而改变。
7. `reference RGB` 不直接参与最终几何解码，只用于：
   - 构造训练期 Bridge；
   - Phase B/C 的跨曝光贡献度一致性监督；
   - 训练可视化。

## 二、RGB-only VGGT 主干

8. 模型将所有 target RGB 堆叠为 `[N,V,3,H,W]`。
9. 只有 RGB 被送入冻结的 VGGT Aggregator：

   ```text
   RGB -> Frozen VGGT Aggregator -> multi-view RGB tokens
   ```

10. 事件流不会进入 VGGT Aggregator，也不会与 RGB 在 Aggregator 输入端拼接。
11. CameraHead 只读取 RGB tokens：

   ```text
   RGB tokens -> RGB-only CameraHead -> camera pose
   ```

12. CameraHead 在统一训练的所有阶段都被冻结，事件不会修改 pose。

## 三、第一次 RGB-only 深度解码：coarse geometry

13. RGB tokens 第一次进入 DepthHead，不传入任何事件特征。
14. 此次 DepthHead 完全走原始 VGGT DPT 路径，输出：
   - `coarse_depth`；
   - `coarse_depth_confidence`。
15. 使用内参从 coarse depth 计算：

   ```text
   coarse normal = depth_to_normals(coarse depth, intrinsics)
   ```

16. 同时从最后一层 RGB tokens 提取 patch-grid coarse RGB feature。
17. 因此 coarse geometry 包括：
   - coarse depth；
   - coarse normal；
   - coarse RGB patch feature。

## 四、ContributionNet

18. ContributionNet 的输入为：

   ```text
   event voxel
   target RGB
   coarse depth
   coarse normal
   coarse RGB patch feature
   ```

19. coarse depth、normal 和 RGB feature 在进入 ContributionNet 时均被 `detach`，ContributionNet 不会反向修改 VGGT RGB 主干。
20. 基础 ContributionNet 使用 U-Net 式二维卷积结构预测空间贡献：

   ```text
   C_spatial: [N,V,H,W]
   ```

21. TemporalContributionNet 再根据归一化事件体，为每个时间 bin 和正负极性预测独立偏移。
22. 最终贡献张量为：

   ```text
   C: [N,V,2B,H,W], C in [0,1]
   ```

23. 空间贡献图用于监督和可视化，它是对所有事件通道按事件质量加权后的平均贡献。

## 五、事件选择

24. Phase B/C 中，ContributionNet 输出直接作用于 full event：

   ```text
   selected_event = C * E_full
   ```

25. Phase A 不运行 ContributionNet，直接令 `selected_event = E_geo`；贡献度只在 Phase B/C 中连续缩放一次 full-event 幅值。
26. 后续 GeometryAdapter 只使用二值 support 保证零事件严格返回 RGB 路径，不再用连续 `C` 进行第二次幅值门控，因此不会形成 `C^2`。

## 六、极性与时间事件编码

27. selected event 按通道拆分为 positive 和 negative 两组。
28. 每组包含 `B` 个时间 bin。
29. 每个 bin 额外加入归一化时间坐标 `t in [-1,1]`：

   ```text
   [event, t * event]
   ```

30. 正负极性分别经过 3D convolution。
31. 两个极性 feature 拼接后再次进行 3D temporal convolution。
32. temporal attention 沿时间维聚合得到二维事件 feature。
33. 事件 feature 经过四个独立 projection，形成四组 event feature。
34. 当前 v2 中四组 event feature 均位于统一 transformer patch grid。
35. 对于 `392x518` 输入和 `patch_size=14`：

   ```text
   patch grid = 28x37
   ```

36. Phase B/C 中，第一组纯事件 patch feature 与 reliability map 还会送入独立 EventNormalDecoder。
37. EventNormalDecoder 不读取 RGB、coarse depth、coarse normal 或 RGB tokens，直接输出单位事件法向。

## 七、当前 GeometryAdapter 与第二次几何解码

38. 当前代码不是“完整 DPT decoder 后接一个 Adapter”。
39. DepthHead 和 PointHead 各自包含四个 GeometryFeatureAdapter，总共八个 Adapter。
40. 对每一个 DPT intermediate layer：
   1. 从对应 Aggregator layer 取出 RGB patch tokens；
   2. 执行 LayerNorm；
   3. 使用原始 DPT `1x1 project` 得到 RGB patch feature；
   4. 在统一 `28x37` patch grid 上，将 RGB patch feature 与对应 event feature 拼接；
   5. GeometryFeatureAdapter 预测 feature update；
   6. update 由可学习的 `tanh(alpha)` 控制；
   7. update 使用 bilinear resize 对齐到该 DPT layer 的 resize 后尺寸；
   8. 原始 RGB feature 仍走预训练 DPT `resize_layer`；
   9. 最终输入 DPT scratch decoder 的 feature 为：

      ```text
      resized RGB feature + bilinear-resized event update
      ```

41. event update 不经过 stride-4/stride-2 ConvTranspose，避免事件 residual 激活转置卷积的周期性 polyphase 网格。
42. 四层融合 feature 进入原始 DPT：

   ```text
   scratch.layer*_rn
   -> refinenet4
   -> refinenet3
   -> refinenet2
   -> refinenet1
   -> output_conv1
   -> output_conv2
   ```

43. DepthHead 输出最终 depth 和 depth confidence。
44. PointHead 独立执行同样的四层事件 Adapter 流程，输出最终 point map 和 point confidence。
45. 因此 DepthHead 实际执行两次：
   - 第一次：RGB-only coarse depth；
   - 第二次：带事件 Adapter 的 final depth。
46. PointHead只执行带事件 Adapter 的最终解码，没有单独导出 coarse point map。

## 八、最终法向与模型输出

45. 模型没有独立 normal prediction head。
46. 最终 normal 由最终 depth 和相机内参计算：

   ```text
   final normal = depth_to_normals(final depth, intrinsics)
   ```

47. 每个 view 最终返回：
   - final depth；
   - coarse depth；
   - final normal；
   - final point map；
   - RGB-only camera pose；
   - contribution tensor；
   - spatial contribution；
   - selected event mass；
   - Adapter alpha；
   - Adapter update magnitude；
   - confidence maps。

## 九、Bridge 与 decomposition target

48. Bridge 只在训练期构造：

   ```text
   Bridge = saturated(target RGB)
            AND visible(reference RGB)
            AND event support
   ```

49. Bridge 不是 ContributionNet 的 GT，也不参与推理。
50. 在 Phase B/C 中，Bridge 仅提高对应区域的 depth/normal 几何监督权重。
51. 若启用 Blender decomposition，数据集额外加载 geometry event。
52. soft physical contribution target 为：

   ```text
   C_gt = sum_t(abs(E_geo)) / (sum_t(abs(E_input)) + eps)
   ```

53. `C_gt` 是事件来源比例，不是 depth edge、binary mask 或 geometry segmentation。

## 十、Phase A：Adapter 预训练

54. ContributionNet 在 Phase A 中被冻结且不会执行前向。
55. Phase A 将每个 view 的模型输入 `event_voxel` 临时替换为 `geometry_event_voxel`。
56. Phase A 的训练和验证均严格使用 `selected_event = E_geo`，不使用 full event，也不使用随机 contribution mask。
57. Phase A 默认训练：
   - PolarityTemporalEventPyramid；
   - 四个 Depth GeometryAdapters；
   - 四个 Point GeometryAdapters。
   - EventNormalDecoder 在 Phase A 冻结且不执行。
58. Phase A 默认冻结：
   - VGGT Aggregator；
   - CameraHead；
   - ContributionNet；
   - 原始 DPT parameters。
59. Phase A 当前有效损失为：

   ```text
   L_A = L_depth
         + lambda_normal * L_normal
         + lambda_point * L_point
         + lambda_grad * L_log-depth-gradient
         + lambda_curv * L_log-depth-curvature
         + lambda_grid * L_patch-boundary-gradient
   ```

   后三项均匹配 GT depth 的导数，而不是对预测深度做无监督平滑；因此它们抑制
   DPT/ViT patch 周期条纹，同时保留 GT 中真实存在的曲面和高频几何。

60. Phase A 中打印的 budget/Cmean/Cstd 只是 override 统计，不参与总损失。

## 十一、Phase B：ContributionNet 与事件法向分支学习

61. 启动时先连续执行两个 Phase-A warm-up epoch。
62. warm-up 后从最佳 Adapter checkpoint 开始 Phase B，并按 `B -> A -> B -> A` 每个 epoch 切换阶段。
   - `EPOCHS_A` 固定为 2，表示启动 warm-up；
   - `EPOCHS_B` 表示后续 `B -> A` 交替循环次数，而不是一整段连续 B epoch。
63. Phase B 恢复 dataset 主 `event_voxel`；在 full 实验中该字段被强制校验为 decomposition `E_full`。
63. Phase B 冻结：
   - Event Encoder；
   - Depth/Point GeometryAdapters；
   - 原始 DPT heads；
   - VGGT Aggregator；
   - CameraHead。
64. Phase B 只更新 ContributionNet 和 EventNormalDecoder，并使用 `selected_event = C_pred * E_full`。
65. 冻结的 Event Encoder 和 Adapter 仍保留计算图，因此最终 geometry loss 可以反向传到贡献度 `C`。
66. Phase B 使用：
   - final depth loss；
   - final normal cosine loss；
   - final point loss；
   - event-mass budget loss；
   - multi-LDR pair consistency；
   - low-contribution update constraint；
   - optional Blender decomposition Smooth-L1；
   - geometry/contribution ranking loss。
   - direct event-normal GT cosine loss；
   - depth-normal/event-normal consistency loss。
67. EventNormalDecoder 只接收纯事件 feature 和 reliability map，直接预测单位法向。
68. event-normal GT loss负责锚定事件法向；深度一致性项使用 `stopgrad(event normal)`，强制深度导出的法向跟随事件法向。
67. reference exposure 的 ContributionNet 前向使用 stop-gradient，只作为跨曝光一致性 anchor。

## 十二、Phase C：短暂联合校准

68. Phase C 开始前加载 `checkpoint-contribution-best.pth`，事件输入与 Phase B 相同，使用 full event。
69. Phase C 更新：
   - ContributionNet：基础学习率；
   - Event Encoder 与 GeometryAdapters：`0.1x` 学习率；
   - Depth/Point DPT heads：`0.03x` 学习率。
70. VGGT Aggregator 和 CameraHead继续冻结。
71. Phase C 使用与 Phase B 相同的 contribution 和 geometry losses。

## 十三、推理流程

72. 推理只输入：
   - 单曝光 RGB sequence；
   - 对应 raw event voxel。
73. 推理不需要：
   - reference RGB；
   - Bridge；
   - GT depth/normal；
   - `E_geo`；
   - decomposition target。
74. 推理流程为：

   ```text
   RGB -> Frozen VGGT Aggregator -> RGB tokens
   RGB tokens -> RGB-only coarse depth
   Event + RGB + coarse geometry -> ContributionNet -> C
   C * Event -> Polarity/Temporal Event Encoder
   RGB tokens + event features -> Depth/Point GeometryAdapters + DPT decoders
   -> final depth / point / normal
   RGB tokens -> CameraHead -> pose
   ```

## 十四、当前实现与“单一 post-DPT Adapter”设想的差异

75. 当前 coarse depth 只直接输入 ContributionNet，没有直接输入 GeometryFeatureAdapter。
76. 当前 Adapter 位于 DPT scratch decoder 之前，而不是完整 DPT decoder 之后。
77. 当前不是一个共享 Adapter，而是：
   - DepthHead 四个 Adapter；
   - PointHead 四个 Adapter。
78. 当前 final geometry 仍由原始 DPT scratch decoder 和 `output_conv2` 输出，不使用 raw-depth residual。
79. 旧代码中的 `SelectedEventRefiner` 仍保留在源码中，但统一 A/B/C 模型不会实例化或调用它。
80. 如果目标设计确定为“完整 DPT decoded feature 后接单一 GeometryAdapter”，仍需要进一步重构当前模型。
