# 消融实验清单

## 1. 实验目的

本文方法包含三个核心策略：

1. **事件可靠性建模**：利用可控的几何事件 (E_{\mathrm{geo}}) 监督模型从带噪事件 (E_{\mathrm{full/cur}}) 中识别对几何有效的响应，并预测融合置信图 (C_{\mathrm{fusion}}) 与细化置信图 (C_{\mathrm{refine}})。
2. **Multi-LDR 几何特征补全**：利用同一场景不同曝光下几何一致的性质，使低质量 LDR 与事件联合恢复接近参考曝光 (ev_0) 的 HDR-like 几何特征。
3. **事件微分几何细化**：事件分支预测法向导数，并通过逐像素深度 refiner、高频深度残差以及深度–法向导数一致性，将局部几何细节传播到最终深度。

消融实验需要分别回答：

- 带噪事件是否可以不经过可靠性建模直接使用？
- Multi-LDR 是否真正改善全局几何与深度尺度？
- 事件法向导数是否能改善局部曲面和高频细节？
- 改善是否来自事件信息，而不是额外参数、微调或测试尺度对齐？

---

## 2. 统一实验协议

所有主表和消融实验必须使用完全相同的：

- 训练场景：固定 12 个场景；
- 测试场景：固定 4 个未见场景；
- 测试曝光：(ev_0,ev_1,ev_2,ev_5,ev_{10})；
- 每个曝光的测试帧数和 4-view 窗口；
- 图像分辨率、事件时间窗口和 5-bin polarity voxel grid；
- 初始 VGGT 权重；
- 训练 batch 数或 epoch 数；
- 深度有效区域 mask；
- 固定深度尺度协议，不允许每张测试图利用 GT 单独对齐；
- checkpoint 选择规则，不允许根据测试集挑选最佳权重。

每组实验至少报告：

### 深度指标

- MAE；
- AbsRel；
- RMSE；
- RMSE-log；
- δ1 / δ2 / δ3。

### 法向指标

- Mean angular error；
- Median angular error；
- (<11.25^\circ)；
- (<22.5^\circ)；
- (<30^\circ)。

法向只在 GT 深度有效 mask 内评估，并统一由预测深度导出，不能混用独立 normal head 的输出。

### 局部细节指标

- 深度梯度误差；
- 深度曲率误差；
- 法向导数误差；
- 高频深度残差误差；
- coarse 到 final 的指标增益。

---

## 3. 主结果与基础对照

| 编号 | 实验 | RGB | Event | 微调 | 用途 |
|---|---|---|---|---|---|
| B0 | RGB pretrained | ✓ |  |  | 原始 VGGT 基线 |
| B1 | RGB fine-tuned | ✓ |  | ✓ | 排除收益仅来自微调 |
| B2 | RGB + raw event | ✓ | ✓ | ✓ | 不使用本文策略的事件融合基线 |
| Full | 完整方法 | ✓ | ✓ | ✓ | 本文最终结果 |

检查项：

- [ ] B0 使用原始基础权重，没有加载事件模型 checkpoint。
- [ ] B1 与 Full 使用相同训练场景、训练量和深度监督。
- [ ] B2 的参数量尽量与 Full 接近，但不使用 (E_{\mathrm{geo}})、Multi-LDR 和法向导数机制。
- [ ] 所有方法使用相同的固定测试尺度。
- [ ] 分别报告每个曝光以及所有曝光的像素加权平均结果。

---

## 4. 最小核心消融集合

这是论文主消融表必须包含的最小集合。

| 编号 | 事件可靠性 | Multi-LDR | Pixel refiner | 法向导数耦合 | 说明 |
|---|---:|---:|---:|---:|---|
| A0 |  |  |  |  | RGB-only fine-tuned |
| A1 |  |  | ✓ | ✓ | 只用带噪事件训练细化分支 |
| A2 |  | ✓ |  |  | 只保留 Multi-LDR 特征补全 |
| A3 | ✓ | ✓ | ✓ |  | 完整模型去掉 refiner 法向优化 |
| A4 | ✓ | ✓ | ✓ | ✓ | 完整方法 |

### A1：Noisy Event Only

- [ ] 主事件只读取带噪 `cur_event` 或 (E_{\mathrm{full}})。
- [ ] 不读取 (E_{\mathrm{geo}})。
- [ ] 不训练 (C_{\mathrm{fusion}}) 和 (C_{\mathrm{refine}})，两者固定为 1。
- [ ] 不使用 Multi-LDR HDR token 对齐。
- [ ] 训练 event encoder、事件法向导数头和 pixel refiner。
- [ ] 用来证明直接把噪声事件加入模型会产生伪细节或不稳定更新。

对应实验目录：

```text
exp_f/latest_three_strategy_ablation/noisy_event_only
```

### A2：Multi-LDR Only

- [ ] 输入为低质量 LDR 与 `cur_event`。
- [ ] 以 (ev_0) 作为 HDR-like 几何特征参考。
- [ ] 只训练 event encoder、event token projection 和 HDR Adapter。
- [ ] 关闭 pixel refiner。
- [ ] 关闭事件法向导数监督。
- [ ] 关闭两个置信图。
- [ ] 用来验证 Multi-LDR 对全局形状、深度和尺度恢复的作用。

对应实验目录：

```text
exp_f/latest_three_strategy_ablation/multi_ldr_only
```

### A3：Without Refiner Normal Optimization

- [ ] 保留可靠性建模、Multi-LDR、HF 深度残差和 pixel refiner。
- [ ] 保留事件法向导数自身的监督。
- [ ] 删除 final normal 对 refiner 的直接监督。
- [ ] 删除 final-depth normal derivative 与 GT normal derivative 的一致性损失。
- [ ] 用来验证法向导数是否真正将事件边缘传播到最终深度。

对应实验目录：

```text
exp_f/latest_three_strategy_ablation/without_refiner_normal
```

一键运行：

```bash
bash paired_token_reliability/run_latest_three_ablations_gpu012.sh
```

---

## 5. 事件可靠性模块消融

| 编号 | (E_{\mathrm{geo}}) 监督 | (C_{\mathrm{fusion}}) | (C_{\mathrm{refine}}) | 几何收益约束 |
|---|---:|---:|---:|---:|
| R0 |  | 固定 1 | 固定 1 |  |
| R1 | ✓ | ✓ | 固定 1 |  |
| R2 | ✓ | 固定 1 | ✓ |  |
| R3 | ✓ | ✓ | ✓ |  |
| R4 | ✓ | ✓ | ✓ | ✓ |

检查项：

- [ ] 分别可视化 (C_{\mathrm{fusion}}) 和 (C_{\mathrm{refine}})，不能只显示一个 C。
- [ ] 报告预测 C 与 (C_{\mathrm{target}}) 的误差。
- [ ] 报告事件质量、曝光等级与 C 均值之间的关系。
- [ ] 检查 C 是否塌缩到全 0 或全 1。
- [ ] 对比 `cur_event`、(E_{\mathrm{full}}) 和 oracle (E_{\mathrm{geo}})。

建议额外报告 oracle 上界：

```text
E_geo + C=1
```

该结果表示事件分解完全正确时能够达到的几何上限。

---

## 6. Multi-LDR 模块消融

| 编号 | RGB 输入 | 参考监督 | 对齐空间 | 说明 |
|---|---|---|---|---|
| M0 | 单一 LDR | 无 | 无 | 无 Multi-LDR |
| M1 | 多 LDR | ev0 | 完整 token L1 | 普通蒸馏基线 |
| M2 | 多 LDR | ev0 | 几何残差空间 | 本文方式 |
| M3 | 多 LDR | ev0 | 几何残差 + keep | 完整 Multi-LDR |

检查项：

- [ ] 对所有方法使用相同 LDR 对和事件数据。
- [ ] 单独报告 coarse/HDR-base 指标，避免 refiner 掩盖 Multi-LDR 的贡献。
- [ ] 检查 (ev_0) 或正常曝光下是否出现负迁移。
- [ ] 分曝光报告增益，验证过曝越严重时事件补偿是否越明显。
- [ ] 报告 HDR token alignment loss，但不能仅用 alignment loss 证明几何改善。

---

## 7. 事件法向导数与 refiner 消融

| 编号 | Refiner 输入 | HF 深度监督 | 事件 dN 监督 | Final-depth dN 约束 |
|---|---|---:|---:|---:|
| G0 | coarse geometry | ✓ |  |  |
| G1 | event feature + coarse | ✓ |  |  |
| G2 | event dN + coarse | ✓ | ✓ |  |
| G3 | event feature + event dN + coarse | ✓ | ✓ |  |
| G4 | 完整输入 | ✓ | ✓ | ✓ |

检查项：

- [ ] 事件分支预测的是法向导数，不是绝对法向。
- [ ] 法向导数损失只在真实事件支持或有效几何区域内计算。
- [ ] 强高频像素单独加权，防止大量零导数像素稀释监督。
- [ ] 保存预测 dN、GT dN、dN error 三联图。
- [ ] 保存 HF target、HF prediction 和误差图。
- [ ] 报告 `pred_abs / target_abs`，检查 refiner 是否只学习到极小幅值。
- [ ] 报告 coarse→final 的逐像素改善/恶化比例。

---

## 8. 训练调度消融

| 编号 | 初始阶段 | HDR Adapter | C | 后续训练 |
|---|---|---|---|---|
| S0 | 全部分支同时训练 | 立即开放 | 立即开放 | 联合训练 |
| S1 | Geo warm-up | 开放 | 延迟 | A/B 交替 |
| S2 | Refiner-first 1k batch | 冻结 | 固定 1 | 小学习率逐渐开放 |

重点比较：

- [ ] 前 1000 batch 的 HF loss 和 `pred_abs/target_abs`。
- [ ] C 开放前后 final depth 是否突变。
- [ ] HDR Adapter 开放前后 coarse depth 是否漂移。
- [ ] 相同训练 batch 数下比较，避免 S2 因训练更久而占优。
- [ ] 使用独立输出目录，禁止不同调度相互 resume。

Refiner-first 的正确数据路由应为：

```text
前 1000 batch：ev0 RGB + E_geo，C=1
之后：低质量 LDR + cur_event，逐步开放 HDR Adapter 与 C
```

---

## 9. 事件表征消融

| 编号 | 表征 | 时间信息 | Polarity | 插值 |
|---|---|---:|---:|---:|
| E0 | 二值 event support |  | ✓ |  |
| E1 | 累积 event frame |  | ✓ |  |
| E2 | 5-bin voxel | ✓ | 合并 | 线性 |
| E3 | 5-bin polarity voxel | ✓ | 正负分通道 | 线性 |
| E4 | polarity voxel + 时间衰减 | ✓ | 正负分通道 | 线性 |

检查项：

- [ ] 不能把正负事件先二值化后相减，必须保留事件计数/强度。
- [ ] 正负事件使用独立通道，避免同一像素相互抵消。
- [ ] 所有表征使用相同时间窗口和空间分辨率。
- [ ] 报告拖影明显区域和稀疏事件区域的结果。

---

## 10. 论文表格建议

### 主消融表

表格列建议：

```text
Reliability | Multi-LDR | dN Refiner | AbsRel↓ | RMSE↓ | N-Mean↓ | N<11.25↑
```

主表优先放 A0–A4，不要一次堆入所有小消融。

### 可靠性子表

```text
C_fusion | C_refine | Geo supervision | AbsRel↓ | HF error↓ | C error↓
```

### Refiner 子表

```text
Event feature | Event dN | HF loss | Final dN | AbsRel↓ | N-Mean↓ | HF error↓
```

### 分曝光结果

```text
Method | ev0 | ev1 | ev2 | ev5 | ev10 | Average
```

至少对 AbsRel 和法向 Mean 分别提供一张分曝光表或折线图。

---

## 11. 定性可视化清单

每种代表性方法至少保存相同样本的：

- [ ] LDR RGB；
- [ ] event voxel；
- [ ] (C_{\mathrm{fusion}})；
- [ ] (C_{\mathrm{refine}})；
- [ ] coarse depth；
- [ ] final depth；
- [ ] GT depth；
- [ ] |final−GT| 深度误差；
- [ ] coarse normal；
- [ ] final normal；
- [ ] GT normal；
- [ ] depth update；
- [ ] event normal derivative；
- [ ] GT normal derivative；
- [ ] derivative error。

深度图要求：

- coarse、final、GT 使用相同色标；
- update 使用固定物理范围与独立自动拉伸两种图；
- 误差图必须是 final 相对 GT，而不是 final 相对 coarse；
- 所有法向和误差仅显示有效 mask 内区域。

---

## 12. 实验完成检查

- [ ] 主方法和所有消融均完成相同 4 场景、5 曝光测试。
- [ ] 每个实验保存 checkpoint、完整配置、日志和 metrics JSON/CSV。
- [ ] 没有使用测试 GT 为每张图估计独立尺度。
- [ ] RGB-only 与事件方法使用完全一致的评测 mask。
- [ ] coarse 与 final 指标来自同一批样本。
- [ ] 法向统一由深度导出并在有效 mask 内评测。
- [ ] 至少运行 3 个随机种子，报告均值和标准差；若算力不足，主表至少运行 3 次。
- [ ] 检查所有 C、HF、dN 分支确实有梯度且参数已解冻。
- [ ] 检查事件输入来源与实验名称一致，防止 `cur_event`、full 和 geo 静默回退。
- [ ] 论文中的定性样本不能仅选择成功案例，同时展示噪声、稀疏事件和极端过曝场景。

