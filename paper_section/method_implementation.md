# 当前方法的实际实现说明

本文档只描述当前方法如何实现，不讨论研究动机、创新性、相关工作或实验结论。

## 1. 输入与输出

每个训练样本包含多个相机视角。第 $i$ 个视角包含：

- 当前曝光的 RGB 图像 $I_i$；
- 同一场景的参考曝光 RGB 图像 $I_i^{ref}$，仅在多 LDR 训练时使用；
- 当前时间窗口内的完整事件流 $E_i^{full}$；
- 受控生成的几何事件流 $E_i^{geo}$，仅在训练时使用；
- 相机内参 $K_i$ 和相机位姿；
- GT depth、GT point map，以及由 GT depth 计算的 GT normal。

模型输出：

- 最终深度 $D_i$；
- 最终法向 $N_i$；
- point map $P_i$；
- 相机位姿 $T_i$；
- 事件几何贡献图 $C_i$；
- 事件法向残差、法向置信度和逐步深度更新。

推理阶段只需要当前曝光 RGB、对应事件流和相机输入，不需要参考曝光、$E^{geo}$、GT 或训练期 bridge。

## 2. 事件流体素化

事件表示为：

\[
e_m=(x_m,y_m,t_m,p_m), \qquad p_m\in\{+1,-1\}.
\]

当前默认使用 $B=5$ 个时间 bin，并保留正负极性。时间首先归一化到：

\[
\bar t_m=(B-1)\frac{t_m-t_0}{t_1-t_0}.
\]

每个事件通过线性插值写入相邻时间 bin：

\[
V_{b,p}(x,y)=
\sum_m
\mathbf 1[(x_m,y_m)=(x,y)]
\mathbf 1[p_m=p]
\max(0,1-|b-\bar t_m|).
\]

最终事件张量为：

\[
V\in\mathbb R^{2B\times H\times W}.
\]

前 $B$ 个通道为正事件，后 $B$ 个通道为负事件。正负事件不会在 dataloader 中直接相加，因此同一像素的正负事件不会因为求和而提前抵消。

空间缩放后会过滤由插值产生的极小非零尾部，避免把数值接近零的插值残留当成真实事件 support。

## 3. RGB 粗几何

RGB 图像首先进入预训练的 RGB VGGT 主干：

\[
Z_i,D_i^c,P_i^c,T_i
=G_{rgb}(I_1,\ldots,I_N).
\]

其中：

- $Z_i$ 为 RGB 多视角 token；
- $D_i^c$ 为 coarse depth；
- $P_i^c$ 为 coarse point map；
- $T_i$ 为相机预测。

事件流不与 RGB 图像在 VGGT 输入端拼接，也不进入 RGB Aggregator。coarse normal 由 coarse depth 和相机内参计算：

\[
N_i^c=\Pi(D_i^c,K_i),
\]

其中 $\Pi$ 为可微分的 depth-to-normal 运算。

训练开始时先进行深度尺度对齐。当前实验的前 1000 个训练 step 只允许深度尺度参数起作用，事件法向残差和事件深度细化保持为零，防止事件分支在 RGB 深度尺度尚未稳定时吸收全局尺度误差。

## 4. 事件几何贡献图

ContributionNet 的输出为像素级贡献图：

\[
C_i=\sigma\left(
R_\theta(V_i^{full},I_i,Z_i,D_i^c,N_i^c)
\right),
\qquad C_i\in[0,1]^{H\times W}.
\]

$C_i$ 表示完整事件流中有多少信息应该进入几何分支。它不是事件存在掩码，也不是物体分割图。

### 4.1 受控事件监督

完整事件与几何事件的时间通道质量分别为：

\[
A_i^{full}(x,y)=\sum_c|V_{i,c}^{full}(x,y)|,
\]

\[
A_i^{geo}(x,y)=\sum_c|V_{i,c}^{geo}(x,y)|.
\]

物理贡献监督为：

\[
C_i^{gt}(x,y)=
\operatorname{clip}\left(
\frac{A_i^{geo}(x,y)}{A_i^{full}(x,y)+\epsilon},0,1
\right).
\]

该目标描述的是几何事件在完整事件中的比例，而不是 depth edge 或 binary geometry mask。

### 4.2 多 LDR 一致性

同一场景、相机和事件时间窗口下，使用两个不同曝光 RGB 分别预测：

\[
C_i^a=R_\theta(V_i^{full},I_i^a,Z_i^a,D_i^{c,a},N_i^{c,a}),
\]

\[
C_i^b=R_\theta(V_i^{full},I_i^b,Z_i^b,D_i^{c,b},N_i^{c,b}).
\]

二者通过一致性损失约束：

\[
L_{pair}=
\frac{\sum M_iS_i|C_i^a-\operatorname{sg}(C_i^b)|}
{\sum M_iS_i+\epsilon}.
\]

其中 $M_i$ 为有效几何区域，$S_i$ 为真实事件 support，参考曝光分支使用 stop-gradient。

贡献图还使用 decomposition regression 和 geometry contribution ranking：

\[
L_C=
\lambda_{dec}L_{dec}
+\lambda_{pair}L_{pair}
+\lambda_{rank}L_{rank}.
\]

### 4.3 事件选择

完整事件流按照贡献图进行连续加权：

\[
\widetilde V_i=C_i\odot V_i^{full}.
\]

这里使用软权重，不把 $C_i$ 二值化。加权发生在事件编码之前。

## 5. 事件特征编码

加权后的事件体进入独立事件编码器：

\[
F_i^e=E_\phi(\widetilde V_i).
\]

编码过程保留时间 bin 和 polarity 信息，并生成与 RGB 几何解码器不同尺度对应的事件特征。

事件特征通过独立 projection 和 geometry adapter 进入 depth/point 解码路径。原始 VGGT RGB token 仍作为基础表示，事件分支只提供额外几何更新。

## 6. 事件分支学习的几何量

事件分支不直接从零预测完整绝对法向，也不把事件图像直接当作法向图。它预测相对于 coarse normal 的局部法向残差：

\[
\Delta N_i^e=	anh\left(
H_{\Delta N}([F_i^e,\log D_i^c,N_i^c])
\right).
\]

同时预测独立的法向可信度：

\[
Q_i^{learned}=\sigma\left(
H_Q([F_i^e,\log D_i^c,N_i^c])
\right).
\]

法向可信度与事件贡献图连接：

\[
Q_i^N=C_i\odot Q_i^{learned}.
\]

最终事件条件法向为：

\[
\widetilde N_i=
\operatorname{normalize}left(
N_i^c+Q_i^N\odot\Delta N_i^e
\right).
\]

因此，$C_i$ 控制事件是否具有几何贡献，$Q_i^{learned}$ 控制当前法向残差是否可信，两者含义不同。

## 7. 事件法向监督

GT normal 由 GT depth 计算：

\[
N_i^{gt}=\Pi(D_i^{gt},K_i).
\]

真实事件 support 为：

\[
S_i(x,y)=
M_i(x,y)\mathbf 1\left[
\sum_c|V_{i,c}^{full}(x,y)|>0
\right].
\]

事件法向方向损失为：

\[
L_{EN}=
\frac{\sum S_i(1-\langle\widetilde N_i,N_i^{gt}\rangle)}
{\sum S_i+\epsilon}.
\]

事件法向导数损失为：

\[
L_{EN\nabla}=
\frac{
\sum S_i^\nabla
\sum_{d\in\{x,y\}}
\left\|
\nabla_d\widetilde N_i-
\nabla_dN_i^{gt}
\right\|_1}
{\sum S_i^\nabla+\epsilon}.
\]

$S_i^\nabla$ 要求当前像素和计算有限差分所需的相邻像素均存在有效事件。完全没有事件的区域不参与事件法向和事件法向导数监督。

事件 support 只用于确定事件法向监督位置，不用于裁剪最终的 depth update。

## 8. 法向引导的稠密深度细化

深度细化以 RGB coarse depth 为锚点：

\[
D_i^{(0)}=D_i^c.
\]

第 $k$ 次迭代首先从当前深度计算法向：

\[
N_i^{(k)}=\Pi(D_i^{(k)},K_i).
\]

深度细化模块读取：

- 事件特征 $F_i^e$；
- 当前 log-depth；
- coarse log-depth；
- 当前深度法向 $N_i^{(k)}$；
- 事件条件法向 $\widetilde N_i$；
- 法向可信度 $Q_i^N$。

每一步预测有界 log-depth 更新：

\[
\delta_i^{(k)}=
s\tanh\left(
\frac{H_D(
[F_i^e,\log D_i^{(k)},\log D_i^c,
N_i^{(k)},\widetilde N_i,Q_i^N])}{s}
\right).
\]

深度按乘性形式更新：

\[
D_i^{(k+1)}=D_i^{(k)}\odot\exp(\delta_i^{(k)}).
\]

最终更新相对 coarse depth 的总幅度受到限制：

\[
\log\frac{D_i}{D_i^c}
\in[\log(1-r),\log(1+r)].
\]

当前 confidence-refine 实验默认：

- 迭代次数 $K=3$；
- 单步 log-depth 更新上限 $s=0.05$；
- 总相对深度更新范围 $r=0.50$。

这些限制是幅度上限，不是把所有更新固定缩放到 5% 或 50%。模型仍可以在允许范围内预测任意较小更新。

深度更新是完整目标区域上的稠密预测：

\[
D_i=H_D(D_i^c,F_i^e,\widetilde N_i,Q_i^N,K_i).
\]

最终深度不乘事件 support，因此无事件区域仍可以通过卷积上下文、coarse depth 和法向一致性保持连续。

最终法向重新由最终深度计算：

\[
N_i=\Pi(D_i,K_i).
\]

## 9. Point map 更新

Point map 分支使用 RGB token、事件多尺度特征和贡献图：

\[
P_i=H_P(Z_i,F_i^e,C_i).
\]

事件特征通过 point geometry adapter 写入 point decoder。Point map 不是简单地由可视化深度图复制得到，而是保留独立的 point prediction head 和 point supervision。

## 10. 深度—法向一致性

事件条件法向用于约束最终深度导出的法向：

\[
L_{DN}=
\frac{
\sum M_iQ_i^N
\left(1-
\left\langle
\Pi(D_i,K_i),
\operatorname{sg}(\widetilde N_i)
\right\rangle
\right)}
{\sum M_iQ_i^N+\epsilon}.
\]

该项对事件法向使用 stop-gradient，使梯度主要推动深度细化结果跟随已经学习到的事件局部法向，而不是让两个分支同时移动到任意中间解。

## 11. 几何损失

最终几何损失由以下部分组成：

\[
L_{geo}=
\lambda_D L_D+
\lambda_N L_N+
\lambda_P L_P+
\lambda_{DG}L_{DG}+
\lambda_{DC}L_{DC}+
\lambda_{Grid}L_{Grid}.
\]

其中：

- $L_D$：最终 depth 与 GT depth 的有效像素损失；
- $L_N$：最终 depth-derived normal 与 GT normal 的余弦损失；
- $L_P$：point map 损失；
- $L_{DG}$：log-depth 一阶梯度匹配；
- $L_{DC}$：log-depth 二阶曲率匹配；
- $L_{Grid}$：patch 边界梯度匹配。

这些导数项匹配 GT depth 的真实导数，不是单纯将预测深度平滑为常数。

深度更新还使用较弱的 TV/curvature regularization，用于抑制无监督高频噪声。该正则不再包含强 magnitude penalty，避免将所有事件深度更新压缩到 $10^{-4}$ 量级。

## 12. 总损失

当前方法的统一训练目标为：

\[
L=
L_{geo}
+L_C
+\lambda_{EN}L_{EN}
+\lambda_{EN\nabla}L_{EN\nabla}
+\lambda_{DN}L_{DN}
+\lambda_{reg}L_{update-reg}.
\]

各项只在对应分支启用时参与反向传播。

## 13. 训练流程

### 13.1 尺度预热

训练开始的前 1000 step：

- 使用 RGB coarse depth 对齐数据集深度尺度；
- 事件法向 residual 为零；
- 事件 depth refinement 为零；
- 新增事件头仍保留在 DDP 计算图中，但不改变最终预测。

### 13.2 几何事件初始化

使用 $V^{geo}$ 初始化事件编码器和 depth/point geometry adapter，使事件分支先学习由受控几何运动产生的有效几何响应。

### 13.3 完整事件贡献学习

恢复 $V^{full}$，训练 ContributionNet：

\[
\widetilde V=C\odot V^{full}.
\]

这一阶段使用 controlled-event contribution target、multi-LDR consistency 和 ranking supervision。

### 13.4 事件法向与深度耦合

尺度预热结束后逐渐增大事件细节、事件法向和 depth-normal consistency 权重，而不是从第一个 step 直接使用最大权重。

事件分支学习：

\[
V\rightarrow F^e\rightarrow
\Delta N^e,Q^N\rightarrow
\widetilde N\rightarrow D.
\]

### 13.5 联合校准

最后以较小学习率联合更新 ContributionNet、事件编码器、法向残差头、深度细化头和 depth/point adapter。RGB Aggregator 和不需要更新的预训练模块保持冻结或使用更小学习率。

## 14. 推理流程

推理时执行：

\[
I\rightarrow Z,D^c,P^c,T,
\]

\[
(I,V^{full},Z,D^c,N^c)\rightarrow C,
\]

\[
C\odot V^{full}\rightarrow F^e,
\]

\[
(F^e,D^c,N^c,C)\rightarrow
\Delta N^e,Q^N,\widetilde N,
\]

\[
(D^c,F^e,\widetilde N,Q^N)\rightarrow D,N,P.
\]

推理阶段不使用：

- 参考曝光 RGB；
- $V^{geo}$；
- contribution GT；
- GT depth/normal；
- saturation bridge；
- 训练阶段的 decomposition target。

## 15. 当前实现中明确不采用的操作

- 不把正负事件直接累加成单通道后再输入模型；
- 不把 raw event visualization 当作 normal prediction；
- 不从事件分支直接预测一张与 coarse normal 无关的绝对法向图；
- 不使用 `normalize(coarse_normal + ungated_delta_normal)`；
- 不将 depth update 硬乘 event support；
- 不在零事件区域施加事件法向导数监督；
- 不让事件 residual 在尺度预热阶段补偿全部全局深度尺度误差；
- 不通过单独的可视化归一化结果判断 depth update 的真实数值幅度。
