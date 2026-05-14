证明一：相关性实验

这是最重要的基础实验。

## 实验设置

用 synthetic 数据，因为你需要 GT depth / normal / curvature。

渲染：

```text
单色物体
固定材质
不同凹凸表面
不同运动方向
不同曝光强度
事件流
GT depth / normal
```

表面可以包括：

```text
sphere
bump
dent
ridge
groove
gear-like surface
sinusoidal surface
concave / convex paired shapes
```

然后计算 GT 微分量：

```text
GT ∇ρ
GT Δρ
GT Hρ
GT normal gradient
GT directional curvature κ_v
```

再计算 event cue：

```text
event_abs
event_signed
event_time_surface_gradient
event local orientation
```

## 指标

算相关性：

```text
Corr(event_abs, |∇N|)
Corr(event_abs, |Δρ|)
Corr(event_signed, sign(κ_v))
Corr(event orientation, geometry boundary orientation)
```

更具体一点：

```text
AUC: event_abs 是否能预测 high-curvature regions
Accuracy: event_signed 是否能区分 convex / concave sign
F1: event edge 与 geometry differential edge 的重合
```

你想得到的结论：

```text
event_abs 与几何细节区域高度相关；
event_signed 对凹凸/曲率符号有区分能力；
这种相关性在 RGB 过曝后仍然存在。
```

这一步不是训练模型，只是证明物理 cue 存在。

---