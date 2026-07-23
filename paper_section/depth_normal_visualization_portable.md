# 可迁移的深度与法向可视化代码

下面的代码不依赖 EventVGGT，可以直接复制到其他对比方法中。输入均为单张图：

- `pred_depth`: `[H,W]`，模型预测深度；
- `gt_depth`: `[H,W]`，GT 深度；
- `intrinsics`: `[3,3]`，相机内参；
- `valid_mask`: `[H,W]`，评测有效区域；
- `rgb`: `[H,W,3]` 或 `[3,H,W]`，可选。

所有方法应采用相同的 `depth_scale`、mask 和可视化范围。

```python
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def depth_to_normals(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """由深度反投影得到相机坐标系法向。

    Args:
        depth: [..., H, W]
        intrinsics: [..., 3, 3]
    Returns:
        normals: [..., H, W, 3]，分量范围为 [-1, 1]
    """
    depth = depth.float()
    intrinsics = intrinsics.to(device=depth.device, dtype=depth.dtype)
    *batch_dims, height, width = depth.shape

    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    spatial_shape = [1] * len(batch_dims) + [height, width]
    xs = xs.view(*spatial_shape)
    ys = ys.view(*spatial_shape)

    fx = intrinsics[..., 0, 0].view(*batch_dims, 1, 1).clamp_min(1e-6)
    fy = intrinsics[..., 1, 1].view(*batch_dims, 1, 1).clamp_min(1e-6)
    cx = intrinsics[..., 0, 2].view(*batch_dims, 1, 1)
    cy = intrinsics[..., 1, 2].view(*batch_dims, 1, 1)

    points = torch.stack(
        (
            (xs - cx) / fx * depth,
            (ys - cy) / fy * depth,
            depth,
        ),
        dim=-1,
    )

    # 中心差分；叉乘顺序与 EventVGGT 当前评测代码一致。
    dx = points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]
    dy = points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]
    core = F.normalize(torch.cross(dy, dx, dim=-1), dim=-1, eps=1e-6)

    normals = torch.zeros_like(points)
    normals[..., 1:-1, 1:-1, :] = core
    return normals


def normal_to_rgb(normal: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    """将 [-1,1] 法向映射为 RGB，并将无效区域置黑。"""
    image = ((normal.detach().float().cpu() + 1.0) * 0.5).clamp(0, 1)
    mask = valid_mask.detach().bool().cpu()
    image = image * mask.unsqueeze(-1)
    return image.numpy()


def prepare_rgb(rgb: torch.Tensor | np.ndarray) -> np.ndarray:
    rgb = torch.as_tensor(rgb).detach().float().cpu()
    if rgb.ndim != 3:
        raise ValueError(f"RGB must have 3 dimensions, got {tuple(rgb.shape)}")
    if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = rgb.permute(1, 2, 0)
    # 若输入使用 [-1,1] 归一化，则还原至 [0,1]。
    if float(rgb.min()) < 0:
        rgb = (rgb + 1.0) * 0.5
    return rgb.clamp(0, 1).numpy()


@torch.inference_mode()
def save_depth_normal_comparison(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    output_path: str | Path,
    *,
    rgb: torch.Tensor | np.ndarray | None = None,
    depth_scale: float = 2.0,
    title: str = "",
) -> None:
    """保存 RGB、预测/GT 深度、误差和深度导出法向。

    depth_scale 必须是由训练集或验证集确定的固定常数，不能利用当前
    测试图片的 GT 单独拟合。
    """
    pred = pred_depth.detach().float().squeeze() * float(depth_scale)
    gt = gt_depth.detach().float().squeeze().to(pred.device)
    K = intrinsics.detach().float().squeeze().to(pred.device)
    mask = valid_mask.detach().bool().squeeze().to(pred.device)
    mask &= torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0) & (gt > 0)

    if pred.ndim != 2 or gt.ndim != 2 or K.shape != (3, 3):
        raise ValueError(
            f"Expected pred/gt=[H,W], K=[3,3], got "
            f"{tuple(pred.shape)}, {tuple(gt.shape)}, {tuple(K.shape)}"
        )
    if not mask.any():
        raise ValueError("valid_mask contains no valid depth pixels")

    pred_normal = depth_to_normals(pred, K)
    gt_normal = depth_to_normals(gt, K)

    # 预测和 GT 使用同一个绝对深度色标。
    values = torch.cat((pred[mask], gt[mask]))
    depth_min = float(values.min())
    depth_max = float(values.max())

    error = (pred - gt).abs()
    error = torch.where(mask, error, torch.zeros_like(error))
    error_max = max(float(error[mask].max()), 1e-6)

    pred_show = torch.where(mask, pred, torch.zeros_like(pred)).cpu().numpy()
    gt_show = torch.where(mask, gt, torch.zeros_like(gt)).cpu().numpy()
    mask_cpu = mask.cpu()

    panels = []
    if rgb is not None:
        panels.append((prepare_rgb(rgb), "RGB input", None, None, None))
    panels.extend(
        (
            (pred_show, "Predicted depth", "viridis", depth_min, depth_max),
            (gt_show, "GT depth", "viridis", depth_min, depth_max),
            (error.cpu().numpy(), "|Pred - GT|", "magma", 0.0, error_max),
            (normal_to_rgb(pred_normal, mask_cpu),
             "Pred depth-derived normal", None, None, None),
            (normal_to_rgb(gt_normal, mask_cpu),
             "GT depth-derived normal", None, None, None),
        )
    )

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    axes = np.asarray(axes).reshape(-1)

    for axis, (image, panel_title, cmap, vmin, vmax) in zip(axes, panels):
        shown = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(panel_title)
        axis.axis("off")
        if cmap is not None:
            figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes[len(panels):]:
        axis.axis("off")

    if title:
        figure.suptitle(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# 使用示例
if __name__ == "__main__":
    sample = torch.load("comparison_sample.pt", map_location="cpu")
    save_depth_normal_comparison(
        pred_depth=sample["pred_depth"],
        gt_depth=sample["gt_depth"],
        intrinsics=sample["camera_intrinsics"],
        valid_mask=sample["valid_mask"],
        rgb=sample.get("rgb"),
        depth_scale=2.0,
        output_path="visualizations/comparison.png",
        title="Comparison method",
    )
```

## 公平对比要求

1. 所有方法使用同一个 GT mask。
2. 所有预测深度使用相同的固定尺度策略。
3. 预测深度和 GT 深度共享 `vmin/vmax`。
4. 深度误差图始终计算 `|pred_depth-gt_depth|`。
5. 所有法向均由各自预测深度和相同内参导出。
6. 不对每张测试图利用 GT 单独估计尺度，也不对每张深度图单独归一化。

