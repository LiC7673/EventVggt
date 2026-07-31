"""Visualize the forward-difference derivative of an EXR normal map.

The output uses a white-to-purple color map: white means a small normal
variation and dark purple means a large local normal variation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


DEFAULT_NORMAL_DIR = Path(
    r"F:\TreeOBJ\reflective_raw\Bayon Lion_Car_Paint_Midnight\Normal"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", type=Path, default=DEFAULT_NORMAL_DIR)
    parser.add_argument("--index", default="104")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Explicit EXR path. When set, --normal-dir/--index are ignored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <normal-dir>/<EXR stem>_normal_derivative_purple.png",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile used as the upper visualization limit.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--include-silhouette",
        action="store_true",
        help="Also show derivatives crossing foreground/background boundaries.",
    )
    return parser.parse_args()


def locate_exr(directory: Path, index: str) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f"Normal directory does not exist: {directory}")

    direct_names = (f"{index}.exr", f"{int(index):06d}.exr")
    for name in direct_names:
        candidate = directory / name
        if candidate.is_file():
            return candidate

    matches = sorted(
        p for p in directory.glob("*.exr")
        if p.stem == index
        or p.stem.lstrip("0") == index.lstrip("0")
        or p.stem.startswith(f"{index}_")
    )
    if not matches:
        raise FileNotFoundError(
            f"No EXR corresponding to index {index!r} under {directory}"
        )
    return matches[0]


def read_normal_exr(path: Path) -> np.ndarray:
    image = None
    bgr_input = False
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        bgr_input = image is not None
    if image is None:
        try:
            import imageio.v3 as iio
            image = iio.imread(path)
        except (ImportError, OSError, ValueError) as error:
            raise RuntimeError(
                f"Failed to read {path}. Install OpenCV with OpenEXR support "
                "or imageio with an EXR-capable backend."
            ) from error
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an HxWx3 EXR normal map, got {image.shape}")

    # OpenCV reads color channels in BGR order; imageio returns RGB.
    normal = image[..., :3]
    if bgr_input:
        normal = normal[..., ::-1]
    normal = normal.copy()
    finite = np.isfinite(normal).all(axis=-1)
    # Rendered normal EXRs conventionally use an all-zero pixel as background.
    foreground = finite & (np.linalg.norm(normal, axis=-1) > 1.0e-6)
    if finite.any():
        values = normal[finite]
        # Some renderers store normals in [0,1], others directly in [-1,1].
        if values.min() >= -1.0e-4 and values.max() <= 1.0001:
            normal = normal * 2.0 - 1.0
    normal[~foreground] = np.nan
    return normal


def forward_normal_derivative(
    normal: np.ndarray, include_silhouette: bool
) -> tuple[np.ndarray, np.ndarray]:
    length = np.linalg.norm(normal, axis=-1)
    valid = np.isfinite(length) & (length > 1.0e-6)
    unit = normal / np.maximum(length[..., None], 1.0e-6)

    dx = np.zeros_like(unit)
    dy = np.zeros_like(unit)
    dx[:, :-1] = unit[:, 1:] - unit[:, :-1]
    dy[:-1, :] = unit[1:, :] - unit[:-1, :]

    if include_silhouette:
        derivative_valid = valid
    else:
        valid_x = np.zeros_like(valid)
        valid_y = np.zeros_like(valid)
        valid_x[:, :-1] = valid[:, :-1] & valid[:, 1:]
        valid_y[:-1, :] = valid[:-1, :] & valid[1:, :]
        dx *= valid_x[..., None]
        dy *= valid_y[..., None]
        derivative_valid = valid_x | valid_y

    magnitude = np.sqrt(np.square(dx).sum(-1) + np.square(dy).sum(-1))
    magnitude[~derivative_valid] = np.nan
    return magnitude, derivative_valid


def save_visualization(
    magnitude: np.ndarray,
    valid: np.ndarray,
    output: Path,
    percentile: float,
    dpi: int,
) -> None:
    values = magnitude[valid & np.isfinite(magnitude)]
    if values.size == 0:
        raise RuntimeError("No valid normal derivatives were found in the EXR")
    vmax = max(float(np.percentile(values, percentile)), 1.0e-6)

    # NaN values use the colormap's explicitly configured white background.
    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad("white")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    shown = ax.imshow(magnitude, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_facecolor("white")
    ax.set_title(r"GT Normal Derivative Magnitude $\|\nabla_f N\|$", fontsize=14)
    ax.axis("off")
    colorbar = fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label(r"$\|\nabla_f N\|$")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    source = args.input if args.input is not None else locate_exr(
        args.normal_dir, args.index
    )
    source = source.resolve()
    output = args.output or (
        source.parent / f"{source.stem}_normal_derivative_purple.png"
    )
    normal = read_normal_exr(source)
    magnitude, valid = forward_normal_derivative(
        normal, include_silhouette=args.include_silhouette
    )
    save_visualization(
        magnitude, valid, output, args.percentile, args.dpi
    )
    values = magnitude[valid & np.isfinite(magnitude)]
    print(f"Input:  {source}")
    print(f"Output: {output.resolve()}")
    print(
        "Derivative: "
        f"mean={values.mean():.6f}, "
        f"p95={np.percentile(values, 95):.6f}, "
        f"max={values.max():.6f}"
    )


if __name__ == "__main__":
    main()
