"""Compute an approximate normal gradient from a rendered normal-map PNG.

This script is for normal images produced with the project visualization rule

    RGB = clamp((N + 1) / 2, 0, 1)

and a black invalid/background region.  Because an 8-bit rendered image has
already been quantized and may have been resized or anti-aliased, the result is
an approximate visualization; use the original EXR/tensor for numeric metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Rendered normal PNG/JPG")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <input stem>_normal_gradient_purple.png",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional .npy output containing the floating-point magnitude map.",
    )
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=8.0 / 255.0,
        help="Pixels with max(R,G,B) below this value are treated as background.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Upper color limit percentile over valid gradient pixels.",
    )
    parser.add_argument(
        "--include-silhouette",
        action="store_true",
        help="Include foreground/background transitions in the gradient.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def decode_visualized_normal(
    path: Path, background_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    foreground = np.max(rgb, axis=-1) > background_threshold

    # Inverse of the project's normal visualization: RGB = (N + 1) / 2.
    normal = rgb * 2.0 - 1.0
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid = foreground & np.isfinite(length[..., 0]) & (length[..., 0] > 1.0e-6)
    normal = normal / np.maximum(length, 1.0e-6)
    normal[~valid] = 0.0
    return rgb, normal, valid


def forward_normal_gradient(
    normal: np.ndarray,
    valid: np.ndarray,
    include_silhouette: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same first-order forward difference as the training target."""
    dx = np.zeros_like(normal)
    dy = np.zeros_like(normal)
    dx[:, :-1] = normal[:, 1:] - normal[:, :-1]
    dy[:-1, :] = normal[1:, :] - normal[:-1, :]

    if include_silhouette:
        gradient_valid = valid.copy()
    else:
        valid_x = np.zeros_like(valid)
        valid_y = np.zeros_like(valid)
        valid_x[:, :-1] = valid[:, :-1] & valid[:, 1:]
        valid_y[:-1, :] = valid[:-1, :] & valid[1:, :]
        dx *= valid_x[..., None]
        dy *= valid_y[..., None]
        gradient_valid = valid_x | valid_y

    magnitude = np.sqrt(np.square(dx).sum(-1) + np.square(dy).sum(-1))
    magnitude[~gradient_valid] = np.nan
    return magnitude, gradient_valid


def save_panel(
    rgb: np.ndarray,
    magnitude: np.ndarray,
    valid: np.ndarray,
    output: Path,
    percentile: float,
    dpi: int,
) -> None:
    values = magnitude[valid & np.isfinite(magnitude)]
    if values.size == 0:
        raise RuntimeError(
            "No valid foreground gradient was found. Adjust "
            "--background-threshold if the background is not black."
        )
    vmax = max(float(np.percentile(values, percentile)), 1.0e-6)
    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad("white")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")
    axes[0].imshow(rgb)
    axes[0].set_title("Rendered normal map")
    axes[0].axis("off")

    shown = axes[1].imshow(magnitude, cmap=cmap, vmin=0.0, vmax=vmax)
    axes[1].set_facecolor("white")
    axes[1].set_title(r"Approx. normal gradient $\|\nabla_f N\|$")
    axes[1].axis("off")
    bar = fig.colorbar(shown, ax=axes[1], fraction=0.046, pad=0.025)
    bar.set_label(r"$\|\nabla_f N\|$")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    output = args.output or args.input.with_name(
        f"{args.input.stem}_normal_gradient_purple.png"
    )

    rgb, normal, valid = decode_visualized_normal(
        args.input, args.background_threshold
    )
    magnitude, gradient_valid = forward_normal_gradient(
        normal, valid, args.include_silhouette
    )
    save_panel(
        rgb, magnitude, gradient_valid, output, args.percentile, args.dpi
    )

    if args.raw_output is not None:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.raw_output, magnitude)

    values = magnitude[gradient_valid & np.isfinite(magnitude)]
    print(f"Input:  {args.input.resolve()}")
    print(f"Output: {output.resolve()}")
    if args.raw_output is not None:
        print(f"Raw:    {args.raw_output.resolve()}")
    print(
        "Approximate gradient statistics (do not use as EXR-level metrics): "
        f"mean={values.mean():.6f}, "
        f"p95={np.percentile(values, 95):.6f}, "
        f"max={values.max():.6f}"
    )


if __name__ == "__main__":
    main()
