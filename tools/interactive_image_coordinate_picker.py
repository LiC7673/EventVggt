#!/usr/bin/env python3
"""Interactively print pixel coordinates clicked on an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_IMAGE = Path(
    r"E:\result\eventvgg\our\split_panels\ev_10_final_normal.png"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Click an image to print its pixel coordinates."
    )
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    image = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image, origin="upper")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_title(
        f"{args.image.name}\n"
        "Left click: print coordinate | Right click: close",
        fontsize=12,
    )
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")

    marker, = ax.plot([], [], marker="+", color="#00FFFF", markersize=15,
                      markeredgewidth=2, linestyle="none")
    annotation = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
        color="white", fontsize=11,
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 5},
    )

    def on_click(event) -> None:
        if event.button == 3:
            plt.close(fig)
            return
        if event.button != 1 or event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < width and 0 <= y < height):
            return

        r, g, b = (int(v) for v in image[y, x])
        print(f"x={x}, y={y}, RGB=({r}, {g}, {b})", flush=True)
        marker.set_data([x], [y])
        annotation.set_text(f"x={x}, y={y}\nRGB=({r}, {g}, {b})")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    print(f"Image: {args.image}")
    print(f"Size: width={width}, height={height}")
    print("Left-click any pixel to print x/y. Right-click to exit.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
