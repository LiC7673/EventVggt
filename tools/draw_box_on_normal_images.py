#!/usr/bin/env python3
"""Draw the same red rectangle on every *normal* image in a folder."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_point(text: str) -> tuple[int, int]:
    """Accept '(x,y)', 'x,y', or 'x y'."""
    values = re.findall(r"-?\d+", text)
    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid point {text!r}; expected format such as (120,80)"
        )
    return int(values[0]), int(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch draw a 3-pixel red box on images containing 'normal'."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path(r"E:\result\eventvgg\our\split_panels"),
    )
    parser.add_argument("--p1", type=parse_point, help='Top-left point, e.g. "(100,80)"')
    parser.add_argument("--p2", type=parse_point, help='Bottom-right point, e.g. "(300,260)"')
    parser.add_argument("--width", type=int, default=3, help="Rectangle line width")
    parser.add_argument("--color", default="#FF0000", help="Rectangle color")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <input_dir>/boxed_normal",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also process matching images in subdirectories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite source images instead of writing a new folder",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    p1 = args.p1 or parse_point(input("请输入第一个点 (x_1,y_1): ").strip())
    p2 = args.p2 or parse_point(input("请输入第二个点 (x_2,y_2): ").strip())
    x1, x2 = sorted((p1[0], p2[0]))
    y1, y2 = sorted((p1[1], p2[1]))
    if x1 == x2 or y1 == y2:
        raise ValueError("Rectangle width and height must both be greater than zero")
    if args.width < 1:
        raise ValueError("--width must be at least 1")

    iterator = args.input_dir.rglob("*") if args.recursive else args.input_dir.glob("*")
    files = sorted(
        p for p in iterator
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and "normal" in p.stem.lower()
        and "boxed_normal" not in p.parts
    )
    if not files:
        raise RuntimeError(f"No image filename containing 'normal' under {args.input_dir}")

    output_root = args.output_dir or (args.input_dir / "boxed_normal")
    if not args.overwrite:
        output_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    for source in files:
        with Image.open(source) as raw:
            image = raw.convert("RGB")
        width, height = image.size
        if not (0 <= x1 < width and 0 <= x2 < width and
                0 <= y1 < height and 0 <= y2 < height):
            print(
                f"SKIP {source}: box ({x1},{y1})-({x2},{y2}) "
                f"is outside image size {width}x{height}"
            )
            continue

        ImageDraw.Draw(image).rectangle(
            [(x1, y1), (x2, y2)],
            outline=args.color,
            width=args.width,
        )
        if args.overwrite:
            destination = source
        else:
            relative = source.relative_to(args.input_dir)
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        print(f"Saved: {destination}")
        completed += 1

    print(
        f"Done: {completed}/{len(files)} images; "
        f"box=({x1},{y1})-({x2},{y2}), width={args.width}px"
    )


if __name__ == "__main__":
    main()
