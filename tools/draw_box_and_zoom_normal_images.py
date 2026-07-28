#!/usr/bin/env python3
"""Draw a detail box and paste a proportional zoom into a bounded region."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_point(text: str) -> tuple[int, int]:
    values = re.findall(r"-?\d+", text)
    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid point {text!r}; expected '(x,y)'"
        )
    return int(values[0]), int(values[1])


def ordered_box(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int, int, int]:
    return min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])


def fit_size(source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int]:
    """Largest proportional size that fits inside target_size."""
    sw, sh = source_size
    tw, th = target_size
    scale = min(tw / sw, th / sh)
    return max(1, round(sw * scale)), max(1, round(sh * scale))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path(r"E:\result\eventvgg\our\split_panels"),
    )
    parser.add_argument("--p1", type=parse_point, default=(343, 226))
    parser.add_argument("--p2", type=parse_point, default=(395, 256))
    parser.add_argument("--p3", type=parse_point, default=(339, 10))
    parser.add_argument("--p4", type=parse_point, default=(549, 167))
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--color", default="#FF0000")
    parser.add_argument("--output-name", default="boxed_zoom_normal")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    if args.width < 1:
        raise ValueError("--width must be >= 1")

    source_box = ordered_box(args.p1, args.p2)
    target_box = ordered_box(args.p3, args.p4)
    sx1, sy1, sx2, sy2 = source_box
    tx1, ty1, tx2, ty2 = target_box
    if sx1 == sx2 or sy1 == sy2 or tx1 == tx2 or ty1 == ty2:
        raise ValueError("Source and target boxes must have non-zero size")

    iterator = args.input_dir.rglob("*") if args.recursive else args.input_dir.glob("*")
    files = sorted(
        p for p in iterator
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and "normal" in p.stem.lower()
        and args.output_name not in p.parts
        and "boxed_normal" not in p.parts
    )
    if not files:
        raise RuntimeError(f"No image filename containing 'normal' under {args.input_dir}")

    output_root = args.output_dir or (args.input_dir / args.output_name)
    output_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    for source in files:
        with Image.open(source) as raw:
            original = raw.convert("RGB")
        image_width, image_height = original.size
        if not (
            0 <= sx1 < sx2 < image_width
            and 0 <= sy1 < sy2 < image_height
            and 0 <= tx1 < tx2 < image_width
            and 0 <= ty1 < ty2 < image_height
        ):
            print(f"SKIP {source}: coordinates exceed {image_width}x{image_height}")
            continue

        # Crop before drawing so the magnified content has no source-box overlay.
        # PIL's right/bottom crop bounds are exclusive, hence +1.
        detail = original.crop((sx1, sy1, sx2 + 1, sy2 + 1))
        fitted = fit_size(detail.size, (tx2 - tx1 + 1, ty2 - ty1 + 1))
        detail = detail.resize(fitted, Image.Resampling.LANCZOS)

        result = original.copy()
        # P3 is the top-left origin, as requested.
        paste_x, paste_y = tx1, ty1
        result.paste(detail, (paste_x, paste_y))
        draw = ImageDraw.Draw(result)
        draw.rectangle(source_box, outline=args.color, width=args.width)
        draw.rectangle(
            (paste_x, paste_y, paste_x + fitted[0] - 1, paste_y + fitted[1] - 1),
            outline=args.color,
            width=args.width,
        )

        relative = source.relative_to(args.input_dir)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.save(destination)
        print(
            f"Saved: {destination} | detail={detail.size}, "
            f"paste=({paste_x},{paste_y})"
        )
        completed += 1

    print(
        f"Done: {completed}/{len(files)} images; "
        f"source=P1{args.p1}->P2{args.p2}; "
        f"target=P3{args.p3}->P4{args.p4}"
    )


if __name__ == "__main__":
    main()
