#!/usr/bin/env python3
"""Extract RGB input and final depth-derived normal from RGB-only panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


# Normalized crop boxes measured from the 1950x1300 RGB-only renderer.
NORMALIZED_BOXES = {
    "input_rgb": (20 / 1950, 156 / 1300, 596 / 1950, 593 / 1300),
    "final_normal": (667 / 1950, 752 / 1300, 1243 / 1950, 1187 / 1300),
    "gt_normal": (1314 / 1950, 752 / 1300, 1890 / 1950, 1187 / 1300),
}
TARGET_SIZE = (576, 437)


def pixel_box(box, width: int, height: int):
    left, top, right, bottom = box
    return (
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    )


def center_pad(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.width > size[0] or image.height > size[1]:
        image.thumbnail(size, Image.Resampling.LANCZOS)
    dw, dh = size[0] - image.width, size[1] - image.height
    return ImageOps.expand(
        image,
        border=(dw // 2, dh // 2, dw - dw // 2, dh - dh // 2),
        fill=(0, 0, 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path(r"E:\result\eventvgg\hdreff")
    )
    parser.add_argument("--output-name", default="split_panels")
    args = parser.parse_args()

    sources = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and p.stem in {"ev_0", "ev_1", "ev_2", "ev_5", "ev_10"}
    )
    if not sources:
        raise RuntimeError(f"No ev_0/1/2/5/10 images under {args.input_dir}")

    output_root = args.input_dir / args.output_name
    for name in NORMALIZED_BOXES:
        (output_root / name).mkdir(parents=True, exist_ok=True)

    for source in sources:
        with Image.open(source) as raw:
            image = raw.convert("RGB")
            for panel_name, box in NORMALIZED_BOXES.items():
                crop = image.crop(pixel_box(box, image.width, image.height))
                result = center_pad(crop, TARGET_SIZE)
                result.save(output_root / panel_name / f"{source.stem}.png")
                result.save(output_root / f"{source.stem}_{panel_name}.png")

    print(f"Extracted {len(sources)} figures to {output_root}")
    print(f"Output size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")


if __name__ == "__main__":
    main()
