#!/usr/bin/env python3
"""Extract RGB, coarse-normal, and final-normal panels from rendered grids.

The renderer used for these figures produces a 2600x1950, 4-column layout.
Crop boxes are stored as normalized coordinates so the script also works when
the complete figure is saved at another resolution with the same layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


# Measured from the existing rendering layout, excluding titles and axes.
NORMALIZED_BOXES = {
    "input_rgb": (20 / 2600, 156 / 1950, 596 / 2600, 593 / 1950),
    "coarse_normal": (1315 / 2600, 779 / 1950, 1891 / 2600, 1213 / 1950),
    "final_normal": (1963 / 2600, 779 / 1950, 2539 / 2600, 1213 / 1950),
    "gt_normal": (20 / 2600, 1401 / 1950, 596 / 2600, 1838 / 1950),
}


def pixel_box(box: tuple[float, float, float, float], width: int, height: int):
    left, top, right, bottom = box
    return (
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    )


def fit_same_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Center an image without rescaling and pad it to a common black canvas."""
    if image.size == size:
        return image
    delta_w = size[0] - image.width
    delta_h = size[1] - image.height
    return ImageOps.expand(
        image,
        border=(
            delta_w // 2,
            delta_h // 2,
            delta_w - delta_w // 2,
            delta_h - delta_h // 2,
        ),
        fill=(0, 0, 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(r"E:\result\eventvgg\our"))
    parser.add_argument("--output-name", default="split_panels")
    args = parser.parse_args()

    sources = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and p.stem in {"ev_0", "ev_1", "ev_2", "ev_5", "ev_10"}
    )
    if not sources:
        raise RuntimeError(f"No ev_0/1/2/5/10 visualization images under {args.input_dir}")

    output_root = args.input_dir / args.output_name
    for panel_name in NORMALIZED_BOXES:
        (output_root / panel_name).mkdir(parents=True, exist_ok=True)

    for source in sources:
        with Image.open(source) as raw:
            image = raw.convert("RGB")
            crops = {
                name: image.crop(pixel_box(box, image.width, image.height))
                for name, box in NORMALIZED_BOXES.items()
            }
        common_size = (
            max(crop.width for crop in crops.values()),
            max(crop.height for crop in crops.values()),
        )
        for panel_name, crop in crops.items():
            result = fit_same_canvas(crop, common_size)
            result.save(output_root / panel_name / f"{source.stem}.png", quality=100)
            # Flat filenames make paper-figure assembly convenient.
            result.save(output_root / f"{source.stem}_{panel_name}.png", quality=100)

    print(f"Extracted {len(sources)} figures to {output_root}")
    print(f"Each extracted panel has size {common_size[0]}x{common_size[1]}")


if __name__ == "__main__":
    main()
