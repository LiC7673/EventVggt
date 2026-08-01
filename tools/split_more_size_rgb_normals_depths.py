#!/usr/bin/env python3
"""Recursively split RGB, three depth, and three normal test panels."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


REFERENCE_SIZE = (2600, 1950)
TARGET_SIZE = (576, 437)
PANEL_BOXES = {
    "rgb": (20, 156, 596, 593),
    "coarse_depth": (1315, 175, 1842, 573),
    "final_depth": (1963, 175, 2490, 573),
    "gt_depth": (20, 797, 547, 1196),
    "coarse_normal": (1315, 779, 1891, 1213),
    "final_normal": (1963, 779, 2539, 1213),
    "gt_normal": (20, 1401, 596, 1838),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def scaled_box(box: tuple[int, int, int, int], size: tuple[int, int]):
    sx = size[0] / REFERENCE_SIZE[0]
    sy = size[1] / REFERENCE_SIZE[1]
    left, top, right, bottom = box
    return (
        round(left * sx),
        round(top * sy),
        round(right * sx),
        round(bottom * sy),
    )


def fit_canvas(image: Image.Image) -> Image.Image:
    if image.size == TARGET_SIZE:
        return image
    return image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)


def find_sources(input_root: Path, output_root: Path) -> list[Path]:
    output_resolved = output_root.resolve()
    sources = []
    for path in input_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            path.resolve().relative_to(output_resolved)
        except ValueError:
            sources.append(path)
    return sorted(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"E:\result\eventvgg\our\more_size"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: <input-dir>/separated_rgb_normals_depths",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_root / "separated_rgb_normals_depths"
    )
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)

    sources = find_sources(input_root, output_root)
    if not sources:
        raise RuntimeError(f"No PNG/JPG test panels found under {input_root}")

    written = 0
    for source in sources:
        relative = source.relative_to(input_root)
        destination = output_root / relative.parent / source.stem
        destination.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as raw:
            image = raw.convert("RGB")
            for panel_name, reference_box in PANEL_BOXES.items():
                target = destination / f"{panel_name}.png"
                if target.exists() and not args.overwrite:
                    continue
                crop = image.crop(scaled_box(reference_box, image.size))
                fit_canvas(crop).save(target)
                written += 1

    manifest = output_root / "manifest.txt"
    manifest.write_text(
        "\n".join(str(path.relative_to(input_root)) for path in sources) + "\n",
        encoding="utf-8",
    )
    print(f"Input panels: {len(sources)}")
    print(f"Panels written: {written}")
    print(f"Each output: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
