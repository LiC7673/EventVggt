#!/usr/bin/env python3
"""Resize the five EAG3R exposure images to exactly 576x437."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


TARGET_SIZE = (576, 437)
EXPOSURES = {"ev_0", "ev_1", "ev_2", "ev_5", "ev_10"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path(r"E:\result\eventvgg\eag3r")
    )
    parser.add_argument("--output-name", default="resized_576x437")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace source files instead of writing a new subdirectory",
    )
    args = parser.parse_args()

    sources = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and p.stem in EXPOSURES
    )
    if not sources:
        raise RuntimeError(f"No ev_0/1/2/5/10 images under {args.input_dir}")

    output_dir = args.input_dir if args.overwrite else args.input_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        with Image.open(source) as raw:
            image = raw.convert("RGB")
            resized = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        destination = output_dir / f"{source.stem}.png"
        resized.save(destination)
        print(f"{source.name}: {image.size} -> {resized.size} | {destination}")

    print(f"Done: {len(sources)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
