"""Apply binary masks to LDR images and save transparent RGBA PNGs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SCENE = Path(r"F:\TreeOBJ\reflective_raw\Bearded Man_Ceramic_Glazed_White")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    p.add_argument("--mask-threshold", type=int, default=250)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def apply_mask(image_path: Path, mask_path: Path, threshold: int) -> Image.Image:
    rgb = Image.open(image_path).convert("RGB")
    # Blender mask PNGs are RGBA with an opaque alpha plane. Geometry is stored
    # in RGB, so convert RGB to luminance and apply the requested hard cutoff.
    mask = Image.open(mask_path).convert("RGB")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.NEAREST)
    mask_rgb = np.asarray(mask, dtype=np.uint8)
    # "mask < 250 is transparent": require every RGB component to be at least
    # 250. This prevents colored/antialiased boundary pixels from leaking in.
    opaque = np.all(mask_rgb >= int(threshold), axis=2)
    rgba = np.empty((rgb.height, rgb.width, 4), dtype=np.uint8)
    rgba[..., :3] = np.asarray(rgb, dtype=np.uint8)
    rgba[..., 3] = np.where(opaque, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def main():
    args = parse_args(); scene = args.scene.expanduser().resolve()
    ldr_root, mask_root, output_root = scene / "LDR", scene / "Mask", scene / "vis_event"
    if not ldr_root.is_dir(): raise FileNotFoundError(ldr_root)
    if not mask_root.is_dir(): raise FileNotFoundError(mask_root)
    exposure_dirs = sorted(
        (p for p in ldr_root.iterdir() if p.is_dir() and p.name.startswith("ev_")),
        key=lambda p: float(p.name.removeprefix("ev_")),
    )
    written = skipped = missing = 0
    for exposure in exposure_dirs:
        destination = output_root / exposure.name; destination.mkdir(parents=True, exist_ok=True)
        images = sorted(exposure.glob("*.png"))
        print(f"[{exposure.name}] {len(images)} images", flush=True)
        for image_path in images:
            mask_path, output_path = mask_root / image_path.name, destination / image_path.name
            if output_path.exists() and not args.overwrite:
                skipped += 1; continue
            if not mask_path.is_file():
                print(f"  [missing mask] {mask_path.name}", flush=True)
                missing += 1; continue
            apply_mask(image_path, mask_path, args.mask_threshold).save(output_path, compress_level=4)
            written += 1
    print(f"Done: written={written}, skipped={skipped}, missing={missing}\nOutput: {output_root}")


if __name__ == "__main__": main()
